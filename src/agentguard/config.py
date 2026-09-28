"""User configuration: custom models, their prices, aliases and exceptions.

The bundled table in :mod:`agentguard.pricing` is a best-effort snapshot of
*public list prices*. It can never know about your fine-tune, your regional
endpoint, your negotiated rate, or the alias your gateway reports. This module is
how a user answers "what does this model cost?" without patching the library or
writing Python:

.. code-block:: json

   {
     "version": 1,
     "models": {
       "my-finetune-v3": {"input": 3.0, "output": 12.0, "cached_input": 0.3},
       "acme-local-7b": [0.05, 0.08]
     },
     "aliases": {
       "acme/fast": "claude-3-5-haiku"
     },
     "disable": ["gpt-4"]
   }

Three keys, three questions:

``models``
    What does this model cost? A price is USD per 1M tokens, written either as an
    object (``input`` / ``output`` / ``cached_input``) or as a short array
    (``[input, output]``). A name that matches a bundled model reprices that model.
``aliases``
    What is this name really? The alias key is matched against the model string
    exactly (case-insensitively), before any other interpretation, so it can
    redirect a gateway's ``acme/fast`` to a model that has a price.
``disable``
    Which bundled prices do I not trust? A disabled model becomes *unpriced*
    rather than wrong, which is the honest outcome: it is reported, counted, and
    excluded from the budget.

The file is found in this order:

1. ``$AGENTGUARD_CONFIG`` — an explicit path. Set it to ``none`` (or ``off``,
   ``0``, ``false``, ``no``) to ignore every config file.
2. ``%APPDATA%\\agentguard\\pricing.json`` on Windows,
   ``$XDG_CONFIG_HOME/agentguard/pricing.json`` (default ``~/.config/...``)
   elsewhere — the per-user file, always read.
3. ``agentguard.json`` (or ``.agentguard.json``) in the working directory or the
   nearest parent — the project-level file, read **only when explicitly trusted**
   with ``$AGENTGUARD_TRUST_PROJECT_CONFIG=1`` (see below).

A project-level file travels with the repository it sits in, so it is written by
whoever wrote that repository. Reading it by default would make "clone this repo
and run your agent in it" a documented way to reprice every model to nearly
nothing, or to ``disable`` the expensive ones so their calls stop counting against
the budget — a guard bypass performed with a data file. So it is skipped unless
:func:`project_config_trusted` says otherwise, and a skipped file is reported once
per process rather than silently ignored.

``Guard`` picks this up automatically; ``Guard(use_config=False)`` opts out, and
anything passed to ``Guard(pricing=...)`` in code still wins over the file. The
file is read when a guard is constructed rather than cached at import, so editing
it takes effect on the next guard: no restart, and no reload thread.

Nothing here is guessed. A malformed file, an unknown key, a negative or ``NaN``
rate, an alias pointing at a model with no price, or a ``disable`` entry that
matches nothing raises :class:`~agentguard.GuardConfigError` with the file path
in the message — a config typo must never quietly change what a budget means.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import GuardConfigError
from .pricing import (
    Price,
    PriceTable,
    _coerce_price,
    normalize_model_key,
)

__all__ = [
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAMES",
    "CONFIG_TRUST_ENV_VAR",
    "CONFIG_VERSION",
    "PricingConfig",
    "Removal",
    "config_paths",
    "ignored_project_config",
    "initialize_config",
    "load_config",
    "merge_configs",
    "parse_config",
    "project_config_path",
    "project_config_trusted",
    "read_config_file",
    "remove_entry",
    "set_alias",
    "set_disabled",
    "set_model_price",
    "user_config_path",
    "write_config_file",
]


#: Environment variable holding an explicit config path, or ``none``/``off``/``0``
#: to disable config loading entirely.
CONFIG_ENV_VAR = "AGENTGUARD_CONFIG"

#: Set this to a truthy value to let a config file inside the project tree be read.
#: See :func:`project_config_trusted`.
CONFIG_TRUST_ENV_VAR = "AGENTGUARD_TRUST_PROJECT_CONFIG"

#: Schema version written by the CLI and understood by this module.
CONFIG_VERSION = 1

#: Project-level file names, checked in order in each directory.
CONFIG_FILENAMES = ("agentguard.json", ".agentguard.json")

#: Values of :data:`CONFIG_ENV_VAR` that mean "load no config file at all".
_OFF_VALUES = frozenset({"none", "off", "0", "false", "no", "disable", "disabled"})

#: Values of :data:`CONFIG_TRUST_ENV_VAR` that mean "yes, read the project file".
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

#: Project configs already reported as ignored, so the warning fires once each.
_warned_untrusted: set[str] = set()

_TOP_LEVEL_KEYS = ("version", "models", "aliases", "disable")


#: The starter document written by ``agentguard config init``.
def _template() -> dict[str, Any]:
    """A fresh starter document. Never a shared constant: callers mutate it."""
    return {"version": CONFIG_VERSION, "models": {}, "aliases": {}, "disable": []}


class _Off:
    """Sentinel type for ``AGENTGUARD_CONFIG=none``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<config disabled>"


_OFF = _Off()


# --------------------------------------------------------------------------- #
# Validated configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PricingConfig:
    """A validated pricing configuration.

    Built by :func:`load_config` or :func:`parse_config`; handed to
    :meth:`agentguard.PriceTable.from_config`. ``models`` and ``aliases`` keep the
    names exactly as the user wrote them, because that is what an error message
    or a config listing should echo back.
    """

    models: Mapping[str, Price] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    disable: frozenset[str] = frozenset()
    sources: tuple[Path, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when nothing at all was configured."""
        return not (self.models or self.aliases or self.disable)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, suitable for ``--json`` output.

        >>> parse_config({"models": {"m": [1.0, 2.0]}}).as_dict()["models"]
        {'m': {'input_per_1m': 1.0, 'output_per_1m': 2.0, 'cached_input_per_1m': None}}
        """
        return {
            "models": {name: price.as_dict() for name, price in self.models.items()},
            "aliases": dict(self.aliases),
            "disable": sorted(self.disable),
            "sources": [str(source) for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class Removal:
    """What :func:`remove_entry` actually deleted from a config file.

    >>> Removal(models=("mine",), aliases=("fast",)).changed
    True
    >>> Removal().changed
    False
    """

    models: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """True when the file was modified."""
        return bool(self.models or self.aliases or self.disabled)

    def describe(self) -> str:
        """A short human-readable summary, e.g. ``model 'mine' and alias 'fast'``."""
        parts = [
            *[f"model {name!r}" for name in self.models],
            *[f"alias {name!r}" for name in self.aliases],
            *[f"disable entry {name!r}" for name in self.disabled],
        ]
        if not parts:
            return "nothing"
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def parse_config(data: object, *, source: Path | str | None = None) -> PricingConfig:
    """Validate an already-decoded config document.

    >>> config = parse_config({"models": {"mine": {"input": 3, "output": 12}}})
    >>> round(config.models["mine"].output_per_1m, 2)
    12.0
    >>> parse_config({"models": {"mine": [1, 2]}, "aliases": {"fast": "mine"}}).aliases
    {'fast': 'mine'}
    >>> parse_config({}).is_empty
    True
    """
    return _parse_document(data, source=source, buildable=True)


def _parse_document(data: object, *, source: Path | str | None, buildable: bool) -> PricingConfig:
    """Shared parser. ``buildable=False`` skips whole-document checks.

    Reading a file *for editing* deliberately stops after the per-field checks:
    a config whose alias target disappeared in an upgrade is exactly the file
    ``agentguard config remove`` has to be able to repair. Writing still goes
    through :func:`parse_config`, so a broken document is never written back.
    """
    where = f"{source}: " if source is not None else ""
    if not isinstance(data, Mapping):
        raise GuardConfigError(f"{where}config must be a JSON object, got {type(data).__name__}")

    unknown = [key for key in data if key not in _TOP_LEVEL_KEYS]
    if unknown:
        raise GuardConfigError(
            f"{where}unknown key(s) {', '.join(repr(str(k)) for k in unknown)}; "
            f"expected {', '.join(_TOP_LEVEL_KEYS)}"
        )

    version = data.get("version", CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise GuardConfigError(f"{where}'version' must be an integer, got {version!r}")
    if version > CONFIG_VERSION:
        raise GuardConfigError(
            f"{where}config version {version} was written for a newer agentguard "
            f"(this one understands version {CONFIG_VERSION})"
        )
    if version < 1:
        raise GuardConfigError(f"{where}'version' must be at least 1, got {version}")

    models = _parse_models(data.get("models", {}), where)
    aliases = _parse_aliases(data.get("aliases", {}), where)
    disable = _parse_disable(data.get("disable", ()), where)

    both = sorted(set(aliases) & set(models))
    if both:
        raise GuardConfigError(
            f"{where}{', '.join(repr(name) for name in both)} is given both a price "
            f"and an alias; keep it in one of 'models' or 'aliases'"
        )

    config = PricingConfig(
        models=models,
        aliases=aliases,
        disable=disable,
        sources=(Path(source),) if source is not None else (),
    )
    if buildable:
        _check_buildable(config, where)
    return config


def _parse_models(raw: object, where: str) -> dict[str, Price]:
    if not isinstance(raw, Mapping):
        raise GuardConfigError(f"{where}'models' must be an object, got {type(raw).__name__}")
    models: dict[str, Price] = {}
    normalized: dict[str, str] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise GuardConfigError(f"{where}'models' has an empty model name: {name!r}")
        try:
            price = _coerce_price(name, spec)
        except GuardConfigError as exc:
            raise GuardConfigError(f"{where}{exc}") from exc
        key = normalize_model_key(name)
        previous = normalized.get(key)
        if previous is not None:
            raise GuardConfigError(
                f"{where}{previous!r} and {name!r} both resolve to the model name "
                f"{key!r}; give them distinct names"
            )
        normalized[key] = name
        models[name] = price
    return models


def _parse_aliases(raw: object, where: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise GuardConfigError(f"{where}'aliases' must be an object, got {type(raw).__name__}")
    aliases: dict[str, str] = {}
    for name, target in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise GuardConfigError(f"{where}'aliases' has an empty model name: {name!r}")
        if not isinstance(target, str) or not target.strip():
            raise GuardConfigError(
                f"{where}alias {name!r} must point at a model name, got {target!r}"
            )
        aliases[name] = target.strip()
    return aliases


def _parse_disable(raw: object, where: str) -> frozenset[str]:
    # An object is rejected rather than iterated: ``{"gpt-4": true}`` reads as a
    # list to a human, and silently accepting it would hide a schema mistake.
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Iterable):
        raise GuardConfigError(f"{where}'disable' must be an array of model names, got {raw!r}")
    disabled: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name.strip():
            raise GuardConfigError(f"{where}'disable' has an empty model name: {name!r}")
        disabled.append(name)
    return frozenset(disabled)


def _check_buildable(config: PricingConfig, where: str) -> None:
    """Prove the config can actually build a table, with the file in the message.

    Building the real :class:`~agentguard.PriceTable` is the cheapest way to
    check everything at once: unknown ``disable`` entries, aliases pointing at
    models with no price, and alias cycles all surface here rather than at the
    first LLM call.
    """
    try:
        PriceTable.from_config(config)
    except GuardConfigError as exc:
        raise GuardConfigError(f"{where}{exc}") from exc


def merge_configs(configs: Iterable[PricingConfig]) -> PricingConfig:
    """Merge configs in order; later entries win, ``disable`` is combined.

    Models are merged on their *normalized* name, so a project file that
    redefines a user-level model replaces it instead of creating a near-duplicate
    that only one of the two lookups would find.
    """
    models: dict[str, Price] = {}
    aliases: dict[str, str] = {}
    disabled: set[str] = set()
    sources: list[Path] = []
    seen: dict[str, str] = {}

    for config in configs:
        for name, price in config.models.items():
            key = normalize_model_key(name)
            previous = seen.get(key)
            if previous is not None and previous != name:
                raise GuardConfigError(
                    f"{previous!r} and {name!r} both resolve to the model name {key!r}; "
                    f"give them distinct names"
                )
            seen[key] = name
            models[name] = price
        aliases.update(config.aliases)
        disabled |= set(config.disable)
        for source in config.sources:
            if source not in sources:
                sources.append(source)

    return PricingConfig(
        models=models,
        aliases=aliases,
        disable=frozenset(disabled),
        sources=tuple(sources),
    )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def user_config_path(*, env: Mapping[str, str] | None = None, windows: bool | None = None) -> Path:
    """The per-user config path, whether or not it exists.

    >>> user_config_path(env={"XDG_CONFIG_HOME": "/tmp/cfg"}, windows=False).as_posix()
    '/tmp/cfg/agentguard/pricing.json'
    >>> user_config_path(env={"APPDATA": "C:/Users/x/AppData/Roaming"}, windows=True).name
    'pricing.json'
    """
    environ = os.environ if env is None else env
    on_windows = os.name == "nt" if windows is None else windows
    if on_windows:
        base = environ.get("APPDATA") or environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "agentguard" / "pricing.json"
    xdg = environ.get("XDG_CONFIG_HOME")
    home = environ.get("HOME")
    root = Path(xdg) if xdg else (Path(home) if home else Path.home()) / ".config"
    return root / "agentguard" / "pricing.json"


def project_config_path(start: str | Path | None = None) -> Path | None:
    """The nearest project-level config at or above ``start``, if any.

    ``start`` is a directory (a file is accepted and its parent used); when it is
    omitted the current working directory is the starting point. The walk stops at
    the first directory containing one of :data:`CONFIG_FILENAMES`, which is what
    makes a monorepo behave: running from a subdirectory finds the config at the
    root instead of quietly using none.
    """
    here = Path(start).absolute() if start is not None else Path.cwd()
    if here.is_file():
        here = here.parent
    for directory in (here, *here.parents):
        for filename in CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def _explicit_path(environ: Mapping[str, str]) -> Path | _Off | None:
    """``$AGENTGUARD_CONFIG`` as a path, the :data:`_OFF` sentinel, or ``None``."""
    raw = (environ.get(CONFIG_ENV_VAR) or "").strip()
    if not raw:
        return None
    if raw.lower() in _OFF_VALUES:
        return _OFF
    return Path(raw).expanduser()


def project_config_trusted(*, env: Mapping[str, str] | None = None) -> bool:
    """Whether a config file inside the project tree may be read.

    >>> project_config_trusted(env={CONFIG_TRUST_ENV_VAR: "1"})
    True
    >>> project_config_trusted(env={})
    False
    """
    environ = os.environ if env is None else env
    return (environ.get(CONFIG_TRUST_ENV_VAR) or "").strip().lower() in _TRUTHY_VALUES


def ignored_project_config(
    *, start: str | Path | None = None, env: Mapping[str, str] | None = None
) -> Path | None:
    """The project config that exists but is not trusted, if any."""
    environ = os.environ if env is None else env
    if _explicit_path(environ) is not None or project_config_trusted(env=environ):
        return None
    return project_config_path(start)


def _warn_ignored_project_config(path: Path) -> None:
    """Say once, loudly, that a file that looks like config was not read."""
    key = str(path)
    if key in _warned_untrusted:
        return
    _warned_untrusted.add(key)
    warnings.warn(
        f"agentguard is ignoring the project config at {path}: a config file inside "
        f"a source tree is written by whoever wrote that repository, so it could "
        f"lower prices or disable models and quietly weaken the budget. Set "
        f"{CONFIG_TRUST_ENV_VAR}=1 to use it, or pass Guard(config_path=...). "
        f"Bundled prices are in effect.",
        RuntimeWarning,
        stacklevel=4,
    )


def config_paths(
    *,
    start: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    windows: bool | None = None,
) -> tuple[Path, ...]:
    """Every config file that exists and would be loaded, in load order.

    An explicit ``$AGENTGUARD_CONFIG`` replaces discovery — it is "use exactly
    this file", not "also read this file". Setting it to ``none``/``off``/``0``
    disables config loading. An explicit path that does not exist yields ``()``;
    :func:`load_config` is the one that turns that into an error.

    The per-user file is always read. A file found in the project tree is not,
    unless :func:`project_config_trusted` says so: it travels with the repository,
    so reading it by default would let whoever wrote the repository reprice the
    models this guard bills.

    >>> config_paths(env={CONFIG_ENV_VAR: "none"})
    ()
    """
    environ = os.environ if env is None else env
    explicit = _explicit_path(environ)
    if explicit is _OFF:
        return ()
    if isinstance(explicit, Path):
        return (explicit,) if explicit.is_file() else ()

    found: list[Path] = []
    user = user_config_path(env=environ, windows=windows)
    if user.is_file():
        found.append(user)
    project = project_config_path(start)
    if project is not None and project != user and project_config_trusted(env=environ):
        found.append(project)
    return tuple(found)


def load_config(
    path: str | Path | None = None,
    *,
    start: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    windows: bool | None = None,
) -> PricingConfig:
    """Load and validate the effective configuration.

    Returns an empty (but valid) :class:`PricingConfig` when nothing is
    configured. Raises :class:`~agentguard.GuardConfigError` when a file was
    explicitly requested and is missing, or when any file is malformed. A
    project-level file that exists but is not trusted is skipped, with a warning.

    >>> load_config(env={CONFIG_ENV_VAR: "off"}).is_empty
    True
    """
    environ = os.environ if env is None else env
    if path is not None:
        return _finalize([_load_file(Path(path), required=True)])

    explicit = _explicit_path(environ)
    if explicit is _OFF:
        return PricingConfig()
    if isinstance(explicit, Path):
        return _finalize([_load_file(explicit, required=True)])

    ignored = ignored_project_config(start=start, env=environ)
    if ignored is not None:
        _warn_ignored_project_config(ignored)
    return _finalize(
        [
            _load_file(candidate, required=False)
            for candidate in config_paths(start=start, env=environ, windows=windows)
        ]
    )


def _finalize(parts: list[PricingConfig]) -> PricingConfig:
    """Merge the files, then validate the result as a whole.

    Whole-document checks run *after* merging, which is what lets a project file
    alias a model the user-level file defines while a dangling alias is still
    caught before any call is billed.
    """
    merged = merge_configs(parts)
    if merged.sources:
        where = " and ".join(str(source) for source in merged.sources)
        _check_buildable(merged, f"{where}: ")
    return merged


def _load_file(path: Path, *, required: bool) -> PricingConfig:
    """Read one file. Field-level checks only; the merge step checks the whole."""
    if not path.is_file():
        if required:
            raise GuardConfigError(f"config file not found: {path}")
        return PricingConfig()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GuardConfigError(f"cannot read config file {path}: {exc}") from exc
    if not text.strip():
        # A touched-but-empty file is a plausible accident; an empty config is
        # still the honest reading of it, and every entry it lacks is unpriced.
        return PricingConfig(sources=(path,))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuardConfigError(f"{path} is not valid JSON: {exc}") from exc
    return _parse_document(data, source=path, buildable=False)


# --------------------------------------------------------------------------- #
# Editing (used by the CLI, and safe to use from code)
# --------------------------------------------------------------------------- #


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Read a config file for editing, or return a fresh template.

    Field-level validation runs here; the whole-document checks (aliases that
    resolve, ``disable`` entries that exist) are left to the write, so that a file
    broken by an upgrade can still be repaired with ``agentguard config remove``.
    """
    target = Path(path)
    if not target.is_file():
        return _template()
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        return _template()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuardConfigError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GuardConfigError(f"{target} must contain a JSON object")
    _parse_document(data, source=target, buildable=False)
    data.setdefault("version", CONFIG_VERSION)
    return data


def write_config_file(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Validate ``data`` and write it atomically.

    Validation happens *before* the write, so a rejected edit leaves the previous
    file untouched. The write itself goes to a temporary file in the same
    directory and is then moved into place, so an interrupted process cannot
    leave a half-written config behind.
    """
    target = Path(path)
    document = dict(data)
    document.setdefault("version", CONFIG_VERSION)
    parse_config(document, source=target)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise
    return target


def initialize_config(path: str | Path, *, force: bool = False) -> Path:
    """Write a starter file, refusing to clobber an existing one.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     written = initialize_config(Path(folder) / "agentguard.json")
    ...     print(written.read_text(encoding="utf-8").splitlines()[0])
    {
    """
    target = Path(path)
    if target.exists() and not force:
        raise GuardConfigError(f"{target} already exists; pass force=True to overwrite it")
    return write_config_file(target, _template())


def _entries(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        value = {}
        document[key] = value
    return value


def set_model_price(
    path: str | Path,
    name: str,
    *,
    input_per_1m: float,
    output_per_1m: float,
    cached_input_per_1m: float | None = None,
) -> Price:
    """Add or update a model price in ``path``. Returns the validated price."""
    price = Price(input_per_1m, output_per_1m, cached_input_per_1m)
    document = read_config_file(path)
    if name.strip().lower() in _entries(document, "aliases"):
        raise GuardConfigError(
            f"{name!r} is an alias in {path}; remove the alias before giving it a price"
        )
    entry: dict[str, float] = {"input": price.input_per_1m, "output": price.output_per_1m}
    if price.cached_input_per_1m is not None:
        entry["cached_input"] = price.cached_input_per_1m
    _entries(document, "models")[name] = entry
    write_config_file(path, document)
    return price


def set_alias(path: str | Path, name: str, target: str) -> None:
    """Point ``name`` at ``target`` in ``path``."""
    document = read_config_file(path)
    priced = {existing.strip().lower() for existing in _entries(document, "models")}
    if name.strip().lower() in priced:
        raise GuardConfigError(
            f"{name!r} has a price in {path}; remove it before making it an alias"
        )
    _entries(document, "aliases")[name] = target.strip()
    write_config_file(path, document)


def set_disabled(path: str | Path, name: str, disabled: bool = True) -> bool:
    """Add or remove a bundled model in the ``disable`` list. Returns changed?

    The name is resolved through the price table first, so a dated model such as
    ``gpt-4o-2024-08-06`` disables the family it is actually billed as.
    """
    table = PriceTable()
    resolved = table.resolve(name)
    if resolved is None:
        raise GuardConfigError(
            f"{name!r} is not a bundled model, so disabling it would do nothing; "
            f"remove it from 'models' instead"
        )
    key = resolved[0]
    document = read_config_file(path)
    priced = {normalize_model_key(existing) for existing in _entries(document, "models")}
    if key in priced:
        # Otherwise the disable would be a silent no-op: an explicit price beats a
        # disable by design, and a no-op is exactly what this library refuses to do.
        raise GuardConfigError(
            f"{name!r} has a price in {path}, and a price beats a disable; remove it "
            f"first with `agentguard config remove {name}`"
        )
    if disabled:
        dependents = []
        for alias, target in _entries(document, "aliases").items():
            found = table.resolve(str(target))
            if found is not None and found[0] == key:
                dependents.append(alias)
        if dependents:
            raise GuardConfigError(
                f"alias(es) {', '.join(repr(alias) for alias in dependents)} point at "
                f"{key!r}, so disabling it would leave them with no price; remove "
                f"those aliases first (or leave {key!r} enabled)"
            )
    current = document.get("disable")
    names = [str(item) for item in current] if isinstance(current, list) else []
    normalized = {normalize_model_key(item): item for item in names}

    if disabled:
        if key in normalized:
            return False
        names.append(key)
    else:
        if key not in normalized:
            return False
        names.remove(normalized[key])

    document["disable"] = sorted(names, key=normalize_model_key)
    write_config_file(path, document)
    return True


def remove_entry(path: str | Path, name: str) -> Removal:
    """Remove ``name`` from the config file at ``path``.

    Matching is case-insensitive, and a model is also matched by its normalized
    name, so ``remove_entry(path, "gpt-4o")`` cleans up an entry written as
    ``OpenAI/GPT-4o``. Aliases that *point at* a removed model are removed too:
    leaving them behind would produce a file that refuses to load, and a dangling
    alias is never what the caller meant.
    """
    document = read_config_file(path)
    key = normalize_model_key(name)
    lowered = name.strip().lower()

    models = _entries(document, "models")
    removed_models = [
        existing
        for existing in list(models)
        if existing == name
        or existing.strip().lower() == lowered
        or (bool(key) and normalize_model_key(existing) == key)
    ]
    for existing in removed_models:
        del models[existing]
    removed_keys = {normalize_model_key(existing) for existing in removed_models}

    aliases = _entries(document, "aliases")
    removed_aliases = [
        existing
        for existing in list(aliases)
        if existing.strip().lower() == lowered
        or (bool(removed_keys) and normalize_model_key(str(aliases[existing])) in removed_keys)
    ]
    for existing in removed_aliases:
        del aliases[existing]

    current = document.get("disable")
    removed_disabled: list[str] = []
    if isinstance(current, list):
        removed_disabled = [item for item in current if normalize_model_key(str(item)) == key]
        if removed_disabled:
            document["disable"] = [
                item for item in current if normalize_model_key(str(item)) != key
            ]

    removal = Removal(
        models=tuple(removed_models),
        aliases=tuple(removed_aliases),
        disabled=tuple(str(item) for item in removed_disabled),
    )
    if removal.changed:
        write_config_file(path, document)
    return removal
