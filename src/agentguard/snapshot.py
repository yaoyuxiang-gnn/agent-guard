"""An opt-in, checksummed refresh of the bundled price table.

``PRICING_AS_OF`` goes stale between releases, and asking a user to wait for a patch
to learn a new model's price is a poor answer. This module is the other half of that
answer: a public price catalogue can be downloaded **once, on purpose**, turned into
a checked snapshot, and merged underneath the user's own config.

Three properties make this safe enough to ship in a library whose whole promise is
that nothing leaves your process:

**Never automatic.** Nothing here runs at import, at :class:`~agentguard.Guard`
construction, or on a timer. The only entry point that touches the network is
:func:`fetch_snapshot`, and it is reached only from
``agentguard pricing --update``. A hidden request during ``import agentguard`` would
be a worse bug than a stale table.

**Checksummed, and verified on every read.** The snapshot records the digest of its
model map and is re-verified whenever it is loaded. A file that has been edited,
truncated by a half-finished write, or corrupted on disk raises rather than quietly
repricing models — a corrupted price table is a corrupted budget.

**Underneath your configuration, always.** The snapshot merges as the lowest layer:
``$AGENTGUARD_CONFIG``, the per-user file and the project file all still win, and
``Guard(pricing=...)`` wins over everything. A downloaded public catalogue can never
override a rate you set deliberately.

The file lives beside the config, in the user's own config directory, and
``agentguard pricing --update`` is the only thing that writes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._util import stable_json
from .exceptions import GuardConfigError
from .pricing import Price, _validate_rate, normalize_model_key

__all__ = [
    "DEFAULT_SNAPSHOT_URL",
    "SNAPSHOT_VERSION",
    "PriceSnapshot",
    "fetch_snapshot",
    "json_to_models",
    "load_snapshot",
    "models_to_json",
    "parse_snapshot_payload",
    "read_snapshot_file",
    "remove_snapshot",
    "snapshot_path",
    "write_snapshot_file",
]

#: Schema version of the snapshot file this module writes and reads.
SNAPSHOT_VERSION = 1

#: Where ``agentguard pricing --update`` downloads from by default.
#:
#: OpenRouter publishes its catalogue as JSON with USD *per token* strings, covering
#: the OpenAI, Anthropic, Google, xAI, DeepSeek, Mistral and Qwen families in one
#: document. It is the same source :data:`agentguard.pricing.DEFAULT_PRICING` is
#: cross-checked against, so a refresh and a release agree on where the numbers come
#: from. It is a **gateway's** catalogue rather than each provider's own page, which
#: is why the snapshot merges underneath your config rather than over it.
DEFAULT_SNAPSHOT_URL = "https://openrouter.ai/api/v1/models"

#: Refuse a download larger than this. A price catalogue is a few hundred kilobytes;
#: anything past this is a misconfigured URL or something that is not a catalogue.
_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024

#: Seconds to wait for the catalogue.
_DOWNLOAD_TIMEOUT = 30.0

#: The CLI sets this on the request so a catalogue that changes format does not get
#: served by a cache that assumes the old one.
_USER_AGENT = "agentguard-pricing-update"

#: Rates are stored rounded to this many decimal places of a USD per 1M tokens, so a
#: per-token string like ``"0.0000002"`` becomes exactly ``0.2``.
_RATE_QUANTUM = Decimal("0.0000001")

#: Parsed snapshots, keyed by ``(path, size, mtime_ns)``. A guard is constructed per
#: request, task or job, so without this a 400-model catalogue would be re-parsed on
#: every one of them. The key changes whenever the file does, so an edit is picked up
#: on the next guard — the same freshness the config files themselves have.
_cache: dict[tuple[str, int, int], PriceSnapshot] = {}
_cache_lock = threading.Lock()
_CACHE_LIMIT = 8


def snapshot_path(*, env: Mapping[str, str] | None = None) -> Path:
    """The snapshot file's location: beside the per-user config file.

    >>> snapshot_path(env={"XDG_CONFIG_HOME": "/tmp/cfg"}).as_posix()
    '/tmp/cfg/agentguard/pricing-snapshot.json'
    """
    # Imported here rather than at module scope, because config imports this module
    # to find the snapshot. A local import keeps that cycle broken by construction.
    from .config import user_config_path

    return user_config_path(env=env).with_name("pricing-snapshot.json")


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """A downloaded price catalogue, validated. Not yet merged with anything.

    ``models`` is keyed by *normalized* model name, so it drops straight into the
    same lookup the bundled table uses.
    """

    models: Mapping[str, Price]
    checksum: str
    url: str
    retrieved_at: str
    source: Path | None = None
    skipped: int = 0
    """Entries the catalogue carried that could not be used as a plain rate."""

    def __post_init__(self) -> None:
        # Read-only, because a parsed snapshot is memoised and shared between every
        # guard built from it: one caller mutating "its" mapping would silently
        # reprice every other guard in the process.
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    @property
    def models_count(self) -> int:
        return len(self.models)

    def verify(self) -> None:
        """Raise unless ``checksum`` still describes ``models``."""
        expected = _checksum(self.models)
        if expected != self.checksum:
            where = f" at {self.source}" if self.source is not None else ""
            raise GuardConfigError(
                f"the price snapshot{where} does not match its own checksum: it "
                f"recorded {self.checksum} but its {len(self.models)} models digest "
                f"to {expected}. The file is corrupt or was edited by hand; re-run "
                f"`agentguard pricing --update`, or delete it to fall back to the "
                f"bundled table."
            )

    def describe(self) -> str:
        """One line for the CLI, e.g. ``412 models, fetched 2026-09-30``."""
        return f"{self.models_count} models, fetched {self.retrieved_at[:10]}"

    def as_document(self) -> dict[str, Any]:
        """The JSON document written to disk."""
        return {
            "version": SNAPSHOT_VERSION,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "models_count": self.models_count,
            "skipped": self.skipped,
            "checksum": self.checksum,
            "models": models_to_json(self.models),
        }


def models_to_json(models: Mapping[str, Price]) -> dict[str, dict[str, float]]:
    """The on-disk shape of a model map: ``{name: {input, output, cached_input?}}``.

    One function produces this on both the write and the read path, which is what
    makes the checksum meaningful. Hashing a live :class:`~agentguard.Price` on one
    side and a decoded JSON object on the other produces two different digests for
    the same data — the file would fail its own integrity check the moment it was
    read back.
    """
    return {
        name: {
            "input": price.input_per_1m,
            "output": price.output_per_1m,
            **(
                {"cached_input": price.cached_input_per_1m}
                if price.cached_input_per_1m is not None
                else {}
            ),
        }
        for name, price in sorted(models.items())
    }


def json_to_models(raw: Mapping[str, Any], *, context: str) -> dict[str, Price]:
    """Read :func:`models_to_json` output back, rejecting anything unreadable.

    The stored rates are already **per 1M tokens**, so they are validated rather
    than rescaled — the per-token conversion belongs to the catalogue parser, and
    applying it twice would multiply every price by a million.

    Strict on purpose: this is the path a snapshot file takes, and a snapshot that
    half-parses is a price table with holes in it.
    """
    prices: dict[str, Price] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            raise GuardConfigError(f"{context}: model entry {name!r} is not an object")
        key = normalize_model_key(name)
        if not key:
            raise GuardConfigError(f"{context}: model entry {name!r} has an empty name")
        rates: dict[str, float | None] = {}
        for field in ("input", "output", "cached_input"):
            value = spec.get(field)
            if value is None and field == "cached_input":
                rates[field] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GuardConfigError(
                    f"{context}: model entry {name!r} needs a numeric {field!r} rate, got {value!r}"
                )
            rates[field] = _validate_rate(float(value), field=field, context=f"{context}: {name}")
        if rates["input"] is None or rates["output"] is None:
            raise GuardConfigError(
                f"{context}: model entry {name!r} needs an 'input' and an 'output' rate"
            )
        prices[key] = Price(rates["input"], rates["output"], rates["cached_input"])
    return prices


def _checksum(models: Mapping[str, Price]) -> str:
    """A stable digest of a model map.

    Taken over the canonical serialisation of :func:`models_to_json`, so a file
    rewritten by a different Python, a different OS or a different dictionary
    insertion order still hashes the same. A checksum that depended on any of those
    would fail on a round trip through the very file it is protecting.
    """
    payload = stable_json(models_to_json(models)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# Parsing a catalogue
# --------------------------------------------------------------------------- #


def _rate_from_per_token(raw: object, *, context: str, field: str) -> float | None:
    """Convert a per-token USD figure to a per-1M-token rate.

    Catalogues publish dollars per token as a decimal string. Scaling by 1,000,000
    is done in :class:`~decimal.Decimal` and rounded to the nearest ten-millionth
    of a cent, because doing it in binary floating point produces numbers like
    ``0.19999999999999998`` for a catalogue that plainly says ``0.2`` — and a rate
    that disagreees with the published one in the sixteenth digit is a rate nobody
    can check against the page it came from.

    Returns ``None`` for an entry carrying no usable rate — including the ``-1``
    some catalogues use as "priced elsewhere" — so the caller skips it rather than
    inventing a number.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise GuardConfigError(f"{context}: {field} must be a number, got {raw!r}")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = Decimal(text)
        except InvalidOperation as exc:
            raise GuardConfigError(f"{context}: {field} is not a number: {raw!r}") from exc
    elif isinstance(raw, (int, float)):
        value = Decimal(str(raw))
    else:
        raise GuardConfigError(f"{context}: {field} must be a number, got {raw!r}")

    if not value.is_finite():
        raise GuardConfigError(f"{context}: {field} must be finite, got {raw!r}")
    if value < 0:
        # A negative rate is a sentinel for "this catalogue cannot price it", not a
        # discount. Skipping keeps "never guess a price" true.
        return None
    scaled = (value * 1_000_000).quantize(_RATE_QUANTUM).normalize()
    return _validate_rate(float(scaled), field=field, context=context)


def _model_name_from_catalogue_id(identifier: str) -> str:
    """Turn a catalogue id into the name a provider would report.

    ``"anthropic/claude-sonnet-4.5"`` is how a gateway names the model;
    ``"claude-sonnet-4.5"`` is what the SDK reports back. The trailing segment is
    what :func:`~agentguard.pricing.normalize_model_key` would look at anyway, and
    using it here means the snapshot's keys line up with the bundled table's.
    """
    name = identifier.split(":")[0]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return normalize_model_key(name)


def _variant_of(identifier: str) -> str | None:
    """The ``:variant`` suffix of a catalogue id, if it has one.

    Catalogues publish several SKUs of one model under one base name —
    ``model:batch`` at half price, ``model:free`` at nothing, ``model:thinking``.
    Each normalizes to the *same* model name, so only one of them can be stored.
    Keeping the cheapest is what a first pass does, and it is wrong in the
    direction that matters: an agent billed at the standard rate would be measured
    against the batch rate, so its cap would fire at half the spend it thought it
    was tracking.

    So a variant is skipped by name rather than resolved by price. A ``:free``
    model therefore reports as unpriced rather than as ``$0`` — the same
    conservative choice this library makes everywhere else, and the honest one,
    since "this call cost nothing" is a claim only the invoice can settle.
    """
    _, separator, suffix = identifier.partition(":")
    return suffix if separator and suffix else None


def _parse_catalogue(models: object, *, context: str) -> tuple[dict[str, Price], int]:
    """Read OpenRouter-shaped entries into ``{normalized_name: Price}``."""
    if not isinstance(models, list):
        raise GuardConfigError(f"{context}: expected a list of models, got {type(models).__name__}")

    prices: dict[str, Price] = {}
    skipped = 0
    for position, entry in enumerate(models):
        where = f"{context}: models[{position}]"
        if not isinstance(entry, Mapping):
            skipped += 1
            continue
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            skipped += 1
            continue
        if _variant_of(identifier) is not None:
            skipped += 1  # a different SKU under the same model name; see above
            continue
        pricing = entry.get("pricing")
        if not isinstance(pricing, Mapping):
            skipped += 1
            continue

        name = _model_name_from_catalogue_id(identifier)
        if not name:
            skipped += 1
            continue

        input_rate = _rate_from_per_token(
            pricing.get("prompt"), context=f"{where} ({identifier})", field="prompt"
        )
        output_rate = _rate_from_per_token(
            pricing.get("completion"), context=f"{where} ({identifier})", field="completion"
        )
        if input_rate is None or output_rate is None:
            skipped += 1
            continue
        cached_rate = _rate_from_per_token(
            pricing.get("input_cache_read"),
            context=f"{where} ({identifier})",
            field="input_cache_read",
        )

        price = Price(input_rate, output_rate, cached_rate)
        existing = prices.get(name)
        # Two distinct ids can still normalize to one name — one provider listing a
        # model twice, or a dotted Bedrock prefix collapsing onto the bare name.
        # Keep the more expensive reading, because for a budget cap the conservative
        # direction is to assume the higher rate and fire early rather than late.
        if existing is None or (price.input_per_1m, price.output_per_1m) > (
            existing.input_per_1m,
            existing.output_per_1m,
        ):
            prices[name] = price
    return prices, skipped


def parse_snapshot_payload(
    payload: object, *, url: str = "", retrieved_at: str | None = None
) -> PriceSnapshot:
    """Build a :class:`PriceSnapshot` from a decoded catalogue document.

    Understands the catalogue shape (``{"data": [{"id": .., "pricing": {..}}]}``)
    and this module's own on-disk shape, so ``--from-file`` can re-import a document
    this tool wrote without a special case.

    >>> payload = {"data": [
    ...     {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025",
    ...                                         "completion": "0.00001"}},
    ... ]}
    >>> snapshot = parse_snapshot_payload(payload)
    >>> snapshot.models["gpt-4o"].input_per_1m
    2.5
    >>> snapshot.verify() is None
    True
    >>> # A negative rate is a sentinel, not a price: the entry is skipped.
    >>> len(parse_snapshot_payload({"data": [
    ...     {"id": "x/router", "pricing": {"prompt": "-1", "completion": "-1"}},
    ... ]}).models)
    0
    """
    if not isinstance(payload, Mapping):
        raise GuardConfigError(
            f"a price catalogue must be a JSON object, got {type(payload).__name__}"
        )

    context = f"catalogue from {url}" if url else "catalogue"
    if "models" in payload and "data" not in payload:
        # This module's own file shape: {"version": 1, "models": {name: {...}}}. The
        # rates in it are already per 1M tokens, so they are validated rather than
        # rescaled — the per-token conversion is the *catalogue's* unit, and applying
        # it here would multiply every price by a million.
        raw = payload.get("models")
        if not isinstance(raw, Mapping):
            raise GuardConfigError(f"{context}: 'models' must be an object")
        prices = json_to_models(raw, context=context)
        skipped = 0
    else:
        prices, skipped = _parse_catalogue(payload.get("data", payload), context=context)

    if not prices:
        raise GuardConfigError(
            f"{context} contains no usable model prices. Check the URL points at a "
            f"model catalogue rather than a web page; the default source is "
            f"{DEFAULT_SNAPSHOT_URL}."
        )

    return PriceSnapshot(
        models=prices,
        checksum=_checksum(prices),
        url=url or str(payload.get("url", "")) or DEFAULT_SNAPSHOT_URL,
        retrieved_at=retrieved_at or str(payload.get("retrieved_at") or _now()),
        skipped=skipped,
    )


# --------------------------------------------------------------------------- #
# Reading and writing the snapshot file
# --------------------------------------------------------------------------- #


def read_snapshot_file(path: str | Path) -> PriceSnapshot:
    """Load and verify the snapshot at ``path``.

    Raises :class:`~agentguard.GuardConfigError` if the file is missing, malformed,
    has a newer schema, or fails its own checksum. Every one of those is a refusal
    rather than a repair: a snapshot that loads *approximately* is a price table
    that is approximately right, and a budget built on it is approximately a budget.

    A parsed snapshot is memoised against the file's size and modification time, so
    constructing many guards costs one parse rather than one per guard. Editing or
    replacing the file invalidates the entry, which is what keeps "editing the config
    takes effect on the next guard" true — see the module docstring of
    :mod:`agentguard.config`.
    """
    target = Path(path)
    if not target.is_file():
        raise GuardConfigError(f"no price snapshot at {target}")

    try:
        stat = target.stat()
        stamp = (str(target), stat.st_size, stat.st_mtime_ns)
    except OSError as exc:
        raise GuardConfigError(f"cannot read price snapshot {target}: {exc}") from exc

    with _cache_lock:
        cached = _cache.get(stamp)
    if cached is not None:
        return cached

    snapshot = _read_snapshot_file(target)
    with _cache_lock:
        # Unbounded growth would be a leak in a long-lived worker, and the key is
        # unique per edit, so keep only the most recent few.
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        _cache[stamp] = snapshot
    return snapshot


def _read_snapshot_file(target: Path) -> PriceSnapshot:
    """Parse and verify one snapshot file. The uncached half of the read path."""
    try:
        # utf-8-sig, not utf-8: a snapshot or catalogue edited in a Windows editor
        # commonly starts with a BOM, and json.loads rejects one outright. Stripping
        # it is safe (a BOM is absent more often than not) and turns an inscrutable
        # "column 1 char 0" into a file that simply loads.
        text = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise GuardConfigError(f"cannot read price snapshot {target}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuardConfigError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise GuardConfigError(f"{target} must contain a JSON object")

    version = data.get("version", SNAPSHOT_VERSION)
    if version != SNAPSHOT_VERSION:
        raise GuardConfigError(
            f"{target} is snapshot format version {version!r}, but this agentguard "
            f"reads version {SNAPSHOT_VERSION}. Re-run `agentguard pricing --update`."
        )

    raw_models = data.get("models")
    if not isinstance(raw_models, Mapping):
        raise GuardConfigError(f"{target} is missing its 'models' object")
    prices = json_to_models(raw_models, context=str(target))

    snapshot = PriceSnapshot(
        models=prices,
        checksum=str(data.get("checksum", "")),
        url=str(data.get("url", "")),
        retrieved_at=str(data.get("retrieved_at", "")),
        source=target,
        skipped=int(data.get("skipped", 0)),
    )
    snapshot.verify()
    return snapshot


def write_snapshot_file(path: str | Path, snapshot: PriceSnapshot) -> Path:
    """Write ``snapshot`` atomically, in the same directory and via a temp file.

    Same reasoning as the config writer: an interrupted ``--update`` must leave the
    previous snapshot intact rather than a truncated one, because a truncated price
    table is a budget with a hole in it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot.as_document(), indent=2, ensure_ascii=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise
    with _cache_lock:
        _cache.clear()
    return target


def remove_snapshot(*, env: Mapping[str, str] | None = None) -> Path | None:
    """Delete the snapshot, returning the path removed (or ``None`` if there was none)."""
    target = snapshot_path(env=env)
    if not target.is_file():
        return None
    target.unlink()
    with _cache_lock:
        _cache.clear()
    return target


def load_snapshot(*, env: Mapping[str, str] | None = None) -> PriceSnapshot | None:
    """The snapshot in effect, or ``None`` when there is none.

    Returns ``None`` — never raises — for a *missing* file, because having no
    snapshot is the normal state. A file that exists and cannot be verified raises:
    silently ignoring a corrupt snapshot would leave the user believing prices were
    refreshed when the bundled table is what is actually in effect.
    """
    target = snapshot_path(env=env)
    if not target.is_file():
        return None
    return read_snapshot_file(target)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def _check_url(url: str) -> str:
    """Accept only http(s), so ``--url file:///etc/passwd`` cannot become a file read."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise GuardConfigError(
            f"a price catalogue must be fetched over http or https, not "
            f"{parsed.scheme or 'a relative path'!r}: {url}"
        )
    if not parsed.netloc:
        raise GuardConfigError(f"not a usable catalogue URL: {url}")
    return url


def fetch_snapshot(
    url: str = DEFAULT_SNAPSHOT_URL,
    *,
    timeout: float = _DOWNLOAD_TIMEOUT,
    max_bytes: int = _MAX_DOWNLOAD_BYTES,
) -> PriceSnapshot:
    """Download a price catalogue and turn it into a verified snapshot.

    The only function in this library that makes a network request, and it is called
    only by ``agentguard pricing --update``. Reading is capped at ``max_bytes`` so a
    wrong URL cannot stream an unbounded body into memory.

    Raises :class:`~agentguard.GuardConfigError` for a non-http(s) scheme, a network
    failure, a non-200 response, an oversized body, or a payload with no usable
    prices — each naming the URL, because "it did not work" is not a useful thing to
    tell someone who just ran a command.
    """
    target = _check_url(url)
    request = urllib.request.Request(
        target,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise GuardConfigError(f"the price catalogue at {target} returned HTTP {status}")
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise GuardConfigError(
            f"the price catalogue at {target} returned HTTP {exc.code} ({exc.reason})"
        ) from exc
    except urllib.error.URLError as exc:
        raise GuardConfigError(
            f"could not reach the price catalogue at {target}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise GuardConfigError(f"could not read the price catalogue at {target}: {exc}") from exc

    if len(body) > max_bytes:
        raise GuardConfigError(
            f"the price catalogue at {target} is larger than {max_bytes} bytes, which "
            f"is not a price catalogue; check the URL"
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardConfigError(f"the price catalogue at {target} is not JSON: {exc}") from exc

    return parse_snapshot_payload(payload, url=target)
