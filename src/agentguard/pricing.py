"""Model pricing table and cost arithmetic.

agent-guard ships a best-effort snapshot of public list prices so a guard can
turn token counts into dollars with no network call and no vendor SDK.

.. warning::

   Provider prices change often, and this table is **not** a billing source of
   truth. It exists so an agent can stop itself before it burns a budget — not so
   you can invoice a customer. Check :data:`PRICING_AS_OF`, and override anything
   that matters to you:

   .. code-block:: python

      from agentguard import Guard, Price

      guard = Guard(
          max_usd=5.0,
          pricing={"my-finetune-v3": Price(3.00, 12.00)},
      )

   Unknown models are never silently billed at a guessed rate. By default the
   guard warns, counts the unpriced calls, and surfaces them in the report, so a
   mis-priced model can never quietly hide an overspend.

   Everything in the table can be added to, repriced, aliased or removed from a
   JSON config file, so a user never has to patch a library to price their own
   models. See :mod:`agentguard.config`::

       agentguard config set my-finetune-v3 --input 3 --output 12

   Underneath, that config becomes ``overrides`` on the :class:`PriceTable`,
   which is also the API for doing it in code::

       PriceTable(overrides={"my-finetune-v3": (3.0, 12.0)})
"""

from __future__ import annotations

import math
import re
from collections.abc import ItemsView, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import GuardConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import PricingConfig

__all__ = [
    "Price",
    "PriceTable",
    "DEFAULT_PRICING",
    "PRICING_AS_OF",
    "ORIGIN_BUILTIN",
    "ORIGIN_CONFIG",
    "ORIGIN_OVERRIDE",
    "normalize_model_key",
]


#: Date the bundled price snapshot was last reviewed. Prices move; treat anything
#: older than a few months as indicative and pass your own ``pricing=`` overrides.
PRICING_AS_OF = "2026-01"

#: Where a resolved price came from. Reported per entry so "why is this model
#: $9?" has an answer that does not require reading the source.
ORIGIN_BUILTIN = "builtin"
ORIGIN_CONFIG = "config"
ORIGIN_OVERRIDE = "override"

#: Accepted spellings for the three rates in a mapping-shaped price. The long
#: forms match the :class:`Price` field names, so a field name copied out of the
#: ``repr()`` works.
_PRICE_RATE_KEYS: dict[str, tuple[str, ...]] = {
    "input": ("input", "input_per_1m"),
    "output": ("output", "output_per_1m"),
    "cached_input": ("cached_input", "cached_input_per_1m", "cached"),
}
_FIELD_FOR_RATE_KEY: dict[str, str] = {
    key: field for field, keys in _PRICE_RATE_KEYS.items() for key in keys
}


def _validate_rate(value: object, *, field: str, context: str) -> float:
    """Return ``value`` as a finite, non-negative float, or raise.

    A ``NaN`` or infinite rate is not a harmless curiosity: ``NaN`` compares
    false against every budget, so a single bad number would silently disarm the
    guard this library exists to provide.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardConfigError(f"{context}: {field} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise GuardConfigError(f"{context}: {field} must be a finite number, got {value!r}")
    if number < 0:
        raise GuardConfigError(f"{context}: {field} must not be negative, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class Price:
    """List price for one model, in **US dollars per 1,000,000 tokens**.

    >>> price = Price(input_per_1m=2.50, output_per_1m=10.00)
    >>> round(price.cost_usd(input_tokens=1_000_000), 4)
    2.5
    >>> round(price.cost_usd(input_tokens=1_000, output_tokens=500), 6)
    0.0075
    >>> price.as_dict()["output_per_1m"]
    10.0
    """

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None

    def __post_init__(self) -> None:
        # Validate here rather than at the point of use: a bad rate must fail
        # where it was written, not fifty calls later inside a budget check.
        for field in ("input_per_1m", "output_per_1m"):
            rate = _validate_rate(getattr(self, field), field=field, context="Price")
            object.__setattr__(self, field, rate)
        if self.cached_input_per_1m is not None:
            cached = _validate_rate(
                self.cached_input_per_1m, field="cached_input_per_1m", context="Price"
            )
            object.__setattr__(self, "cached_input_per_1m", cached)

    def as_dict(self) -> dict[str, float | None]:
        """Serialisable form, mirroring the config file's key names.

        >>> Price(1.0, 2.0, 0.5).as_dict()["cached_input_per_1m"]
        0.5
        """
        return {
            "input_per_1m": self.input_per_1m,
            "output_per_1m": self.output_per_1m,
            "cached_input_per_1m": self.cached_input_per_1m,
        }

    def cost_usd(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> float:
        """Exact cost of one call in USD.

        ``cached_input_tokens`` is treated as a *subset* of ``input_tokens`` —
        which is how both OpenAI and Anthropic report cache hits — and is billed
        at the discounted rate when one is known.
        """
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cached = max(0, min(int(cached_input_tokens), input_tokens))

        total = (input_tokens - cached) * self.input_per_1m
        if cached:
            rate = (
                self.cached_input_per_1m
                if self.cached_input_per_1m is not None
                else self.input_per_1m
            )
            total += cached * rate
        total += output_tokens * self.output_per_1m
        return total / 1_000_000.0

    def worst_case_usd(self, *, input_tokens: int, max_output_tokens: int) -> float:
        """Upper bound on a call whose output length is only known as a cap.

        This is what :meth:`agentguard.Guard.preflight` uses to refuse a call
        that *could* blow the budget, before any money is spent.
        """
        return self.cost_usd(
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(max_output_tokens)),
        )


#: Keys are canonical, lower-case model names. Aliases resolve via
#: :func:`normalize_model_key` plus a conservative prefix match.
DEFAULT_PRICING: dict[str, Price] = {
    # ---------------------------------------------------------------- OpenAI
    "gpt-5": Price(1.25, 10.00, 0.125),
    "gpt-5-mini": Price(0.25, 2.00, 0.025),
    "gpt-5-nano": Price(0.05, 0.40, 0.005),
    "gpt-4.1": Price(2.00, 8.00, 0.50),
    "gpt-4.1-mini": Price(0.40, 1.60, 0.10),
    "gpt-4.1-nano": Price(0.10, 0.40, 0.025),
    "gpt-4o": Price(2.50, 10.00, 1.25),
    "gpt-4o-mini": Price(0.15, 0.60, 0.075),
    "gpt-4-turbo": Price(10.00, 30.00),
    "gpt-4": Price(30.00, 60.00),
    "gpt-3.5-turbo": Price(0.50, 1.50),
    "o3": Price(2.00, 8.00, 0.50),
    "o3-mini": Price(1.10, 4.40, 0.55),
    "o4-mini": Price(1.10, 4.40, 0.275),
    # ------------------------------------------------------------- Anthropic
    "claude-opus-4-1": Price(15.00, 75.00, 1.50),
    "claude-opus-4": Price(15.00, 75.00, 1.50),
    "claude-sonnet-4-5": Price(3.00, 15.00, 0.30),
    "claude-sonnet-4": Price(3.00, 15.00, 0.30),
    "claude-3-7-sonnet": Price(3.00, 15.00, 0.30),
    "claude-3-5-sonnet": Price(3.00, 15.00, 0.30),
    "claude-3-5-haiku": Price(0.80, 4.00, 0.08),
    "claude-3-opus": Price(15.00, 75.00, 1.50),
    "claude-3-haiku": Price(0.25, 1.25, 0.03),
    # ---------------------------------------------------------------- Google
    "gemini-2.5-pro": Price(1.25, 10.00, 0.31),
    "gemini-2.5-flash": Price(0.30, 2.50, 0.075),
    "gemini-2.0-flash": Price(0.10, 0.40, 0.025),
    "gemini-1.5-pro": Price(1.25, 5.00, 0.3125),
    "gemini-1.5-flash": Price(0.075, 0.30, 0.01875),
    # -------------------------------------------------------------- DeepSeek
    "deepseek-chat": Price(0.27, 1.10, 0.07),
    "deepseek-reasoner": Price(0.55, 2.19, 0.14),
    # --------------------------------------------------------------- Mistral
    "mistral-large": Price(2.00, 6.00),
    "mistral-small": Price(0.20, 0.60),
    "codestral": Price(0.30, 0.90),
    # ------------------------------------------------------------------ xAI
    "grok-4": Price(3.00, 15.00, 0.75),
    "grok-3": Price(3.00, 15.00, 0.75),
    "grok-3-mini": Price(0.30, 0.50, 0.075),
    # ------------------------------------------------------- Open-weight hosts
    "llama-3.3-70b": Price(0.59, 0.79),
    "llama-3.1-8b": Price(0.05, 0.08),
    "qwen-max": Price(1.60, 6.40),
    "qwen-plus": Price(0.40, 1.20),
    "qwen-turbo": Price(0.05, 0.20),
}


# Bedrock / Vertex / gateway style dotted namespacing, e.g.
# "us.anthropic.claude-3-5-sonnet-20241022-v2". Stripped repeatedly, outermost
# prefix first. Never applied to bare model names, so "gpt-3.5-turbo" is safe.
_DOTTED_PREFIXES: tuple[str, ...] = (
    "us",
    "eu",
    "apac",
    "global",
    "anthropic",
    "amazon",
    "meta",
    "mistral",
    "cohere",
    "ai21",
    "stability",
    "openai",
    "google",
    "qwen",
    "deepseek",
)

# A trailing segment that is a pure version marker, so dropping it cannot change
# which model family we are looking at: "20241022", "v2", "1.5".
_VERSION_SUFFIX = re.compile(r"^v?\d[\d.]*$")

# Trailing segments that are pure build/rollout markers with no pricing meaning.
_ALLOWED_SUFFIXES = frozenset({"latest", "preview", "beta", "free", "rc"})


def normalize_model_key(model: str) -> str:
    """Fold a provider-specific model string into a canonical table key.

    Handles the four shapes seen in the wild: gateway namespacing (``/``),
    version pinning (``@``), variant tags (``:free``) and dotted provider
    prefixes (Bedrock / Vertex).

    >>> normalize_model_key("OpenAI/GPT-4o")
    'gpt-4o'
    >>> normalize_model_key("anthropic.claude-sonnet-4@20250514")
    'claude-sonnet-4'
    >>> normalize_model_key("deepseek/deepseek-chat-v3:free")
    'deepseek-chat-v3'
    >>> normalize_model_key("gpt-3.5-turbo")
    'gpt-3.5-turbo'
    """
    key = (model or "").strip().lower()
    if not key:
        return ""

    # Variant tags first: "deepseek-chat:free", "bedrock-version:0".
    key = key.split(":", 1)[0]
    # Gateway namespacing: "openai/gpt-4o", "meta-llama/Llama-3.3-70B-Instruct".
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    # Version pinning: "claude-sonnet-4@20250514".
    if "@" in key:
        key = key.split("@", 1)[0]

    # Dotted provider prefixes, possibly stacked: "us.anthropic.claude-3-5-sonnet".
    changed = True
    while changed:
        changed = False
        for prefix in _DOTTED_PREFIXES:
            if key.startswith(prefix + "."):
                key = key[len(prefix) + 1 :]
                changed = True
                break

    return key.strip()


def _split_trailing_version(key: str, table: Mapping[str, Price]) -> tuple[str, Price] | None:
    """Drop trailing version/build segments until a known family is found.

    Only segments that cannot change the model family are dropped, so
    ``gpt-4o-2024-08-06`` resolves to ``gpt-4o`` while ``gpt-4-turbo`` never
    gets mis-billed as ``gpt-4``.
    """
    parts = key.split("-")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:end])
        if candidate not in table:
            continue
        suffix = parts[end:]
        if all(_VERSION_SUFFIX.match(p) or p in _ALLOWED_SUFFIXES for p in suffix):
            return candidate, table[candidate]
    return None


class PriceTable:
    """A model price lookup with user overrides, aliases and disabled entries.

    Overrides accept a :class:`Price`, a 2-tuple ``(input, output)``, a 3-tuple
    ``(input, output, cached_input)``, or a mapping with ``input`` / ``output`` /
    ``cached_input`` keys — always USD per 1M tokens.

    >>> table = PriceTable(overrides={"my-model": (1.0, 2.0)})
    >>> round(table.resolve_price("my-model").output_per_1m, 2)
    2.0
    >>> table.resolve_price("gpt-4o-2024-08-06").input_per_1m
    2.5
    >>> table.resolve_price("totally-unknown-model") is None
    True

    ``aliases`` map a name your gateway reports onto a name that has a price, and
    ``disable`` drops bundled entries you do not trust:

    >>> table = PriceTable(
    ...     aliases={"acme/fast": "claude-3-5-haiku"},
    ...     disable=["gpt-4"],
    ... )
    >>> table.resolve("acme/fast")[0]
    'claude-3-5-haiku'
    >>> table.resolve_price("gpt-4") is None
    True
    >>> table.origin("acme/fast")
    'builtin'
    """

    __slots__ = ("_aliases", "_disabled", "_origins", "_prices", "_sources")

    def __init__(
        self,
        base: Mapping[str, Price] | None = None,
        overrides: Mapping[str, Price | tuple[float, ...] | Mapping[str, Any]] | None = None,
        *,
        aliases: Mapping[str, str] | None = None,
        disable: Iterable[str] = (),
        sources: Iterable[Path] = (),
    ) -> None:
        self._assemble(base, None, overrides, aliases, disable, sources)

    @classmethod
    def from_config(
        cls,
        config: PricingConfig | None,
        *,
        base: Mapping[str, Price] | None = None,
        overrides: Mapping[str, Price | tuple[float, ...] | Mapping[str, Any]] | None = None,
        aliases: Mapping[str, str] | None = None,
        disable: Iterable[str] = (),
    ) -> PriceTable:
        """Build a table from a :class:`~agentguard.PricingConfig` plus code.

        Precedence, lowest first: bundled prices, config-file models, then the
        keyword arguments passed here. Disabling a model removes it from the
        bundled table only, so an explicit ``overrides`` entry still wins.

        >>> from agentguard.config import parse_config
        >>> config = parse_config({"models": {"mine": [1.0, 2.0]}})
        >>> table = PriceTable.from_config(config, overrides={"mine": (3.0, 4.0)})
        >>> table.origin("mine")
        'override'
        >>> round(table.resolve_price("mine").input_per_1m, 2)
        3.0
        >>> PriceTable.from_config(config).origin("mine")
        'config'
        """
        table = cls.__new__(cls)
        table._assemble(
            base,
            config,
            overrides,
            aliases,
            disable,
            config.sources if config is not None else (),
        )
        return table

    def _assemble(
        self,
        base: Mapping[str, Price] | None,
        config: PricingConfig | None,
        overrides: Mapping[str, Price | tuple[float, ...] | Mapping[str, Any]] | None,
        aliases: Mapping[str, str] | None,
        disable: Iterable[str],
        sources: Iterable[Path],
    ) -> None:
        prices: dict[str, Price] = {}
        origins: dict[str, str] = {}
        for name, price in (DEFAULT_PRICING if base is None else base).items():
            key = normalize_model_key(name)
            if not key:
                raise GuardConfigError(f"price table has an empty model name: {name!r}")
            prices[key] = price
            origins[key] = ORIGIN_BUILTIN

        disabled = frozenset(
            _normalize_name(name, "disable")
            for name in [*(config.disable if config is not None else ()), *disable]
        )
        unknown = sorted(disabled - set(prices))
        if unknown:
            raise GuardConfigError(
                f"cannot disable {', '.join(repr(name) for name in unknown)}: "
                f"no such entry in the bundled price table"
            )
        for key in disabled:
            del prices[key]
            del origins[key]

        self._prices = prices
        self._origins = origins
        self._disabled = disabled
        self._aliases: dict[str, str] = {}
        self._sources = tuple(Path(source) for source in sources)

        # Prices first — bundled, then config file, then code — so that the
        # caller's most explicit statement is the one that ends up in the table.
        defined: set[str] = set()
        config_names: dict[str, str] = {}
        if config is not None:
            for name, price in config.models.items():
                key = _normalize_name(name, "custom model")
                previous = config_names.get(key)
                if previous is not None:
                    raise GuardConfigError(
                        f"{previous!r} and {name!r} both resolve to the model name {key!r}; "
                        f"give them distinct names"
                    )
                config_names[key] = name
                self._prices[key] = price
                self._origins[key] = ORIGIN_CONFIG
                defined.add(key)

        if overrides:
            for name, spec in overrides.items():
                key = _normalize_name(name, "pricing override")
                self._prices[key] = _coerce_price(name, spec)
                self._origins[key] = ORIGIN_OVERRIDE
                defined.add(key)

        # Aliases last, so a name can never be both a price and an alias.
        if config is not None:
            self._apply_aliases(config.aliases, defined)
            self._remember_source(config.sources)
        self._apply_aliases(aliases, defined)
        self._check_aliases()

    def _remember_source(self, sources: Iterable[Path]) -> None:
        known = list(self._sources)
        for source in sources:
            path = Path(source)
            if path not in known:
                known.append(path)
        self._sources = tuple(known)

    def _apply_aliases(self, aliases: Mapping[str, str] | None, defined: set[str]) -> None:
        """Register ``name -> target`` aliases, matched verbatim (case-insensitively).

        Alias names are deliberately *not* normalized: ``{"acme/gpt-4o": "gpt-4o"}``
        must mean the gateway's name only, and normalizing it would silently turn
        the alias into a redefinition of the bundled ``gpt-4o`` entry.
        """
        if not aliases:
            return
        for name, target in aliases.items():
            alias = (name or "").strip().lower() if isinstance(name, str) else ""
            if not alias:
                raise GuardConfigError(f"alias has an empty model name: {name!r}")
            if alias in defined:
                raise GuardConfigError(
                    f"{name!r} is given both a price and an alias; use one or the other"
                )
            if not isinstance(target, str) or not target.strip():
                raise GuardConfigError(
                    f"alias {name!r} must point at a non-empty model name, got {target!r}"
                )
            self._aliases[alias] = target.strip().lower()

    def _check_aliases(self) -> None:
        """Fail at construction, not mid-run, when an alias cannot be resolved."""
        for alias in sorted(self._aliases):
            if self.resolve(alias) is None:
                raise GuardConfigError(
                    f"alias {alias!r} points at {self._aliases[alias]!r}, which has no price; "
                    f"define that model or point the alias somewhere else"
                )

    def _follow_alias(self, name: str) -> str:
        current = name
        seen: list[str] = []
        while current in self._aliases:
            if current in seen:
                raise GuardConfigError(
                    f"alias cycle in the price table: {' -> '.join([*seen, current])}"
                )
            seen.append(current)
            current = self._aliases[current]
        return current

    # -- mapping-ish surface -------------------------------------------------

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and self.resolve_price(model) is not None

    def __len__(self) -> int:
        return len(self._prices)

    def __iter__(self) -> Iterator[str]:
        return iter(self._prices)

    def items(self) -> ItemsView[str, Price]:
        return self._prices.items()

    # -- lookup --------------------------------------------------------------

    def get(self, model: str) -> Price | None:
        """Exact lookup on the normalized key only. No alias, no version fallback."""
        return self._prices.get(normalize_model_key(model))

    def resolve(self, model: str) -> tuple[str, Price] | None:
        """Resolve ``model`` to ``(canonical_name, price)``, or ``None``.

        Tries, in order: an alias on the exact name, the exact normalized key,
        then a conservative trailing version-strip. Never guesses between
        *different* model families.
        """
        key = normalize_model_key(model)
        if not key:
            return None
        alias = (model or "").strip().lower()
        if alias in self._aliases:
            key = normalize_model_key(self._follow_alias(alias))
            if not key:
                return None
        exact = self._prices.get(key)
        if exact is not None:
            return key, exact
        return _split_trailing_version(key, self._prices)

    def resolve_price(self, model: str) -> Price | None:
        """Convenience wrapper around :meth:`resolve` returning only the price."""
        found = self.resolve(model)
        return found[1] if found else None

    # -- provenance ----------------------------------------------------------

    def origin(self, model: str) -> str | None:
        """Where a model's price came from: ``builtin``, ``config`` or ``override``.

        ``None`` when the model has no price at all.

        >>> PriceTable(overrides={"mine": (1.0, 2.0)}).origin("mine")
        'override'
        >>> PriceTable().origin("gpt-4o")
        'builtin'
        """
        found = self.resolve(model)
        return self._origins.get(found[0]) if found else None

    def alias_of(self, model: str) -> str | None:
        """The alias target for ``model``, or ``None`` if it is not an alias."""
        return self._aliases.get((model or "").strip().lower())

    @property
    def aliases(self) -> dict[str, str]:
        """A copy of the alias map."""
        return dict(self._aliases)

    @property
    def disabled(self) -> frozenset[str]:
        """Normalized names removed from the bundled table."""
        return self._disabled

    @property
    def sources(self) -> tuple[Path, ...]:
        """Config files this table was built from, in load order."""
        return self._sources


def _normalize_name(name: object, what: str) -> str:
    """Normalize a user-supplied model name, rejecting anything unusable."""
    if not isinstance(name, str):
        raise GuardConfigError(f"{what} name must be a string, got {name!r}")
    key = normalize_model_key(name)
    if not key:
        raise GuardConfigError(f"{what} has an empty model name: {name!r}")
    return key


def _coerce_price(name: str, spec: Price | tuple[float, ...] | Mapping[str, Any]) -> Price:
    """Turn one config/override entry into a :class:`Price`.

    >>> _coerce_price("m", (1.0, 2.0))
    Price(input_per_1m=1.0, output_per_1m=2.0, cached_input_per_1m=None)
    >>> _coerce_price("m", {"input": 1.0, "output": 2.0, "cached": 0.5})
    Price(input_per_1m=1.0, output_per_1m=2.0, cached_input_per_1m=0.5)
    """
    if isinstance(spec, Price):
        return spec
    if isinstance(spec, Mapping):
        return _price_from_mapping(name, spec)
    if isinstance(spec, (tuple, list)):
        values = [
            _validate_rate(v, field="price", context=f"pricing entry for {name!r}") for v in spec
        ]
        if len(values) == 2:
            return Price(values[0], values[1])
        if len(values) == 3:
            return Price(values[0], values[1], values[2])
        raise GuardConfigError(
            f"pricing override for {name!r} must be (input, output) or "
            f"(input, output, cached_input); got {len(values)} values"
        )
    raise GuardConfigError(
        f"pricing override for {name!r} must be a Price, a tuple of floats, or a "
        f"mapping with input/output keys, got {type(spec).__name__}"
    )


def _price_from_mapping(name: str, spec: Mapping[str, Any]) -> Price:
    """Read ``{"input": .., "output": .., "cached_input": ..}`` into a Price."""
    context = f"pricing entry for {name!r}"
    rates: dict[str, float | None] = {"input": None, "output": None, "cached_input": None}
    for raw_key, raw_value in spec.items():
        field = _FIELD_FOR_RATE_KEY.get(raw_key) if isinstance(raw_key, str) else None
        if field is None:
            raise GuardConfigError(
                f"{context} has unknown key {raw_key!r}; expected one of "
                f"{', '.join(sorted(_FIELD_FOR_RATE_KEY))}"
            )
        if rates[field] is not None:
            raise GuardConfigError(f"{context} sets {field!r} twice")
        if raw_value is None and field == "cached_input":
            continue
        rates[field] = _validate_rate(raw_value, field=raw_key, context=context)

    input_rate, output_rate = rates["input"], rates["output"]
    if input_rate is None or output_rate is None:
        missing = "input" if input_rate is None else "output"
        raise GuardConfigError(f"{context} is missing {missing!r}")
    return Price(input_rate, output_rate, rates["cached_input"])
