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
"""

from __future__ import annotations

import re
from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import dataclass

from .exceptions import GuardConfigError

__all__ = [
    "Price",
    "PriceTable",
    "DEFAULT_PRICING",
    "PRICING_AS_OF",
    "normalize_model_key",
]


#: Date the bundled price snapshot was last reviewed. Prices move; treat anything
#: older than a few months as indicative and pass your own ``pricing=`` overrides.
PRICING_AS_OF = "2026-01"


@dataclass(frozen=True, slots=True)
class Price:
    """List price for one model, in **US dollars per 1,000,000 tokens**.

    >>> price = Price(input_per_1m=2.50, output_per_1m=10.00)
    >>> round(price.cost_usd(input_tokens=1_000_000), 4)
    2.5
    >>> round(price.cost_usd(input_tokens=1_000, output_tokens=500), 6)
    0.0075
    """

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None

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
    """A model price lookup with user overrides.

    Overrides accept a :class:`Price`, a 2-tuple ``(input, output)``, or a
    3-tuple ``(input, output, cached_input)`` — always USD per 1M tokens.

    >>> table = PriceTable(overrides={"my-model": (1.0, 2.0)})
    >>> round(table.resolve_price("my-model").output_per_1m, 2)
    2.0
    >>> table.resolve_price("gpt-4o-2024-08-06").input_per_1m
    2.5
    >>> table.resolve_price("totally-unknown-model") is None
    True
    """

    __slots__ = ("_prices",)

    def __init__(
        self,
        base: Mapping[str, Price] | None = None,
        overrides: Mapping[str, Price | tuple[float, ...]] | None = None,
    ) -> None:
        prices = dict(DEFAULT_PRICING if base is None else base)
        if overrides:
            for name, spec in overrides.items():
                normalized = normalize_model_key(name)
                if not normalized:
                    raise GuardConfigError(f"pricing override has an empty model name: {name!r}")
                prices[normalized] = _coerce_price(name, spec)
        self._prices: dict[str, Price] = prices

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
        """Exact lookup on the normalized key only. No version fallback."""
        return self._prices.get(normalize_model_key(model))

    def resolve(self, model: str) -> tuple[str, Price] | None:
        """Resolve ``model`` to ``(canonical_name, price)``, or ``None``.

        Tries, in order: exact normalized key, then a conservative trailing
        version-strip. Never guesses between *different* model families.
        """
        key = normalize_model_key(model)
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


def _coerce_price(name: str, spec: Price | tuple[float, ...]) -> Price:
    if isinstance(spec, Price):
        return spec
    if isinstance(spec, (tuple, list)):
        try:
            values = [float(v) for v in spec]
        except (TypeError, ValueError) as exc:
            raise GuardConfigError(
                f"pricing override for {name!r} must contain numbers, got {spec!r}"
            ) from exc
        if len(values) == 2:
            return Price(values[0], values[1])
        if len(values) == 3:
            return Price(values[0], values[1], values[2])
        raise GuardConfigError(
            f"pricing override for {name!r} must be (input, output) or "
            f"(input, output, cached_input); got {len(values)} values"
        )
    raise GuardConfigError(
        f"pricing override for {name!r} must be a Price or a tuple of floats, "
        f"got {type(spec).__name__}"
    )
