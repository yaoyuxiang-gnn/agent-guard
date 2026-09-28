"""Token and cost accounting.

The tracker is deliberately boring and defensive: it is the component an agent
loop calls on every single LLM response, so it is lock-protected, allocation-light
and never raises for a mere accounting problem. Anything it cannot price is
counted as *unpriced* rather than guessed, and surfaced in the report — a safety
tool that silently assumes ``$0.00`` would be worse than useless.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .pricing import Price, PriceTable

__all__ = [
    "Usage",
    "CallRecord",
    "ModelSummary",
    "CostTracker",
    "extract_usage",
    "extract_model",
]


# --------------------------------------------------------------------------- #
# Usage extraction
# --------------------------------------------------------------------------- #

_INPUT_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count")
_OUTPUT_KEYS = (
    "completion_tokens",
    "output_tokens",
    "candidates_token_count",
    "output_token_count",
)
_CACHED_KEYS = (
    "cached_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "cache_read_tokens",
)
_REASONING_KEYS = ("reasoning_tokens", "reasoning_token_count", "thoughts_token_count")

# Nested detail objects, tried when the flat key is absent.
_NESTED_CACHED = (
    ("prompt_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cached_tokens"),
)
_NESTED_REASONING = (
    ("completion_tokens_details", "reasoning_tokens"),
    ("output_tokens_details", "reasoning_tokens"),
)


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or an object, without raising."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_int(obj: Any, keys: tuple[str, ...]) -> int:
    for key in keys:
        value = _get(obj, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _first_nested_int(obj: Any, paths: tuple[tuple[str, str], ...]) -> int:
    for outer, inner in paths:
        value = _get(_get(obj, outer), inner)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


@dataclass(slots=True)
class Usage:
    """Token counts for a single LLM call.

    ``cached_input_tokens`` is a *subset* of ``input_tokens`` (how both OpenAI
    and Anthropic report cache hits), and ``reasoning_tokens`` is reported for
    information only — providers already include it in ``output_tokens``.

    >>> usage = Usage(input_tokens=1200, output_tokens=300)
    >>> usage.total_tokens
    1500
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Billable token count: input plus output."""
        return self.input_tokens + self.output_tokens

    @property
    def is_empty(self) -> bool:
        """True when nothing was reported — usually means extraction failed."""
        return not (self.input_tokens or self.output_tokens)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


def extract_usage(response: Any) -> Usage | None:
    """Pull token counts out of *any* provider response object.

    Understands OpenAI-style (``prompt_tokens``/``completion_tokens``),
    Anthropic-style (``input_tokens``/``output_tokens``) and Gemini-style
    (``prompt_token_count``/``candidates_token_count``) payloads, as either
    objects or plain dicts. Returns ``None`` when nothing usable is present, so
    callers can distinguish "no usage reported" from "zero tokens".
    """
    usage = _get(response, "usage")
    if usage is None:
        # Some SDKs put token counts on the response itself (Gemini).
        usage = _get(response, "usage_metadata")
    if usage is None:
        return None

    input_tokens = _first_int(usage, _INPUT_KEYS)
    output_tokens = _first_int(usage, _OUTPUT_KEYS)
    cached = _first_int(usage, _CACHED_KEYS) or _first_nested_int(usage, _NESTED_CACHED)
    reasoning = _first_int(usage, _REASONING_KEYS) or _first_nested_int(usage, _NESTED_REASONING)

    # Anthropic bills cache *writes* separately, but reports reads on the same
    # object. Only the read count is a discount, so ignore creation counts here.
    if input_tokens == 0 and output_tokens == 0 and cached == 0 and reasoning == 0:
        return None

    return Usage(
        input_tokens=max(0, input_tokens),
        output_tokens=max(0, output_tokens),
        cached_input_tokens=max(0, cached),
        reasoning_tokens=max(0, reasoning),
    )


def extract_model(response: Any, default: str | None = None) -> str | None:
    """Best-effort model name from a response object or mapping."""
    for key in ("model", "model_name", "model_id"):
        value = _get(response, key)
        if isinstance(value, str) and value:
            return value
    return default


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One accounted LLM call."""

    index: int
    model: str
    canonical_model: str | None
    usage: Usage
    cost_usd: float | None
    at: float
    elapsed_s: float
    tag: str | None = None
    step: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def priced(self) -> bool:
        """False when the model was not in the price table."""
        return self.cost_usd is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "model": self.model,
            "canonical_model": self.canonical_model,
            "cost_usd": self.cost_usd,
            "priced": self.priced,
            "tag": self.tag,
            "step": self.step,
            "elapsed_s": round(self.elapsed_s, 6),
            **self.usage.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Aggregated spend for one model."""

    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    unpriced_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "unpriced_calls": self.unpriced_calls,
        }


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #


class CostTracker:
    """Thread-safe accumulator of calls, tokens and dollars.

    Agents routinely fan out concurrent tool calls, so every mutation happens
    under a single re-entrant lock and every read returns a consistent snapshot.

    >>> tracker = CostTracker(on_unknown_model="ignore")
    >>> rec = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), elapsed_s=0.0)
    >>> rec.priced, round(tracker.total_usd, 2)
    (True, 2.5)
    >>> summary = tracker.record(model="who-knows", usage=Usage(100, 100), elapsed_s=0.0)
    >>> summary.priced, tracker.unpriced_calls
    (False, 1)

    The default ``on_unknown_model="warn"`` emits a :class:`RuntimeWarning` the
    first time an unpriced model is seen, so a missing price can never quietly
    hide an overspend.
    """

    __slots__ = (
        "_default_price",
        "_lock",
        "_models_warned",
        "_on_unknown_model",
        "_records",
        "_table",
        "_unpriced_calls",
    )

    def __init__(
        self,
        table: PriceTable | None = None,
        *,
        default_price: Price | None = None,
        on_unknown_model: str = "warn",
    ) -> None:
        if on_unknown_model not in ("warn", "error", "ignore"):
            raise ValueError(
                f"on_unknown_model must be 'warn', 'error' or 'ignore', got {on_unknown_model!r}"
            )
        self._lock = threading.RLock()
        self._records: list[CallRecord] = []
        self._table = table if table is not None else PriceTable()
        self._default_price = default_price
        self._on_unknown_model = on_unknown_model
        self._models_warned: set[str] = set()
        self._unpriced_calls = 0

    # -- mutation ------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        usage: Usage,
        elapsed_s: float = 0.0,
        tag: str | None = None,
        step: int | None = None,
        meta: Mapping[str, Any] | None = None,
        price: Price | None = None,
    ) -> CallRecord:
        """Account for one call and return its immutable record."""
        resolved = None if price is not None else self._table.resolve(model)
        effective = price or (resolved[1] if resolved else None) or self._default_price

        if effective is None:
            cost: float | None = None
            self._handle_unknown_model(model)
        else:
            cost = effective.cost_usd(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
            )

        with self._lock:
            record = CallRecord(
                index=len(self._records),
                model=model,
                canonical_model=resolved[0] if resolved else None,
                usage=usage,
                cost_usd=cost,
                at=time.time(),
                elapsed_s=elapsed_s,
                tag=tag,
                step=step,
                meta=dict(meta) if meta else {},
            )
            self._records.append(record)
            if cost is None:
                self._unpriced_calls += 1
            return record

    def _handle_unknown_model(self, model: str) -> None:
        if self._on_unknown_model == "error":
            from .exceptions import GuardConfigError

            raise GuardConfigError(
                f"no price known for model {model!r}. Add it via "
                f"Guard(pricing={{{model!r}: (input_per_1m, output_per_1m)}}), pass "
                f"a default_price=, or set on_unknown_model='warn'."
            )
        with self._lock:
            already_warned = model in self._models_warned
            self._models_warned.add(model)
        if self._on_unknown_model == "warn" and not already_warned:
            if model == "unknown":
                message = (
                    "agentguard could not determine the model for a recorded call, so "
                    "it counts as $0 and will not move the budget. Pass model= (or a "
                    "response carrying .model) to record it properly."
                )
            else:
                message = (
                    f"agentguard has no price for model {model!r}; its cost is excluded "
                    f"from the budget and counted as unpriced. Add it with "
                    f"`agentguard config set {model} <input> <output>`, a pricing "
                    f"override, or a default_price=."
                )
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    # -- read-only views -----------------------------------------------------

    @property
    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def calls(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd or 0.0 for r in self._records)

    @property
    def unpriced_calls(self) -> int:
        with self._lock:
            return self._unpriced_calls

    @property
    def unpriced_models(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted({r.model for r in self._records if not r.priced}))

    @property
    def usage(self) -> Usage:
        total = Usage()
        with self._lock:
            for record in self._records:
                total = total + record.usage
        return total

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def price_table(self) -> PriceTable:
        return self._table

    def by_model(self) -> dict[str, ModelSummary]:
        """Per-model totals, ordered by descending cost then name."""
        buckets: dict[str, dict[str, Any]] = {}
        with self._lock:
            for record in self._records:
                key = record.canonical_model or record.model
                bucket = buckets.setdefault(
                    key,
                    {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "unpriced_calls": 0,
                    },
                )
                bucket["calls"] += 1
                bucket["input_tokens"] += record.usage.input_tokens
                bucket["output_tokens"] += record.usage.output_tokens
                bucket["cost_usd"] += record.cost_usd or 0.0
                if not record.priced:
                    bucket["unpriced_calls"] += 1

        summaries = [
            ModelSummary(
                model=name,
                calls=values["calls"],
                input_tokens=values["input_tokens"],
                output_tokens=values["output_tokens"],
                cost_usd=values["cost_usd"],
                unpriced_calls=values["unpriced_calls"],
            )
            for name, values in buckets.items()
        ]
        summaries.sort(key=lambda s: (-s.cost_usd, s.model))
        return {s.model: s for s in summaries}

    def burn_rate_usd_per_step(self) -> float | None:
        """Average dollars spent per accounted call, or ``None`` if no calls."""
        with self._lock:
            if not self._records:
                return None
            total = sum(r.cost_usd or 0.0 for r in self._records)
            return total / len(self._records)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self._records),
                "unpriced_calls": self._unpriced_calls,
                "unpriced_models": sorted({r.model for r in self._records if not r.priced}),
                "cost_usd": round(sum(r.cost_usd or 0.0 for r in self._records), 8),
                "usage": self.usage.as_dict(),
                "by_model": [s.as_dict() for s in self.by_model().values()],
                "records": [r.as_dict() for r in self._records],
            }
