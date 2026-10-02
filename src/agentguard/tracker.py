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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .exceptions import GuardConfigError
from .pricing import Price, PriceTable

__all__ = [
    "Usage",
    "CallRecord",
    "ModelSummary",
    "AttributionSummary",
    "CostTracker",
    "UNATTRIBUTED",
    "extract_usage",
    "extract_model",
]

#: Bucket name for calls that carried no tag (or no tool). Kept as one literal so
#: the parts of an attribution breakdown always add up to the whole.
UNATTRIBUTED = "(unattributed)"

#: Format version for checkpointed state. A snapshot written by a different
#: layout is refused rather than half-read: a budget that silently restores the
#: wrong arithmetic is worse than one that refuses to restore at all.
SNAPSHOT_VERSION = 1


# --------------------------------------------------------------------------- #
# Usage extraction
# --------------------------------------------------------------------------- #

_INPUT_KEYS = (
    "prompt_tokens",
    "input_tokens",
    "prompt_token_count",
    "input_token_count",
    # AWS Bedrock's Converse API, which uses camelCase where every other provider
    # uses snake_case. Its usage object is the only place token counts appear for a
    # Bedrock agent, so without these a Bedrock run is billed as unpriced.
    "inputTokens",
    "promptTokens",
)
_OUTPUT_KEYS = (
    "completion_tokens",
    "output_tokens",
    "candidates_token_count",
    "output_token_count",
    "outputTokens",
    "completionTokens",
)
_CACHED_KEYS = (
    "cached_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "cache_read_tokens",
    "cacheReadInputTokens",
    "cachedTokens",
)
_REASONING_KEYS = (
    "reasoning_tokens",
    "reasoning_token_count",
    "thoughts_token_count",
    "reasoningTokens",
)


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
    tool: str | None = None
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
            "tool": self.tool,
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


@dataclass(frozen=True, slots=True)
class AttributionSummary:
    """Aggregated spend for one tag or one tool.

    ``name`` is the step's ``tag=`` or the name passed to ``with guard.tool(...)``,
    or :data:`UNATTRIBUTED` for calls that had neither — so the parts always add up
    to the whole, and money nobody labelled is visible rather than implied.
    """

    name: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    unpriced_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
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
        "_scope_totals",
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
        # Running per-tool and per-tag totals. Maintained as records arrive so a
        # scoped budget is O(1) to check rather than a scan of every call so far,
        # which would make an evaluated-per-call limit quadratic in the run length.
        self._scope_totals: dict[tuple[str, str], float] = {}

    # -- mutation ------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        usage: Usage,
        elapsed_s: float = 0.0,
        tag: str | None = None,
        step: int | None = None,
        tool: str | None = None,
        meta: Mapping[str, Any] | None = None,
        price: Price | None = None,
    ) -> CallRecord:
        """Account for one call and return its immutable record.

        Equivalent to :meth:`record_with_policy` for every ``on_unknown_model``
        mode except ``"error"``, which raises :class:`~agentguard.GuardConfigError`
        **after** the call has been recorded — so the spend is still in
        :attr:`total_usd` and in :attr:`records` when the caller catches it.
        """
        record, failure = self.record_with_policy(
            model=model,
            usage=usage,
            elapsed_s=elapsed_s,
            tag=tag,
            step=step,
            tool=tool,
            meta=meta,
            price=price,
        )
        if failure is not None:
            raise failure
        return record

    def record_with_policy(
        self,
        *,
        model: str,
        usage: Usage,
        elapsed_s: float = 0.0,
        tag: str | None = None,
        step: int | None = None,
        tool: str | None = None,
        meta: Mapping[str, Any] | None = None,
        price: Price | None = None,
    ) -> tuple[CallRecord, GuardConfigError | None]:
        """Account for one call, returning ``(record, failure)``.

        The record is **always** produced and always counted; ``failure`` is the
        error the caller should raise afterwards, and is ``None`` in every mode but
        ``on_unknown_model="error"``.

        Reporting the failure instead of raising it is what lets
        :meth:`agentguard.Guard.record` re-evaluate the limits and raise in the
        documented order. The call being recorded has already been made and already
        been paid for, so an exception raised *instead* of the record would take
        that spend out of the total, out of ``by_model``, and out of the report: a
        guard that quietly forgets money it was told about.
        """
        resolved = None if price is not None else self._table.resolve(model)
        effective = price or (resolved[1] if resolved else None) or self._default_price

        if effective is None:
            cost: float | None = None
            failure = self._handle_unknown_model(model)
        else:
            cost = effective.cost_usd(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
            )
            failure = None

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
                tool=tool,
                meta=dict(meta) if meta else {},
            )
            self._records.append(record)
            if cost is None:
                self._unpriced_calls += 1
            else:
                self._add_scope_totals(record.tool, record.tag, cost)
            return record, failure

    def _add_scope_totals(self, tool: str | None, tag: str | None, cost: float) -> None:
        """Fold one priced call into the running per-tool / per-tag totals.

        Only *priced* calls are added. An unpriced call has no cost to add, and
        counting it as ``0.0`` would put a bucket in :meth:`scope_totals` that a
        scoped budget reads as "this scope has spent nothing" rather than "this
        scope cannot be measured" — the same distinction the report draws between
        ``$0`` and *unpriced*.
        """
        for scope, name in (("tool", tool), ("tag", tag)):
            if name:
                key = (scope, name)
                self._scope_totals[key] = self._scope_totals.get(key, 0.0) + cost

    def scope_totals(self, scope: str) -> dict[str, float]:
        """Dollars spent per tool or per tag, for ``scope`` of ``"tool"``/``"tag"``.

        >>> tracker = CostTracker(on_unknown_model="ignore")
        >>> _ = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tool="search")
        >>> round(tracker.scope_totals("tool")["search"], 2)
        2.5
        """
        with self._lock:
            return {
                name: total for (kind, name), total in self._scope_totals.items() if kind == scope
            }

    def _handle_unknown_model(self, model: str) -> GuardConfigError | None:
        """Note that a model could not be priced; return the error to raise, if any.

        Never raises: see :meth:`record_with_policy` for why the caller raises once
        the record exists. In ``"warn"`` mode the warning is emitted here, once per
        model.
        """
        with self._lock:
            already_warned = model in self._models_warned
            self._models_warned.add(model)

        if self._on_unknown_model == "error":
            return GuardConfigError(
                f"no price known for model {model!r}, so its cost cannot be counted "
                f"against the budget and is recorded as unpriced. Add it via "
                f"Guard(pricing={{{model!r}: (input_per_1m, output_per_1m)}}), pass "
                f"a default_price=, or set on_unknown_model='warn'."
            )
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
        return None

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
        buckets = self._buckets(lambda record: record.canonical_model or record.model)
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

    def by_tag(self) -> dict[str, AttributionSummary]:
        """Per-tag totals, ordered by descending cost then name.

        The tag is the free-form label a step was opened with
        (``guard.step(tag="search")``) — the answer to "which part of my agent is
        eating the budget?". Calls recorded outside any tagged step land under
        :data:`UNATTRIBUTED`, so the breakdown still adds up to the total.

        >>> tracker = CostTracker(on_unknown_model="ignore")
        >>> _ = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tag="search")
        >>> _ = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tag="write")
        >>> {name: round(s.cost_usd, 2) for name, s in tracker.by_tag().items()}
        {'search': 2.5, 'write': 2.5}
        """
        return self._attribution(lambda record: record.tag)

    def by_tool(self) -> dict[str, AttributionSummary]:
        """Per-tool totals, ordered by descending cost then name.

        A call is attributed to the tool whose ``with guard.tool(...)`` block it was
        made in, which is what makes "which tool is eating my budget?" answerable
        without summing records by hand.

        >>> tracker = CostTracker(on_unknown_model="ignore")
        >>> _ = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tool="search")
        >>> round(tracker.by_tool()["search"].cost_usd, 2)
        2.5
        """
        return self._attribution(lambda record: record.tool)

    def _attribution(
        self, key_of: Callable[[CallRecord], str | None]
    ) -> dict[str, AttributionSummary]:
        buckets = self._buckets(lambda record: key_of(record) or UNATTRIBUTED)
        summaries = [
            AttributionSummary(
                name=name,
                calls=values["calls"],
                input_tokens=values["input_tokens"],
                output_tokens=values["output_tokens"],
                cost_usd=values["cost_usd"],
                unpriced_calls=values["unpriced_calls"],
            )
            for name, values in buckets.items()
        ]
        summaries.sort(key=lambda s: (-s.cost_usd, s.name))
        return {s.name: s for s in summaries}

    def _buckets(self, key_of: Callable[[CallRecord], str]) -> dict[str, dict[str, Any]]:
        """Group records by an arbitrary key, summing calls, tokens and dollars."""
        buckets: dict[str, dict[str, Any]] = {}
        with self._lock:
            for record in self._records:
                bucket = buckets.setdefault(
                    key_of(record),
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
        return buckets

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
                "by_tag": [s.as_dict() for s in self.by_tag().values()],
                "by_tool": [s.as_dict() for s in self.by_tool().values()],
                "records": [r.as_dict() for r in self._records],
            }

    # -- checkpointing -------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Calls grouped by ``(model, tag, tool)``, small enough to checkpoint.

        Deliberately **not** :meth:`as_dict`, which carries every
        :class:`CallRecord` — the wrong weight for something written on every loop
        iteration. Grouping rather than three separate marginal breakdowns is what
        makes the checkpoint invertible: :meth:`set_state` can rebuild records from
        this and recover ``by_model``, ``by_tag`` and ``by_tool`` exactly, whereas
        independent marginals would each be right only if the others were ignored.

        What is genuinely lost is per-call detail: individual records, their
        timestamps, their ``meta``, and the order they happened in.

        >>> tracker = CostTracker(on_unknown_model="ignore")
        >>> _ = tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tag="search")
        >>> tracker.get_state()["groups"][0]["calls"]
        1
        """
        with self._lock:
            groups: dict[tuple[str, str, str], dict[str, Any]] = {}
            for record in self._records:
                model = record.canonical_model or record.model
                key = (model, record.tag or UNATTRIBUTED, record.tool or UNATTRIBUTED)
                group = groups.setdefault(
                    key,
                    {
                        "calls": 0,
                        "unpriced_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
                group["calls"] += 1
                if not record.priced:
                    group["unpriced_calls"] += 1
                group["input_tokens"] += record.usage.input_tokens
                group["output_tokens"] += record.usage.output_tokens
                # Summed once here and carried as a single number, so the float
                # total survives the round trip instead of being re-added per
                # reconstructed record.
                group["cost_usd"] += record.cost_usd or 0.0

            rows = [
                {
                    "model": model,
                    "tag": tag,
                    "tool": tool,
                    **values,
                    "cost_usd": round(values["cost_usd"], 8),
                }
                for (model, tag, tool), values in groups.items()
            ]
            return {
                "version": SNAPSHOT_VERSION,
                "calls": len(self._records),
                "unpriced_calls": self._unpriced_calls,
                "groups": rows,
            }

    @classmethod
    def restore(
        cls,
        state: Mapping[str, Any],
        table: PriceTable | None = None,
        *,
        default_price: Price | None = None,
        on_unknown_model: str = "warn",
    ) -> CostTracker:
        """Rebuild a tracker from :meth:`get_state` output.

        >>> original = CostTracker(on_unknown_model="ignore")
        >>> _ = original.record(model="gpt-4o", usage=Usage(2_000_000, 0))
        >>> rebuilt = CostTracker.restore(original.get_state(), original.price_table,
        ...                               on_unknown_model="ignore")
        >>> round(rebuilt.total_usd, 2)
        5.0
        """
        tracker = cls(
            table,
            default_price=default_price,
            on_unknown_model=on_unknown_model,
        )
        tracker.set_state(state)
        return tracker

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Replace all accounting with a previously captured state.

        Group costs are divided evenly across the priced calls in the group, so a
        restored report keeps every breakdown total intact. Per-call costs are
        therefore an average rather than what each call really cost — which is
        exactly the granularity a checkpoint does not carry, and why the report
        says so rather than implying it recovered the original records.
        """
        records, unpriced_calls, warned = _read_tracker_state(state)

        with self._lock:
            self._records = records
            self._unpriced_calls = unpriced_calls
            self._models_warned = warned
            # Rebuilt from the restored records, so a scoped budget keeps counting
            # against what the run already spent before the checkpoint.
            totals: dict[tuple[str, str], float] = {}
            for record in records:
                if record.cost_usd is None:
                    continue
                for scope, name in (("tool", record.tool), ("tag", record.tag)):
                    if name:
                        key = (scope, name)
                        totals[key] = totals.get(key, 0.0) + record.cost_usd
            self._scope_totals = totals


def _read_int(source: Mapping[str, Any], key: str, context: str) -> int:
    value = source.get(key, 0)
    if isinstance(value, bool):
        raise GuardConfigError(f"{context}: {key} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise GuardConfigError(f"{context}: {key} must be an integer, got {value!r}")


def _read_label(
    source: Mapping[str, Any], key: str, context: str, *, required: bool = False
) -> str:
    value = source.get(key)
    if value is None:
        if required:
            raise GuardConfigError(f"{context}: {key} is required")
        return UNATTRIBUTED
    if not isinstance(value, str) or not value:
        raise GuardConfigError(f"{context}: {key} must be a non-empty string, got {value!r}")
    return value


def _restored_record(
    *,
    index: int,
    model: str,
    tag: str | None,
    tool: str | None,
    usage: Usage,
    cost_usd: float | None,
) -> CallRecord:
    """One record rebuilt from a snapshot group.

    ``at`` is ``0.0`` rather than the current time on purpose: a restored record was
    not observed now, and stamping it fresh would date it to the restore and make
    the report repeat a time that never happened.
    """
    return CallRecord(
        index=index,
        model=model,
        canonical_model=model,
        usage=usage,
        cost_usd=cost_usd,
        at=0.0,
        elapsed_s=0.0,
        tag=tag,
        tool=tool,
    )


def _read_tracker_state(
    state: Mapping[str, Any],
) -> tuple[list[CallRecord], int, set[str]]:
    """Validate a checkpoint and rebuild the records it describes.

    Every failure here is a refusal rather than a repair. A checkpoint that cannot
    be read exactly is one whose budget would be wrong, and a wrong budget is the
    single outcome this library exists to prevent.
    """
    if not isinstance(state, Mapping):
        raise GuardConfigError(f"tracker state must be a mapping, got {type(state).__name__}")
    version = state.get("version", SNAPSHOT_VERSION)
    if version != SNAPSHOT_VERSION:
        raise GuardConfigError(
            f"this snapshot was written in format version {version!r}, but this "
            f"version of agentguard reads {SNAPSHOT_VERSION}; a budget restored from "
            f"a format it does not understand would be wrong rather than merely "
            f"incomplete"
        )

    groups = state.get("groups")
    if not isinstance(groups, (list, tuple)):
        raise GuardConfigError("tracker state: 'groups' must be a list")

    records: list[CallRecord] = []
    unpriced_total = 0
    warned: set[str] = set()

    for position, raw in enumerate(groups):
        context = f"tracker state: groups[{position}]"
        if not isinstance(raw, Mapping):
            raise GuardConfigError(f"{context} must be a mapping, got {type(raw).__name__}")
        model = _read_label(raw, "model", context, required=True)
        tag = _read_label(raw, "tag", context)
        tool = _read_label(raw, "tool", context)
        calls = _read_int(raw, "calls", context)
        unpriced = _read_int(raw, "unpriced_calls", context)
        if calls < 0 or unpriced < 0:
            raise GuardConfigError(f"{context}: call counts must not be negative")
        if unpriced > calls:
            raise GuardConfigError(
                f"{context}: unpriced_calls ({unpriced}) exceeds calls ({calls})"
            )
        raw_cost = raw.get("cost_usd", 0.0)
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise GuardConfigError(f"{context}: cost_usd must be a number, got {raw_cost!r}")
        cost = float(raw_cost)
        priced_calls = calls - unpriced
        if cost and not priced_calls:
            raise GuardConfigError(
                f"{context}: cost_usd is {cost!r} but every call is unpriced; a model "
                f"with no price cannot have a cost"
            )
        priced_cost = cost / priced_calls if priced_calls else 0.0
        record_tag = None if tag == UNATTRIBUTED else tag
        record_tool = None if tool == UNATTRIBUTED else tool

        # A group can mix priced and unpriced calls of the same model — an unknown
        # model falls back to `default_price`, and a known one can be unpriced by a
        # `disable` entry. Both kinds must be reproduced, or the group's money lands
        # on a record the attribution breakdown then labels "unpriced".
        #
        # Token counts belong to the group, and the group has `calls` members with
        # no record of how the tokens were split. Giving them all to the first
        # record and none to the rest keeps every total and every breakdown exact;
        # only the per-record view is coarse, and that is the view a checkpoint
        # cannot honestly reconstruct.
        plan: list[float | None] = [None] * unpriced + [priced_cost] * priced_calls
        for position_in_group, record_cost in enumerate(plan):
            records.append(
                _restored_record(
                    index=len(records),
                    model=model,
                    tag=record_tag,
                    tool=record_tool,
                    usage=Usage(
                        input_tokens=(
                            _read_int(raw, "input_tokens", context) if position_in_group == 0 else 0
                        ),
                        output_tokens=(
                            _read_int(raw, "output_tokens", context)
                            if position_in_group == 0
                            else 0
                        ),
                    ),
                    cost_usd=record_cost,
                )
            )
        if unpriced:
            warned.add(model)
        unpriced_total += unpriced

    claimed = state.get("calls")
    if isinstance(claimed, int) and claimed != len(records):
        raise GuardConfigError(
            f"tracker state claims {claimed} call(s) but describes {len(records)}"
        )
    return records, unpriced_total, warned
