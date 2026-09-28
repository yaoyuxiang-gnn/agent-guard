"""The :class:`Guard`: budget caps, limits and loop detection for one agent run.

A guard is a plain object with no I/O and no dependencies. It is cheap to create
one per request, per task, or per background job — which is the recommended usage,
because a global guard cannot tell two concurrent agents apart.

Three integration depths, from least to most magic::

    # 1. Manual: framework-agnostic, works with anything.
    guard = Guard(max_usd=1.0)
    guard.record("gpt-4o", input_tokens=1200, output_tokens=300)
    guard.check()

    # 2. Structured: one step per loop iteration, tools fingerprinted for you.
    with Guard(max_usd=1.0, max_steps=25) as guard:
        while True:
            with guard.step() as step:
                response = call_llm(...)
                step.record(response)
                with step.tool("search", {"q": query}):
                    ...

    # 3. Wrapped client: every call accounted automatically.
    from agentguard.adapters.openai import guard_openai
    client = guard_openai(OpenAI(), guard)
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from ._util import stable_json
from .exceptions import (
    BudgetExceeded,
    GuardConfigError,
    GuardTripped,
    LoopDetected,
    StepLimitExceeded,
    TimeLimitExceeded,
    TokenLimitExceeded,
)
from .loop import (
    Detector,
    LoopMonitor,
    LoopVerdict,
    call_signature,
    default_detectors,
    default_progress_detectors,
)
from .pricing import Price, PriceTable
from .report import Report, build_limits
from .tracker import CallRecord, CostTracker, Usage, extract_model, extract_usage

__all__ = ["Guard", "Step", "current_guard"]

_ON_TRIP_MODES = ("raise", "warn", "stop")

_current_guard: contextvars.ContextVar[Guard | None] = contextvars.ContextVar(
    "agentguard_current", default=None
)
_current_step: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "agentguard_step", default=None
)


def current_guard() -> Guard | None:
    """Return the guard installed by the innermost ``with guard:`` block.

    Useful deep inside an agent's call stack, where threading a guard parameter
    through every function would be noise::

        from agentguard import current_guard

        def my_tool(query: str) -> str:
            guard = current_guard()
            if guard is not None:
                guard.observe(f"my_tool({query!r})")
            ...
    """
    return _current_guard.get()


class Step:
    """One iteration of an agent loop, opened by ``with guard.step():``.

    The step index is pushed onto a :mod:`contextvars` context, so
    ``guard.record(...)`` called anywhere inside the block is attributed to the
    right step, even across ``await`` points and threads spawned by
    :mod:`asyncio`.

    >>> guard = Guard(max_steps=5)
    >>> with guard.step(tag="research") as step:
    ...     step.index, step.tag
    (1, 'research')
    """

    __slots__ = ("_guard", "_token", "index", "tag")

    def __init__(self, guard: Guard, index: int, tag: str | None = None) -> None:
        self._guard = guard
        self._token: contextvars.Token[int | None] | None = None
        #: 1-based step number.
        self.index = index
        #: Optional free-form label, carried onto every call recorded in the step.
        self.tag = tag

    def __enter__(self) -> Step:
        self._token = _current_step.set(self.index)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if self._token is not None:
            _current_step.reset(self._token)
            self._token = None
        return False

    def record(self, response: Any = None, **kwargs: Any) -> CallRecord:
        """Account for an LLM call made in this step. See :meth:`Guard.record`."""
        kwargs.setdefault("step", self.index)
        kwargs.setdefault("tag", self.tag)
        return self._guard.record(response, **kwargs)

    @contextmanager
    def tool(self, name: str, args: Any = None) -> Iterator[str]:
        """Fingerprint a tool call for loop detection. See :meth:`Guard.tool`."""
        with self._guard.tool(name, args, step=self.index) as signature:
            yield signature

    def observe(self, signature: str) -> None:
        """Feed a raw signature to the loop detectors."""
        self._guard.observe(signature, step=self.index)

    def progress(self, value: Any) -> None:
        """Report a progress marker. See :meth:`Guard.progress`."""
        self._guard.progress(value, step=self.index)

    def __repr__(self) -> str:
        return f"Step(index={self.index}, tag={self.tag!r})"


class Guard:
    """Caps spend, steps, tokens and wall-clock time for one agent run.

    Every limit is optional; a guard with no limits still detects loops and still
    produces a cost report. The most common configuration is a dollar cap plus a
    step cap, because those are the two failure modes that actually cost money::

        Guard(max_usd=1.00, max_steps=25)

    :param max_usd: Stop once total spend exceeds this many US dollars.
    :param max_tokens: Stop once total input+output tokens exceed this count.
    :param max_steps: Stop once this many steps have been opened.
    :param max_seconds: Stop once this much wall-clock time has elapsed.
    :param on_trip: What to do when a limit fires. ``"raise"`` (default) raises
        the trip exception; ``"warn"`` emits a :class:`RuntimeWarning` and keeps
        going; ``"stop"`` sets :attr:`stopped` and calls ``on_trip_callback``
        without raising, for loops that prefer to break out themselves.
    :param pricing: Extra or overriding prices, as ``{model: Price}`` or
        ``{model: (input_per_1m, output_per_1m)}``.
    :param price_table: A prebuilt :class:`~agentguard.PriceTable` to use instead.
    :param default_price: Price to assume for models with no entry. When omitted,
        unknown models are counted as *unpriced* rather than guessed.
    :param on_unknown_model: ``"warn"`` (default), ``"error"`` or ``"ignore"``.
    :param loop_detection: Set ``False`` to disable detectors entirely.
    :param detectors: Replace the default action detectors. See :mod:`agentguard.loop`.
    :param progress_detectors: Replace the default progress detectors.
    :param on_trip_callback: Called once, with the trip exception, the first time
        a limit fires. Ideal for alerting or for gracefully stopping a remote job.
    :param name: Label used in :meth:`report` output.
    :param clock: Monotonic clock used for the time limit. Injectable for tests.
    """

    __slots__ = (
        "_action_monitor",
        "_clock",
        "_created_at",
        "_default_price",
        "_lock",
        "_max_seconds",
        "_max_steps",
        "_max_tokens",
        "_max_usd",
        "_name",
        "_on_trip",
        "_on_trip_callback",
        "_on_unknown_model",
        "_progress_monitor",
        "_steps",
        "_tokens",
        "_tracker",
        "_tripped",
        "_warned_empty",
    )

    def __init__(
        self,
        *,
        max_usd: float | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        on_trip: str = "raise",
        on_trip_callback: Callable[[GuardTripped], None] | None = None,
        name: str | None = None,
        pricing: Mapping[str, Price | tuple[float, ...]] | None = None,
        price_table: PriceTable | None = None,
        default_price: Price | None = None,
        on_unknown_model: str = "warn",
        loop_detection: bool = True,
        detectors: Sequence[Detector] | None = None,
        progress_detectors: Sequence[Detector] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_limits(
            max_usd=max_usd,
            max_tokens=max_tokens,
            max_steps=max_steps,
            max_seconds=max_seconds,
            on_trip=on_trip,
        )

        self._lock = threading.RLock()
        self._clock = clock
        self._created_at = clock()
        self._steps = 0
        self._name = name
        self._max_usd = float(max_usd) if max_usd is not None else None
        self._max_tokens = int(max_tokens) if max_tokens is not None else None
        self._max_steps = int(max_steps) if max_steps is not None else None
        self._max_seconds = float(max_seconds) if max_seconds is not None else None
        self._on_trip = on_trip
        self._on_trip_callback = on_trip_callback
        self._on_unknown_model = on_unknown_model
        self._default_price = default_price
        self._tripped: GuardTripped | None = None
        self._tokens: list[contextvars.Token[Any]] = []
        self._warned_empty = False

        table = price_table if price_table is not None else PriceTable(overrides=pricing)
        self._tracker = CostTracker(
            table,
            default_price=default_price,
            on_unknown_model=on_unknown_model,
        )

        if not loop_detection:
            self._action_monitor = LoopMonitor([])
            self._progress_monitor = LoopMonitor([])
        else:
            self._action_monitor = LoopMonitor(
                list(detectors) if detectors is not None else default_detectors()
            )
            self._progress_monitor = LoopMonitor(
                list(progress_detectors)
                if progress_detectors is not None
                else default_progress_detectors()
            )

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> Guard:
        self._tokens.append(_current_guard.set(self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if self._tokens:
            _current_guard.reset(self._tokens.pop())
        return False

    def reset(self) -> None:
        """Clear all counters, spend and detector state, and restart the clock.

        Useful when one guard object is reused across independent runs — a worker
        processing a queue, for example.
        """
        with self._lock:
            self._created_at = self._clock()
            self._steps = 0
            self._tripped = None
            self._warned_empty = False
            self._tracker = CostTracker(
                self._tracker.price_table,
                default_price=self._default_price,
                on_unknown_model=self._on_unknown_model,
            )
            self._action_monitor.reset()
            self._progress_monitor.reset()

    # -- read-only state -----------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def elapsed_s(self) -> float:
        """Seconds since this guard was created."""
        return self._clock() - self._created_at

    @property
    def steps(self) -> int:
        """Number of steps opened so far."""
        return self._steps

    @property
    def calls(self) -> int:
        """Number of LLM calls accounted so far."""
        return self._tracker.calls

    @property
    def spent_usd(self) -> float:
        """Total known spend in USD."""
        return self._tracker.total_usd

    @property
    def remaining_usd(self) -> float | None:
        """Dollars left before the budget trips, or ``None`` if uncapped."""
        if self._max_usd is None:
            return None
        return max(0.0, self._max_usd - self.spent_usd)

    @property
    def usage(self) -> Usage:
        """Aggregate token usage."""
        return self._tracker.usage

    @property
    def tracker(self) -> CostTracker:
        """The underlying :class:`~agentguard.CostTracker`."""
        return self._tracker

    @property
    def stopped(self) -> bool:
        """True once any limit has fired, regardless of ``on_trip`` mode."""
        return self._tripped is not None

    @property
    def tripped(self) -> GuardTripped | None:
        """The exception that stopped this run, if any."""
        return self._tripped

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The active action detectors."""
        return self._action_monitor.detectors

    # -- steps ---------------------------------------------------------------

    def step(self, tag: str | None = None) -> Step:
        """Open one agent step. Use as ``with guard.step() as step:``.

        Raises immediately if the run has already tripped, so a loop that ignores
        the exception cannot silently keep burning budget.
        """
        self.raise_if_tripped()
        with self._lock:
            self._steps += 1
            index = self._steps
        self._evaluate()
        return Step(self, index, tag)

    # -- accounting ----------------------------------------------------------

    def record(
        self,
        response: Any = None,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        tag: str | None = None,
        step: int | None = None,
        meta: Mapping[str, Any] | None = None,
        price: Price | None = None,
    ) -> CallRecord:
        """Account for one LLM call, then re-check every limit.

        The first positional argument accepts **either** a raw provider response
        **or** a model name — a bare string is always read as the model, since no
        SDK ever returns one as a response::

            guard.record(response)                                     # extract everything
            guard.record("gpt-4o", input_tokens=1200, output_tokens=300)
            guard.record(response, model="my-proxy-alias")

        Token counts and the model name are extracted from ``response`` when
        present; explicit keyword arguments always win, so you can supply counts
        for a provider agent-guard does not recognise.
        """
        self.raise_if_tripped()
        if isinstance(response, str):
            if model is None:
                model = response
            response = None

        usage = extract_usage(response) or Usage()

        if input_tokens is not None:
            usage.input_tokens = max(0, int(input_tokens))
        if output_tokens is not None:
            usage.output_tokens = max(0, int(output_tokens))
        if cached_input_tokens is not None:
            usage.cached_input_tokens = max(0, int(cached_input_tokens))
        if reasoning_tokens is not None:
            usage.reasoning_tokens = max(0, int(reasoning_tokens))

        resolved_model = model or extract_model(response) or "unknown"
        step_index = step if step is not None else _current_step.get()

        # A response that reports no usage at all is the single most dangerous
        # silent failure for a budget guard: the call happens, the cap never
        # moves, and the overspend is invisible. Say so once, loudly.
        if (
            response is not None
            and usage.is_empty
            and input_tokens is None
            and output_tokens is None
            and not self._warned_empty
        ):
            self._warned_empty = True
            warnings.warn(
                "agentguard could not find token usage on the recorded response; "
                "this call counts as $0 and will not move the budget. Pass "
                "input_tokens=/output_tokens= explicitly, or use "
                "Guard.record(...) with the counts your provider reports.",
                RuntimeWarning,
                stacklevel=2,
            )

        record = self._tracker.record(
            model=resolved_model,
            usage=usage,
            elapsed_s=self.elapsed_s,
            tag=tag,
            step=step_index,
            meta=meta,
            price=price,
        )
        self._evaluate()
        return record

    def preflight(
        self,
        model: str,
        *,
        input_tokens: int,
        max_output_tokens: int = 1024,
        price: Price | None = None,
    ) -> float:
        """Refuse a call *before* making it if its worst case cannot fit.

        Returns the worst-case cost in USD when the call is affordable. Raises
        :class:`~agentguard.BudgetExceeded` when it is not — which is the only
        way to stop an expensive single call from overshooting a budget that a
        post-hoc check could only report after the money was gone.

        Models with no known price return ``0.0``: agent-guard will not guess a
        rate, and the call is flagged as unpriced afterwards instead.
        """
        resolved = price or self._tracker.price_table.resolve_price(model)
        if resolved is None:
            resolved = self._default_price
        if resolved is None:
            return 0.0

        worst = resolved.worst_case_usd(
            input_tokens=input_tokens, max_output_tokens=max_output_tokens
        )
        if self._max_usd is not None:
            projected = self.spent_usd + worst
            if projected > self._max_usd:
                self._trip(
                    BudgetExceeded(
                        self.spent_usd,
                        self._max_usd,
                        projected_usd=projected,
                        model=model,
                        call_cost_usd=worst,
                    )
                )
        return worst

    # -- loop detection ------------------------------------------------------

    @staticmethod
    def call_signature(name: str, args: Any = None) -> str:
        """Build a stable fingerprint for a tool call. See :func:`agentguard.call_signature`."""
        return call_signature(name, args)

    def observe(self, signature: str, *, step: int | None = None) -> None:
        """Feed one action signature to the loop detectors."""
        if not self._action_monitor:
            return
        index = step if step is not None else (_current_step.get() or self._steps)
        verdict = self._action_monitor.observe(signature, index)
        if verdict is not None:
            self._trip(_loop_exception(verdict))

    @contextmanager
    def tool(self, name: str, args: Any = None, *, step: int | None = None) -> Iterator[str]:
        """Fingerprint a tool call for loop detection, then run it::

            with guard.tool("search", {"q": query}) as signature:
                result = search(query)

        The signature is computed *before* the tool runs, so a looping agent is
        stopped before it executes the same side effect a fourth time.
        """
        signature = call_signature(name, args)
        self.observe(signature, step=step)
        yield signature

    def progress(self, value: Any, *, step: int | None = None) -> None:
        """Report a progress marker, so stagnation can be detected.

        Only your agent knows what progress means, so this is explicit::

            guard.progress(len(rows_written))   # unchanged 6 times -> trip
        """
        if not self._progress_monitor:
            return
        index = step if step is not None else (_current_step.get() or self._steps)
        verdict = self._progress_monitor.observe(stable_json(value), index)
        if verdict is not None:
            self._trip(_loop_exception(verdict))

    # -- limits --------------------------------------------------------------

    def check(self) -> None:
        """Re-evaluate every limit. Safe to call anywhere, cheap, idempotent."""
        self._evaluate()

    def raise_if_tripped(self) -> None:
        """Raise the stored trip if one exists and ``on_trip="raise"``."""
        if self._tripped is not None and self._on_trip == "raise":
            raise self._tripped

    def _evaluate(self) -> None:
        if self._tripped is not None:
            return

        spent = self._tracker.total_usd
        if self._max_usd is not None and spent > self._max_usd:
            self._trip(BudgetExceeded(spent, self._max_usd))
            return

        tokens = self._tracker.total_tokens
        if self._max_tokens is not None and tokens > self._max_tokens:
            self._trip(TokenLimitExceeded(tokens, self._max_tokens))
            return

        if self._max_steps is not None and self._steps > self._max_steps:
            self._trip(StepLimitExceeded(self._steps, self._max_steps))
            return

        elapsed = self.elapsed_s
        if self._max_seconds is not None and elapsed > self._max_seconds:
            self._trip(TimeLimitExceeded(elapsed, self._max_seconds))

    def _trip(self, exc: GuardTripped) -> None:
        """Record a trip, then honour the configured ``on_trip`` mode."""
        first = False
        with self._lock:
            if self._tripped is None:
                self._tripped = exc
                first = True
            stored = self._tripped or exc

        if first and self._on_trip_callback is not None:
            try:
                self._on_trip_callback(stored)
            except Exception:  # a broken callback must not mask the trip
                warnings.warn(
                    "agentguard on_trip_callback raised; the trip is still recorded",
                    RuntimeWarning,
                    stacklevel=3,
                )

        if self._on_trip == "raise":
            raise stored
        if self._on_trip == "warn" and first:
            warnings.warn(str(stored), RuntimeWarning, stacklevel=3)
        # "stop": the trip is recorded and `stopped` becomes True; the caller's
        # loop is responsible for checking it and breaking out.

    # -- reporting -----------------------------------------------------------

    def report(self) -> Report:
        """Snapshot this run as a :class:`~agentguard.Report`."""
        by_model = tuple(self._tracker.by_model().values())
        trip_verdict: LoopVerdict | None = None
        tripped_reason: str | None = None

        if isinstance(self._tripped, LoopDetected):
            trip_verdict = LoopVerdict(
                kind=self._tripped.kind,
                detail=self._tripped.detail,
                signature=self._tripped.signature,
                count=self._tripped.count,
                step=self._tripped.step,
            )
        elif self._tripped is not None:
            tripped_reason = self._tripped.reason

        return Report(
            name=self._name,
            elapsed_s=self.elapsed_s,
            steps=self._steps,
            calls=self._tracker.calls,
            usage=self._tracker.usage,
            cost_usd=self._tracker.total_usd,
            limits=build_limits(
                cost_usd=self._tracker.total_usd,
                max_usd=self._max_usd,
                total_tokens=self._tracker.total_tokens,
                max_tokens=self._max_tokens,
                steps=self._steps,
                max_steps=self._max_steps,
                elapsed_s=self.elapsed_s,
                max_seconds=self._max_seconds,
            ),
            by_model=by_model,
            unpriced_models=self._tracker.unpriced_models,
            unpriced_calls=self._tracker.unpriced_calls,
            trip=trip_verdict,
            tripped_reason=tripped_reason,
        )

    def as_dict(self) -> dict[str, Any]:
        """The report as a JSON-serialisable dict."""
        return self.report().as_dict()

    def to_json(self, *, indent: int = 2) -> str:
        """The report serialised to JSON."""
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def save(self, path: str | os.PathLike[str], *, indent: int = 2) -> Path:
        """Write the JSON report to ``path``.

        Pairs with the bundled CLI, which renders a saved report without needing
        agent-guard installed in the reading process::

            guard.save("run.json")

            $ agentguard report run.json
        """
        target = Path(path)
        # newline="\n" so a report written on Windows is byte-identical to one
        # written on Linux, and CI diffs stay meaningful.
        target.write_text(self.to_json(indent=indent), encoding="utf-8", newline="\n")
        return target

    def __repr__(self) -> str:
        bits = [f"steps={self._steps}", f"calls={self._tracker.calls}"]
        if self._max_usd is not None:
            bits.append(f"spent=${self.spent_usd:.4f}/${self._max_usd:.4f}")
        if self._tripped is not None:
            bits.append(f"tripped={self._tripped.reason}")
        return f"Guard({', '.join(bits)})"


def _loop_exception(verdict: LoopVerdict) -> LoopDetected:
    return LoopDetected(
        kind=verdict.kind,
        detail=verdict.detail,
        signature=verdict.signature,
        count=verdict.count,
        step=verdict.step,
    )


def _validate_limits(
    *,
    max_usd: float | None,
    max_tokens: int | None,
    max_steps: int | None,
    max_seconds: float | None,
    on_trip: str,
) -> None:
    """Fail loudly at construction time rather than mid-run."""
    for label, value in (
        ("max_usd", max_usd),
        ("max_tokens", max_tokens),
        ("max_steps", max_steps),
        ("max_seconds", max_seconds),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GuardConfigError(f"{label} must be a number, got {type(value).__name__}")
        if value <= 0:
            raise GuardConfigError(f"{label} must be > 0, got {value!r}")
    if on_trip not in _ON_TRIP_MODES:
        raise GuardConfigError(
            f"on_trip must be one of {', '.join(map(repr, _ON_TRIP_MODES))}; got {on_trip!r}"
        )
