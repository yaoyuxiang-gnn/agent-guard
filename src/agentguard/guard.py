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
from .config import PricingConfig, load_config
from .exceptions import (
    BudgetExceeded,
    GuardConfigError,
    GuardStopped,
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
from .tracker import (
    SNAPSHOT_VERSION,
    CallRecord,
    CostTracker,
    Usage,
    extract_model,
    extract_usage,
)

__all__ = ["Guard", "Step", "current_guard"]

_ON_TRIP_MODES = ("raise", "warn", "stop")

_current_guard: contextvars.ContextVar[Guard | None] = contextvars.ContextVar(
    "agentguard_current", default=None
)
_current_step: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "agentguard_step", default=None
)
#: The tool whose ``with guard.tool(...)`` block is on the stack, so a call made
#: inside it can be attributed to that tool without the call site repeating itself.
_current_tool: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentguard_tool", default=None
)
# Tokens returned by ``_current_guard.set`` can only be reset in the context that
# created them, so the entry stack must be context-local too. A plain list on the
# Guard breaks a shared guard entered concurrently (two threads, or two asyncio
# tasks, ``with guard:``): one side pops the other's token and ``reset`` raises
# ValueError. An immutable tuple per context keeps each thread/task on its own
# stack while nesting stays LIFO within one context.
_entry_tokens: contextvars.ContextVar[tuple[contextvars.Token[Any], ...]] = contextvars.ContextVar(
    "agentguard_entry_tokens", default=()
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
        the trip exception at that moment; ``"warn"`` emits a
        :class:`RuntimeWarning` and keeps going, for measuring before enforcing;
        ``"stop"`` records the trip, calls :paramref:`on_trip_callback` and lets
        the current step unwind, then raises
        :class:`~agentguard.GuardStopped` from every entry point afterwards, so a
        loop that forgets to check :attr:`stopped` still stops.
    :param pricing: Extra or overriding prices, as ``{model: Price}``,
        ``{model: (input_per_1m, output_per_1m)}`` or
        ``{model: {"input": .., "output": ..}}``.
    :param aliases: Names your provider reports that should resolve to another
        model, as ``{reported_name: priced_name}``.
    :param disable: Bundled model names to drop, so they count as *unpriced*
        instead of being billed at a price you do not trust.
    :param price_table: A prebuilt :class:`~agentguard.PriceTable` to use instead
        of building one. Config discovery is skipped when this is passed.
    :param use_config: Read the user's ``agentguard.json`` /
        ``$AGENTGUARD_CONFIG`` pricing config (default ``True``). Set ``False`` to
        use only the bundled table and this constructor's arguments. A config file
        inside the project tree is only read when
        ``$AGENTGUARD_TRUST_PROJECT_CONFIG=1`` is set, because it travels with the
        repository rather than with you.
    :param config_path: Load exactly this config file instead of discovering one.
    :param config: A prebuilt :class:`~agentguard.PricingConfig`, bypassing
        discovery. ``Guard(pricing=...)`` still wins over anything from a file.
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
        "_checkpointed_calls",
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
        "_pricing_config",
        "_progress_monitor",
        "_skipped_detectors",
        "_steps",
        "_tracker",
        "_tripped",
        "_warned_empty",
        "_warned_skipped_detectors",
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
        pricing: Mapping[str, Price | tuple[float, ...] | Mapping[str, Any]] | None = None,
        aliases: Mapping[str, str] | None = None,
        disable: Sequence[str] = (),
        price_table: PriceTable | None = None,
        use_config: bool = True,
        config_path: str | Path | None = None,
        config: PricingConfig | None = None,
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
        self._warned_empty = False
        # Calls this run was handed by a restored checkpoint, and the detectors
        # whose history a checkpoint could not carry. Both exist so the report can
        # say what it did not observe instead of implying it observed everything.
        self._checkpointed_calls = 0
        self._skipped_detectors = 0
        self._warned_skipped_detectors = False

        if price_table is not None:
            # A prebuilt table is the caller's complete answer to "what do things
            # cost?", so config discovery is skipped rather than silently merged
            # underneath it.
            table = price_table
            self._pricing_config = PricingConfig(sources=price_table.sources)
        else:
            self._pricing_config = self._load_pricing_config(config, config_path, use_config)
            table = PriceTable.from_config(
                self._pricing_config,
                overrides=pricing,
                aliases=aliases,
                disable=disable,
            )
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

    @staticmethod
    def _load_pricing_config(
        config: PricingConfig | None,
        config_path: str | Path | None,
        use_config: bool,
    ) -> PricingConfig:
        """Pick the pricing config: explicit object, explicit path, or discovery.

        An explicitly passed ``config`` wins over everything, an explicit
        ``config_path`` is loaded exactly as given (a missing file is an error,
        not an empty config), and discovery is what ``use_config=True`` means.
        """
        if config is not None:
            return config
        if config_path is not None:
            return load_config(config_path)
        if not use_config:
            return PricingConfig()
        return load_config()

    def __enter__(self) -> Guard:
        _entry_tokens.set((*_entry_tokens.get(), _current_guard.set(self)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        tokens = _entry_tokens.get()
        if tokens:
            _entry_tokens.set(tokens[:-1])
            _current_guard.reset(tokens[-1])
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
            self._checkpointed_calls = 0
            self._skipped_detectors = 0
            self._warned_skipped_detectors = False
            self._tracker = CostTracker(
                self._tracker.price_table,
                default_price=self._default_price,
                on_unknown_model=self._on_unknown_model,
            )
            self._action_monitor.reset()
            self._progress_monitor.reset()

    # -- checkpointing -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """This run's counters as a small, JSON-serialisable dict.

        Pairs with :meth:`restore` so an agent that checkpoints its own state can
        checkpoint its spend too, and a resumed run does not start with a fresh
        budget::

            state = {"cursor": 41, "guard": guard.snapshot()}
            write_checkpoint(state)

            # ... later, in a new process ...
            guard = Guard.from_snapshot(read_checkpoint()["guard"], max_usd=5.0)
            guard.remaining_usd      # what is actually left, not the full budget

        The snapshot is **counters, not records**: calls are grouped by
        ``(model, tag, tool)``, so ``by_model``, ``by_tag`` and ``by_tool`` all
        survive exactly, while per-call detail (timestamps, ``meta``, the order
        calls happened in) does not. That is a deliberate size decision — a
        checkpoint written on every loop iteration cannot carry a full call log —
        and :meth:`agentguard.Guard.report` says which numbers it inherited rather
        than presenting them as its own observations.

        Two things are deliberately *not* checkpointed:

        * **Detector windows are** (they describe work already observed, and a loop
          that spans a checkpoint is still a loop).
        * **Wall-clock time is not.** ``max_seconds`` caps how long *this process*
          may run, so restoring an elapsed duration would make a resumed run trip
          on time it did not spend. Money and steps accumulate; the clock restarts.
        """
        with self._lock:
            return {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": self._steps},
                "tracker": self._tracker.get_state(),
                "detectors": {
                    "actions": self._action_monitor.get_states(),
                    "progress": self._progress_monitor.get_states(),
                },
            }

    def as_snapshot(self, *, indent: int | None = None) -> str:
        """The snapshot serialised to JSON, for writing into your own checkpoint."""
        # newline=/encoding are the writer's business; this only produces the text.
        return json.dumps(self.snapshot(), indent=indent, sort_keys=False)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | str,
        *,
        max_usd: float | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        **kwargs: Any,
    ) -> Guard:
        """Build a guard that continues the run a snapshot came from.

        Every argument takes the same meaning as on :class:`Guard`, so the limits
        are stated once here rather than being carried in the checkpoint — a
        budget is a property of the run you are starting, and letting a stale
        checkpoint re-impose yesterday's cap would be the wrong default::

            guard = Guard.from_snapshot(payload, max_usd=5.0, max_steps=100)
            guard.spent_usd          # carried over
            guard.steps              # carried over
            guard.elapsed_s          # starts at 0 — see Guard.snapshot
        """
        parsed = _parse_snapshot(snapshot)
        guard = cls(
            max_usd=max_usd,
            max_tokens=max_tokens,
            max_steps=max_steps,
            max_seconds=max_seconds,
            **kwargs,
        )
        guard.restore(parsed)
        return guard

    def restore(self, snapshot: Mapping[str, Any] | str) -> None:
        """Adopt a snapshot's counters, replacing whatever this guard had.

        Raises :class:`~agentguard.GuardConfigError` for a snapshot this version
        cannot read exactly. Refusing is the point: a budget that restores
        *approximately* is a budget that might not stop.
        """
        parsed = _parse_snapshot(snapshot)
        version = parsed.get("version", SNAPSHOT_VERSION)
        if version != SNAPSHOT_VERSION:
            raise GuardConfigError(
                f"this snapshot was written in format version {version!r}, but this "
                f"version of agentguard reads {SNAPSHOT_VERSION}"
            )

        guard_state = parsed.get("guard") or {}
        if not isinstance(guard_state, Mapping):
            raise GuardConfigError(
                f"snapshot 'guard' must be a mapping, got {type(guard_state).__name__}"
            )
        steps = guard_state.get("steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise GuardConfigError(
                f"snapshot 'guard.steps' must be a non-negative int, got {steps!r}"
            )

        tracker_state = parsed.get("tracker")
        if not isinstance(tracker_state, Mapping):
            raise GuardConfigError("snapshot is missing its 'tracker' state")

        detectors = parsed.get("detectors") or {}
        if not isinstance(detectors, Mapping):
            raise GuardConfigError("snapshot 'detectors' must be a mapping")
        actions = _read_detector_states(detectors.get("actions"))
        progress = _read_detector_states(detectors.get("progress"))

        with self._lock:
            # Trackers are rebuilt rather than mutated, so a failed restore cannot
            # leave a half-adopted budget behind: the same reason the validation
            # above runs before anything is assigned.
            self._tracker = CostTracker.restore(
                tracker_state,
                self._tracker.price_table,
                default_price=self._default_price,
                on_unknown_model=self._on_unknown_model,
            )
            self._steps = steps
            self._created_at = self._clock()
            self._tripped = None
            self._warned_empty = False
            self._checkpointed_calls = self._tracker.calls
            self._action_monitor.set_states(actions)
            self._progress_monitor.set_states(progress)
            self._skipped_detectors = sum(1 for state in (*actions, *progress) if state is None)
            self._warned_skipped_detectors = False

        if self._skipped_detectors:
            self._warn_skipped_detectors()

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
    def price_table(self) -> PriceTable:
        """The price lookup this guard bills with."""
        return self._tracker.price_table

    @property
    def pricing_config(self) -> PricingConfig:
        """The user config this guard was built from (empty when there is none).

        ``guard.pricing_config.sources`` names the files that were read, which is
        the quickest way to answer "where did this price come from?".
        """
        return self._pricing_config

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
        tool: str | None = None,
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

        Accounting happens *before* any trip is raised: the call already went out,
        so its cost is real, and an exception that skipped the bookkeeping would
        quietly remove spent money from the report. A call that *trips* a limit is
        recorded and returns, so ``on_trip="stop"`` can finish the current step; the
        next call is refused with :class:`~agentguard.GuardStopped`.
        """
        already_stopped = self._tripped is not None
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
        tool_name = tool if tool is not None else _current_tool.get()

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
            tool=tool_name,
            meta=meta,
            price=price,
        )
        self._evaluate()
        if already_stopped:
            self.raise_if_tripped()
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
        self.raise_if_tripped()
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

    def _warn_skipped_detectors(self) -> None:
        """Say once that a checkpoint could not carry every detector's history."""
        if self._warned_skipped_detectors or not self._skipped_detectors:
            return
        self._warned_skipped_detectors = True
        count = self._skipped_detectors
        warnings.warn(
            f"{count} detector(s) in this guard do not implement get_state/set_state, "
            f"so their observation history was not restored from the checkpoint; a "
            f"loop that began before it may need more observations before it trips. "
            f"The budget, step count and accounting are unaffected.",
            RuntimeWarning,
            stacklevel=3,
        )

    # -- loop detection ------------------------------------------------------

    @staticmethod
    def call_signature(name: str, args: Any = None) -> str:
        """Build a stable fingerprint for a tool call. See :func:`agentguard.call_signature`."""
        return call_signature(name, args)

    def observe(self, signature: str, *, step: int | None = None) -> None:
        """Feed one action signature to the loop detectors."""
        self.raise_if_tripped()
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
        stopped before it executes the same side effect a fourth time — and a guard
        that has already tripped refuses to enter at all, so a stopped run cannot
        keep performing side effects.

        While the block is open, every call recorded inside it is attributed to
        ``name``, which is what makes ``report().by_tool()`` answer "which tool is
        eating my budget?". Nested blocks attribute to the innermost tool.
        """
        self.raise_if_tripped()
        signature = call_signature(name, args)
        self.observe(signature, step=step)
        token = _current_tool.set(name)
        try:
            yield signature
        finally:
            _current_tool.reset(token)

    def progress(self, value: Any, *, step: int | None = None) -> None:
        """Report a progress marker, so stagnation can be detected.

        Only your agent knows what progress means, so this is explicit::

            guard.progress(len(rows_written))   # unchanged 6 times -> trip
        """
        self.raise_if_tripped()
        if not self._progress_monitor:
            return
        index = step if step is not None else (_current_step.get() or self._steps)
        verdict = self._progress_monitor.observe(stable_json(value), index)
        if verdict is not None:
            self._trip(_loop_exception(verdict))

    # -- limits --------------------------------------------------------------

    def check(self) -> None:
        """Re-evaluate every limit, and raise if the guard has tripped.

        Safe to call anywhere in a loop — it is cheap and does not double-count —
        which is what makes it the natural hook for the manual integration::

            while not guard.stopped:
                ...
                guard.check()

        Under ``on_trip="raise"`` it raises the trip; under ``"stop"`` it raises
        :class:`~agentguard.GuardStopped`; under ``"warn"`` it only warns.
        """
        self._evaluate()
        self.raise_if_tripped()

    def raise_if_tripped(self) -> None:
        """Raise if this guard has tripped, following the ``on_trip`` mode.

        ``"raise"`` re-raises the trip itself. ``"stop"`` raises
        :class:`~agentguard.GuardStopped`, which carries the trip as
        :attr:`~agentguard.GuardStopped.cause` — so a loop that never checks
        :attr:`stopped` still stops, rather than quietly spending on. ``"warn"``
        raises nothing: measuring before enforcing is the point of that mode.
        """
        tripped = self._tripped
        if tripped is None:
            return
        if self._on_trip == "raise":
            raise tripped
        if self._on_trip == "stop":
            raise GuardStopped(tripped)

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
        # "stop": the trip is recorded and `stopped` becomes True. The current step
        # is allowed to finish so the caller can clean up, and every entry point
        # afterwards raises GuardStopped -- see `raise_if_tripped`.

    # -- reporting -----------------------------------------------------------

    def report(self) -> Report:
        """Snapshot this run as a :class:`~agentguard.Report`."""
        by_model = tuple(self._tracker.by_model().values())
        by_tag = tuple(self._tracker.by_tag().values())
        by_tool = tuple(self._tracker.by_tool().values())
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
            by_tag=by_tag,
            by_tool=by_tool,
            unpriced_models=self._tracker.unpriced_models,
            unpriced_calls=self._tracker.unpriced_calls,
            trip=trip_verdict,
            tripped_reason=tripped_reason,
            pricing_sources=tuple(str(source) for source in self._pricing_config.sources),
            checkpointed_calls=self._checkpointed_calls,
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


def _parse_snapshot(snapshot: Mapping[str, Any] | str) -> Mapping[str, Any]:
    """Accept a snapshot as the dict :meth:`Guard.snapshot` returned, or as JSON."""
    if isinstance(snapshot, str):
        try:
            decoded = json.loads(snapshot)
        except ValueError as exc:
            raise GuardConfigError(f"snapshot is not valid JSON: {exc}") from exc
    else:
        decoded = snapshot
    if not isinstance(decoded, Mapping):
        raise GuardConfigError(
            f"snapshot must be a mapping or JSON object, got {type(decoded).__name__}"
        )
    return decoded


def _read_detector_states(value: Any) -> list[Mapping[str, Any] | None]:
    """One optional state per detector, as :meth:`LoopMonitor.get_states` emits."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise GuardConfigError(
            f"snapshot detector states must be a list, got {type(value).__name__}"
        )
    states: list[Mapping[str, Any] | None] = []
    for position, state in enumerate(value):
        if state is None:
            states.append(None)
        elif isinstance(state, Mapping):
            states.append(state)
        else:
            raise GuardConfigError(
                f"snapshot detector state {position} must be a mapping or null, "
                f"got {type(state).__name__}"
            )
    return states


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
