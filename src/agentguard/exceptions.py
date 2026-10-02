"""Exception hierarchy for agent-guard.

Every error raised by agent-guard derives from :class:`GuardError`, so wrapping a
whole agent run in ``except GuardError`` is always safe. Errors that *stopped* a
run additionally derive from :class:`GuardTripped`, which lets you handle
"the agent was cut off" uniformly no matter which limit fired::

    from agentguard import Guard, GuardTripped

    guard = Guard(max_usd=0.01)
    try:
        with guard:
            guard.record("gpt-4o", input_tokens=50_000, output_tokens=50_000)
    except GuardTripped as exc:
        print(exc.reason)      # 'budget'
        print(exc.spent_usd)   # 0.625
        print(guard.report())
"""

from __future__ import annotations

from typing import Any

from ._util import format_tokens, format_usd

__all__ = [
    "GuardError",
    "GuardConfigError",
    "GuardTripped",
    "GuardStopped",
    "BudgetExceeded",
    "BudgetScopeExceeded",
    "TokenLimitExceeded",
    "StepLimitExceeded",
    "TimeLimitExceeded",
    "LoopDetected",
]


class GuardError(Exception):
    """Base class for every error raised by agent-guard."""


class GuardConfigError(GuardError, ValueError):
    """agent-guard was configured in a way that cannot work.

    Raised eagerly, while the :class:`~agentguard.Guard` is being constructed,
    rather than in the middle of an agent run. A misconfigured guard should fail
    loudly at import time, never as a mysterious mid-run exception.
    """


class GuardTripped(GuardError):
    """Base class for a guard that *stopped* the run.

    Catch this when you want one handler for "the agent was cut off", regardless
    of which limit fired::

        try:
            run_agent()
        except GuardTripped as exc:
            log.warning("agent stopped (%s): %s", exc.reason, exc)

    Subclasses expose structured attributes (``spent_usd``, ``limit_usd``, ...)
    plus a :attr:`context` mapping, so callers never have to parse the message.
    """

    #: Short, stable, machine-readable identifier for the trip. One of
    #: ``"budget"``, ``"tokens"``, ``"steps"``, ``"time"`` or ``"loop"``.
    reason: str = "tripped"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        #: Structured details about the trip, safe to log or serialise.
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


class GuardStopped(GuardTripped):
    """A tripped guard refused the next piece of work, under ``on_trip="stop"``.

    ``"raise"`` raises the trip itself the moment a limit fires. ``"stop"`` records
    the trip, calls ``on_trip_callback`` and lets the current step finish, so a loop
    can clean up — but the run is over, and every entry point afterwards
    (:meth:`~agentguard.Guard.step`, ``record``, ``tool``, ``check``, ``preflight``,
    a wrapped client call) raises this. That is what makes ``"stop"`` safe: a loop
    that forgets to check ``guard.stopped`` stops anyway, instead of spending the
    rest of the budget.

    :attr:`cause` is the trip that ended the run, so a handler can still tell a
    budget overrun from a loop::

        try:
            run_agent()
        except GuardStopped as exc:
            log.warning("stopped by %s: %s", exc.cause.reason, exc.cause)
        finally:
            print(guard.report())
    """

    reason = "stopped"

    def __init__(self, cause: GuardTripped) -> None:
        self.cause = cause
        # Mirror the cause's reason, so ``except GuardTripped`` handlers that switch
        # on ``reason`` keep working when a guard stops instead of raising.
        self.reason = cause.reason
        super().__init__(
            f"Guard stopped: {cause}",
            reason=cause.reason,
            cause_message=cause.message,
        )


class BudgetExceeded(GuardTripped):
    """The run would exceed its dollar budget.

    Raised in three situations, distinguished by :attr:`projected_usd` and
    :attr:`unpriced_model`:

    * **After the fact** — a recorded call pushed the total over the limit.
    * **Pre-flight** — :meth:`Guard.preflight` refused a call *before* it was
      made, because its worst-case cost would not fit in the remaining budget.
    * **Pre-flight, unpriced** — the same, because the model has no known price at
      all and so cannot be shown to fit. Only under ``strict=True``; see
      :meth:`Guard.preflight`.
    """

    reason = "budget"

    def __init__(
        self,
        spent_usd: float,
        limit_usd: float,
        *,
        projected_usd: float | None = None,
        model: str | None = None,
        call_cost_usd: float | None = None,
        unpriced_model: bool = False,
    ) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.projected_usd = projected_usd
        self.model = model
        self.call_cost_usd = call_cost_usd
        self.unpriced_model = unpriced_model

        target = f"a {model} call" if model else "the next call"
        if unpriced_model:
            detail = (
                f"refused before spending: {target} has no known price, so it cannot "
                f"be shown to fit the {format_usd(limit_usd)} limit (already spent "
                f"{format_usd(spent_usd)})"
            )
        elif projected_usd is not None:
            detail = (
                f"refused before spending: {target} could reach "
                f"{format_usd(projected_usd)}, over the {format_usd(limit_usd)} limit "
                f"(already spent {format_usd(spent_usd)})"
            )
        else:
            detail = f"spent {format_usd(spent_usd)} of a {format_usd(limit_usd)} limit"
        super().__init__(
            f"Budget exceeded: {detail}.",
            spent_usd=spent_usd,
            limit_usd=limit_usd,
            projected_usd=projected_usd,
            model=model,
            call_cost_usd=call_cost_usd,
            unpriced_model=unpriced_model,
        )


class BudgetScopeExceeded(GuardTripped):
    """One tool or one tag exceeded the budget set aside for it.

    A whole-run cap answers "is this run too expensive?". It cannot answer "which
    part of it is too expensive?", so a single runaway tool can spend the entire
    allowance before the run-level limit notices. This is that limit, per scope::

        Guard(max_usd=5.0, scoped_budgets={"tool:search": 1.0})

    :attr:`scope` is ``"tool"`` or ``"tag"`` and :attr:`name` is the tool or tag
    that blew it, so the message names the culprit rather than the run.
    """

    reason = "scope-budget"

    def __init__(
        self,
        scope: str,
        name: str,
        spent_usd: float,
        limit_usd: float,
    ) -> None:
        self.scope = scope
        self.name = name
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Budget exceeded for {scope} {name!r}: spent {format_usd(spent_usd)} "
            f"of its {format_usd(limit_usd)} allowance.",
            scope=scope,
            name=name,
            spent_usd=spent_usd,
            limit_usd=limit_usd,
        )


class TokenLimitExceeded(GuardTripped):
    """The run exceeded its total-token allowance."""

    reason = "tokens"

    def __init__(self, used_tokens: int, limit_tokens: int) -> None:
        self.used_tokens = used_tokens
        self.limit_tokens = limit_tokens
        super().__init__(
            f"Token limit exceeded: used {format_tokens(used_tokens)} of a "
            f"{format_tokens(limit_tokens)} token allowance.",
            used_tokens=used_tokens,
            limit_tokens=limit_tokens,
        )


class StepLimitExceeded(GuardTripped):
    """The run took more steps than allowed."""

    reason = "steps"

    def __init__(self, steps: int, limit_steps: int) -> None:
        self.steps = steps
        self.limit_steps = limit_steps
        super().__init__(
            f"Step limit exceeded: reached step {steps} of a {limit_steps} step limit.",
            steps=steps,
            limit_steps=limit_steps,
        )


class TimeLimitExceeded(GuardTripped):
    """The run exceeded its wall-clock allowance."""

    reason = "time"

    def __init__(self, elapsed_s: float, limit_s: float) -> None:
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        super().__init__(
            f"Time limit exceeded: ran for {elapsed_s:.3f}s of a {limit_s:.3f}s limit.",
            elapsed_s=elapsed_s,
            limit_s=limit_s,
        )


class LoopDetected(GuardTripped):
    """The agent repeated itself in a way that looks like a runaway loop.

    Carries the detector's verdict so callers can log *why* it concluded a loop,
    which is usually the single most useful line in a post-mortem.
    """

    reason = "loop"

    def __init__(
        self,
        *,
        kind: str,
        detail: str,
        signature: str | None = None,
        count: int = 0,
        step: int | None = None,
    ) -> None:
        self.kind = kind
        self.detail = detail
        self.signature = signature
        self.count = count
        self.step = step
        super().__init__(
            f"Loop detected [{kind}]: {detail}.",
            kind=kind,
            detail=detail,
            signature=signature,
            count=count,
            step=step,
        )
