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
    "BudgetExceeded",
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


class BudgetExceeded(GuardTripped):
    """The run would exceed its dollar budget.

    Raised in two situations, distinguished by :attr:`projected_usd`:

    * **After the fact** — a recorded call pushed the total over the limit.
    * **Pre-flight** — :meth:`Guard.preflight` refused a call *before* it was
      made, because its worst-case cost would not fit in the remaining budget.
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
    ) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.projected_usd = projected_usd
        self.model = model
        self.call_cost_usd = call_cost_usd

        target = f"a {model} call" if model else "the next call"
        if projected_usd is not None:
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
