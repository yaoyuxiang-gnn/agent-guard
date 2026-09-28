"""agent-guard — budget caps, runaway-loop detection and circuit breakers for AI agents.

Stop an agent that is burning money, looping forever, or running past its
deadline. Zero runtime dependencies, no vendor SDK, framework agnostic.

Quick start::

    from agentguard import Guard, BudgetExceeded

    guard = Guard(max_usd=1.00, max_steps=25, name="research-agent")

    try:
        with guard:
            while True:
                with guard.step() as step:
                    response = call_llm(...)
                    step.record(response)
                    with step.tool("search", {"q": query}):
                        results = search(query)
    except BudgetExceeded as exc:
        print(exc)
    finally:
        print(guard.report())

The public surface is deliberately small: one class, one decorator, one exception
base, and a handful of value types.
"""

from __future__ import annotations

from ._version import __version__
from .decorators import guarded
from .exceptions import (
    BudgetExceeded,
    GuardConfigError,
    GuardError,
    GuardTripped,
    LoopDetected,
    StepLimitExceeded,
    TimeLimitExceeded,
    TokenLimitExceeded,
)
from .guard import Guard, Step, current_guard
from .loop import (
    CycleDetector,
    Detector,
    LoopMonitor,
    LoopVerdict,
    NoProgressDetector,
    RepeatDetector,
    SimilarityDetector,
    call_signature,
)
from .pricing import DEFAULT_PRICING, PRICING_AS_OF, Price, PriceTable
from .report import LimitStatus, Report
from .tracker import CallRecord, CostTracker, ModelSummary, Usage

__all__ = [
    # core
    "Guard",
    "Step",
    "guarded",
    "current_guard",
    # errors
    "GuardError",
    "GuardTripped",
    "GuardConfigError",
    "BudgetExceeded",
    "TokenLimitExceeded",
    "StepLimitExceeded",
    "TimeLimitExceeded",
    "LoopDetected",
    # loop detection
    "Detector",
    "RepeatDetector",
    "CycleDetector",
    "SimilarityDetector",
    "NoProgressDetector",
    "LoopMonitor",
    "LoopVerdict",
    "call_signature",
    # accounting
    "Usage",
    "CallRecord",
    "CostTracker",
    "ModelSummary",
    "Price",
    "PriceTable",
    "DEFAULT_PRICING",
    "PRICING_AS_OF",
    # reporting
    "Report",
    "LimitStatus",
    # metadata
    "__version__",
]
