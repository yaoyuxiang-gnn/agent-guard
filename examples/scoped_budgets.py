"""Cap one tool, not just the whole run — and keep the attribution in worker threads.

Run it::

    python examples/scoped_budgets.py

Runs fully offline. The model is simulated, but its responses have the shape a real
SDK returns.

A run-level ``max_usd`` can only tell you the run got expensive. It cannot tell you
*which part* of it did, so a single tool that has gone into a retry storm can spend
the entire allowance before the run-level cap notices — and by then the answer to
"what ate the budget?" is a report you read afterwards.

``scoped_budgets`` puts a smaller cap on one tool or one tag. This example fans work
out across a thread pool, which is the case where attribution is easiest to lose:
worker threads start with an empty ``contextvars`` context, so without
``guard.bind(...)`` the calls are still *counted* but belong to no tool, and the
per-tool cap silently never fires.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentguard import BudgetScopeExceeded, Guard

#: $0.0025 per call at gpt-4o list prices: 1,000 prompt tokens.
CALL_TOKENS = {"prompt_tokens": 1_000, "completion_tokens": 0}

#: Fetching is allowed $0.012 — under five calls — while the run may spend $1.
FETCH_BUDGET_USD = 0.012


def fake_model(url: str) -> dict[str, Any]:
    """Stand-in for a provider call: same response shape, no network."""
    return {"model": "gpt-4o", "usage": dict(CALL_TOKENS)}


def main() -> None:
    guard = Guard(
        max_usd=1.00,
        max_steps=50,
        scoped_budgets={"tool:fetch": FETCH_BUDGET_USD},
        name="scoped-agent",
    )

    print(f"A $1.00 run, with only ${FETCH_BUDGET_USD:.3f} of it allowed for `fetch`.\n")
    urls = [f"https://example.test/{page}" for page in range(12)]

    try:
        with guard, guard.step(tag="crawl") as step, guard.tool("fetch"):
            # `bind` captures this moment's step and tool, so a call made on
            # a pool worker is attributed to `step` and to `fetch` rather
            # than to nothing.
            bound = guard.bind(fake_model)
            with ThreadPoolExecutor(4) as pool:
                for index, response in enumerate(pool.map(bound, urls), start=1):
                    step.record(response)
                    print(
                        f"  fetch {index:>2}  {urls[index - 1]:<28}"
                        f" total ${guard.spent_usd:.4f}"
                        f"  fetch ${guard.scope_spend('tool', 'fetch'):.4f}"
                    )
    except BudgetScopeExceeded as exc:
        print(f"\n  STOPPED: {exc}")
        print("  The run had $0.9875 left, so `max_usd` would not have fired here.")

    print()
    print(guard.report())


if __name__ == "__main__":
    main()
