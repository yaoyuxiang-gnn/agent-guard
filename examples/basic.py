"""Cap an agent's spend with three lines of setup.

Run it::

    python examples/basic.py

Runs fully offline. The model below is simulated, but its response has the same
shape a real SDK returns, so ``step.record(response)`` keeps working unchanged
when you swap in OpenAI or Anthropic.
"""

from __future__ import annotations

from typing import Any

from agentguard import BudgetExceeded, Guard

#: 4,200 prompt + 850 completion tokens at gpt-4o list prices is about $0.019,
#: so a five-cent cap should buy two calls and stop the third.
CALL_TOKENS = {"prompt_tokens": 4_200, "completion_tokens": 850}


def fake_model(prompt: str) -> dict[str, Any]:
    """Stand-in for a provider call: same response shape, no network."""
    return {"model": "gpt-4o", "usage": dict(CALL_TOKENS)}


def main() -> None:
    guard = Guard(max_usd=0.05, max_steps=20, name="research-agent")

    print("Running an agent with a $0.05 budget.\n")
    try:
        with guard:
            for task in ("find papers", "summarise them", "draft outline", "polish prose"):
                with guard.step(tag=task) as step:
                    response = fake_model(task)
                    step.record(response)
                    print(
                        f"  step {step.index}  {task:<14}"
                        f" spent ${guard.spent_usd:.4f}"
                        f"  left ${guard.remaining_usd:.4f}"
                    )
    except BudgetExceeded as exc:
        print(f"\n  STOPPED: {exc}")

    print()
    print(guard.report())


if __name__ == "__main__":
    main()
