"""Record every LLM call without changing a single call site.

Run it::

    python examples/wrapped_client.py

The client below is a fake, but it is shaped exactly like the OpenAI SDK:
``client.chat.completions.create(...)``. agent-guard never imports a provider SDK,
so the same wrapper covers Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq
and Azure OpenAI.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

from agentguard import BudgetExceeded, Guard
from agentguard.adapters.openai import guard_openai


class _Completions:
    def create(self, **kwargs: Any) -> Any:
        return NS(
            model=kwargs.get("model", "gpt-4o"),
            usage=NS(prompt_tokens=1_200, completion_tokens=300),
        )


class FakeOpenAI:
    """Duck-typed stand-in for ``openai.OpenAI``."""

    def __init__(self) -> None:
        self.chat = NS(completions=_Completions())


def main() -> None:
    # 1,200 + 300 tokens at gpt-4o prices is $0.006 per call, so a two-cent cap
    # allows three calls and refuses the fourth.
    client = guard_openai(FakeOpenAI(), max_usd=0.02, max_steps=10, name="qa-bot")

    print("Calling `client.chat.completions.create(...)` with no accounting code:\n")
    try:
        for question in (
            "What is a bloom filter?",
            "When would one beat a hash set?",
            "How do I size the bit array?",
            "What about deletion?",  # <- this one exceeds the budget
        ):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
            )
            print(f"  asked: {question:<32} spent ${client.guard.spent_usd:.4f}")
    except BudgetExceeded as exc:
        print(f"\n  STOPPED after the call was recorded: {exc}")
        print("  A post-hoc check can only report overspend. See pre-flight below.")

    print()
    print(
        "Now the same wrapper with pre-flight estimation enabled. A single huge\n"
        "call is refused *before* it is issued, which a post-hoc check cannot do.\n"
    )

    careful = guard_openai(FakeOpenAI(), Guard(max_usd=0.05), preflight=True)
    try:
        careful.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "summarise this 200 page report"}],
            max_tokens=60_000,
        )
    except BudgetExceeded as exc:
        print(f"  refused: {exc}")

    print()
    print(careful.guard.report())


if __name__ == "__main__":
    main()
