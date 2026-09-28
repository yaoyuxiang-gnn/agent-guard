"""Record streaming responses — usage is counted once, when the stream ends.

Run it::

    python examples/streaming.py

The clients below are fakes, but their streams are shaped exactly like the real
SDKs: OpenAI chunks carry ``usage`` on the final chunk (when the request sets
``stream_options={"include_usage": True}``), Anthropic events carry it on
``message_start`` and ``message_delta``. agent-guard wraps either shape in a
``GuardedStream`` and records exactly one call when the stream is drained.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

from agentguard.adapters.openai import guard_openai


class _Completions:
    def create(self, **kwargs: Any) -> Any:
        chunks = [
            NS(model="gpt-4o", choices=[NS(delta=NS(content="Hello"))], usage=None),
            NS(model="gpt-4o", choices=[NS(delta=NS(content=" world"))], usage=None),
            NS(
                model="gpt-4o",
                choices=[],
                usage=NS(prompt_tokens=1_200, completion_tokens=300),
            ),
        ]
        return iter(chunks)


class FakeOpenAI:
    """Duck-typed stand-in for ``openai.OpenAI`` streaming."""

    def __init__(self) -> None:
        self.chat = NS(completions=_Completions())


def main() -> None:
    client = guard_openai(FakeOpenAI(), max_usd=0.02, name="stream-bot")

    print("Streaming with stream=True — the loop sees plain chunks:\n")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            print(f"  chunk: {delta!r}")
    # The stream is drained here: usage was recorded exactly once.
    print(f"\n  recorded: {client.guard.calls} call, spent ${client.guard.spent_usd:.4f}")

    print()
    print(client.guard.report())


if __name__ == "__main__":
    main()
