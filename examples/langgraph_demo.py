"""Guard a LangGraph agent with one callback handler.

Run it::

    python examples/langgraph_demo.py

With a real graph the wiring is one line — pass the handler in the config::

    from agentguard import Guard
    from agentguard.integrations.langgraph import guard_langgraph

    guard = Guard(max_usd=1.0, max_steps=25)
    handler = guard_langgraph(guard)
    graph.invoke(inputs, config={"callbacks": [handler]})

This demo runs offline: it drives the handler with payloads shaped exactly like
LangChain's ``LLMResult`` (``llm_output`` plus ``generations`` of messages with
``usage_metadata``) and like ``on_tool_start`` — so you can watch cost
accounting and tool-loop detection fire without an API key.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from agentguard import Guard, LoopDetected
from agentguard.integrations.langgraph import guard_langgraph


def fake_llm_result(model: str, prompt_tokens: int, completion_tokens: int) -> NS:
    """Shaped like langchain-core's LLMResult for an OpenAI-backed chat model."""
    return NS(
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
        generations=[],
    )


def main() -> None:
    guard = Guard(max_usd=0.01, name="research-agent")
    handler = guard_langgraph(guard)

    print("Two LLM calls recorded from LangChain callbacks:\n")
    handler.on_llm_end(fake_llm_result("gpt-4o", 1_000, 50))
    handler.on_llm_end(fake_llm_result("gpt-4o", 1_200, 80))
    print(f"  calls={guard.calls}  spent ${guard.spent_usd:.4f} / $0.01")

    print("\nA LangGraph agent stuck calling the same tool with the same input:\n")
    try:
        for _ in range(3):
            handler.on_tool_start({"name": "web_search"}, '{"q": "agent guard pricing"}')
    except LoopDetected as exc:
        print(f"  TRIPPED: {exc}")
        print("  The exception propagates (raise_error=True), so the run stops.")

    print()
    print(guard.report())


if __name__ == "__main__":
    main()
