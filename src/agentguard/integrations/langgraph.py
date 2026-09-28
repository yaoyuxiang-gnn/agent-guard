"""LangGraph / LangChain integration: a callback handler bound to a guard.

.. code-block:: python

    from agentguard import Guard
    from agentguard.integrations.langgraph import guard_langgraph

    guard = Guard(max_usd=1.0, max_steps=25)
    handler = guard_langgraph(guard)

    graph.invoke(inputs, config={"callbacks": [handler]})

The handler subclasses ``langchain_core.callbacks.BaseCallbackHandler`` when
langchain-core is installed — LangChain's callback manager dispatches on
isinstance, so pure duck-typing is not enough — and falls back to a plain
class otherwise, which keeps the test suite and duck-typed callers working
with no dependency at all. langchain-core is imported lazily here, never at
package import time.

Two hooks are wired up:

* ``on_llm_end`` records token usage and cost, reading
  ``llm_output["token_usage"]`` (OpenAI-style), ``llm_output["usage"]``
  (Anthropic-style) or per-generation ``message.usage_metadata``.
* ``on_tool_start`` fingerprints the tool call for loop detection, so a
  LangGraph agent stuck bouncing between tools trips the same detectors a
  hand-written loop does.

``raise_error`` is set on the handler so a trip propagates out of the callback
manager and stops the run, instead of being logged and swallowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..adapters import resolve_guard
from ..guard import Guard
from ..loop import call_signature
from ..tracker import Usage, extract_model, extract_usage

__all__ = ["guard_langgraph"]


def guard_langgraph(guard: Guard | None = None, **guard_kwargs: Any) -> Any:
    """Return a LangChain callback handler that records onto ``guard``.

    Pass an existing ``guard``, or Guard keyword arguments to create one::

        handler = guard_langgraph(max_usd=1.0, max_steps=25)
        graph.invoke(inputs, config={"callbacks": [handler]})
        print(handler)   # the handler; the guard is `handler.guard`

    The returned object exposes the guard as ``.guard`` so the run report is
    reachable after the graph finishes.
    """
    resolved = resolve_guard(guard, dict(guard_kwargs), adapter="guard_langgraph")

    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        BaseCallbackHandler = object

    class AgentGuardHandler(BaseCallbackHandler):  # type: ignore[misc]
        """Records LLM calls and fingerprints tool calls onto one guard."""

        #: A guard trip must kill the run, not be logged and swallowed by the
        #: callback manager. LangChain honours this flag and re-raises.
        raise_error = True

        @property
        def guard(self) -> Guard:
            return resolved

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            _record_llm_result(resolved, response)

        def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
            tool_name = kwargs.get("name")
            if not tool_name and isinstance(serialized, Mapping):
                tool_name = serialized.get("name")
            resolved.observe(call_signature(str(tool_name or "tool"), input_str))

    return AgentGuardHandler()


def _record_llm_result(guard: Guard, response: Any) -> None:
    """Extract usage from a LangChain ``LLMResult`` and record one call.

    LangChain normalises nothing here: OpenAI integrations report
    ``llm_output["token_usage"]`` with ``prompt_tokens``/``completion_tokens``,
    Anthropic integrations report ``llm_output["usage"]`` with
    ``input_tokens``/``output_tokens``, and chat models additionally stamp each
    generation's message with ``usage_metadata``. All three shapes are read;
    per-generation counts are summed with any top-level figure.
    """
    model: str | None = None
    usage: Usage | None = None

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        raw_model = llm_output.get("model_name")
        if isinstance(raw_model, str) and raw_model:
            model = raw_model
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, Mapping):
            usage = extract_usage({"usage": token_usage})

    for generation in getattr(response, "generations", None) or ():
        for chunk in generation:
            message = getattr(chunk, "message", None)
            if message is None:
                continue
            found = extract_usage(message)
            if found is not None:
                usage = found if usage is None else usage + found
            if model is None:
                model = extract_model(getattr(message, "response_metadata", None))

    if usage is None:
        # Hand the raw response to record() so its "no usable usage" warning
        # fires exactly as it does for direct SDK responses.
        guard.record(response, model=model)
        return
    guard.record(
        model=model or "unknown",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
    )
