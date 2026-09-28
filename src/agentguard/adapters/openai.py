"""OpenAI-compatible client adapter."""

from __future__ import annotations

from typing import Any

from ..guard import Guard
from . import GuardedClient, resolve_guard

__all__ = ["guard_openai"]


#: Methods that return a response carrying ``.usage``. ``create`` covers chat
#: completions, the Responses API and embeddings; ``parse`` covers structured
#: outputs, which wrap the same usage object.
_RECORD_ON = ("create", "parse")


def guard_openai(
    client: Any,
    guard: Guard | None = None,
    *,
    preflight: bool = False,
    chars_per_token: float = 3.0,
    **guard_kwargs: Any,
) -> GuardedClient:
    """Wrap an OpenAI-compatible client so every call is accounted automatically.

    Works with the official ``openai`` package, Azure OpenAI, LiteLLM, OpenRouter,
    vLLM, Together, Groq and anything else exposing
    ``client.chat.completions.create``.

    Pass an existing ``guard``, or Guard keyword arguments to create one::

        from openai import OpenAI
        from agentguard.adapters.openai import guard_openai

        client = guard_openai(OpenAI(), max_usd=1.0, max_steps=25)
        response = client.chat.completions.create(...)   # recorded, no extra code

    :param preflight: Estimate each call's worst-case cost before issuing it, and
        refuse calls that cannot fit in the remaining budget. Input tokens are
        estimated from the serialised prompt, so treat this as a safety net for
        catastrophic calls rather than an accounting figure.
    :param chars_per_token: Divisor for the pre-flight input estimate. Lower is
        more conservative; 3.0 over-estimates slightly for English prose and still
        under-estimates dense CJK text.
    """
    resolved = resolve_guard(guard, dict(guard_kwargs), adapter="guard_openai")
    return GuardedClient(
        client,
        resolved,
        record_on=_RECORD_ON,
        preflight=preflight,
        chars_per_token=chars_per_token,
    )
