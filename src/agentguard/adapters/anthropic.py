"""Anthropic client adapter."""

from __future__ import annotations

from typing import Any

from ..guard import Guard
from . import GuardedClient, resolve_guard

__all__ = ["guard_anthropic"]


#: ``messages.create`` is the workhorse. ``stream`` is deliberately excluded: a
#: stream carries no usage until it is drained, and agent-guard will not guess.
_RECORD_ON = ("create",)


def guard_anthropic(
    client: Any,
    guard: Guard | None = None,
    *,
    preflight: bool = False,
    chars_per_token: float = 3.0,
    **guard_kwargs: Any,
) -> GuardedClient:
    """Wrap an Anthropic client so every message is accounted automatically.

    Anthropic reports ``input_tokens``/``output_tokens`` and, for prompt caching,
    ``cache_read_input_tokens`` — all of which agent-guard extracts and bills at
    the cached rate when one is known::

        from anthropic import Anthropic
        from agentguard.adapters.anthropic import guard_anthropic

        client = guard_anthropic(Anthropic(), max_usd=2.0, max_steps=40)
        message = client.messages.create(...)   # recorded, no extra code

    For streaming, collect usage from the final ``message_delta`` event and call
    ``guard.record(...)`` yourself once the stream is drained.
    """
    resolved = resolve_guard(guard, dict(guard_kwargs), adapter="guard_anthropic")
    return GuardedClient(
        client,
        resolved,
        record_on=_RECORD_ON,
        preflight=preflight,
        chars_per_token=chars_per_token,
    )
