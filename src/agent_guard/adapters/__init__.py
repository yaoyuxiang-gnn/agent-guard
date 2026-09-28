"""Optional client adapters.

agent-guard never imports a provider SDK. These adapters rely only on the *shape*
of a client — attribute access down to a ``create()`` method — so they work with
OpenAI, Azure OpenAI, Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq and
anything else following the same convention.

.. code-block:: python

   from openai import OpenAI
   from agent_guard import Guard
   from agent_guard.adapters.openai import guard_openai

   guard = Guard(max_usd=1.0, max_steps=25)
   client = guard_openai(OpenAI(), guard)

   response = client.chat.completions.create(...)   # recorded automatically

Streaming responses do not carry usage until the stream is drained, and
agent-guard will not guess. For streaming, collect the final usage chunk yourself
and call ``guard.record(response)`` once the stream completes.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Sequence

from .._util import stable_json
from ..exceptions import GuardConfigError
from ..guard import Guard

__all__ = ["GuardedClient", "guard_client"]


#: How many attribute levels below the root client are still treated as resources
#: (``client.chat.completions`` is two, ``client.beta.chat.completions`` is three).
_DEFAULT_DEPTH = 3


def _is_resource(obj: Any) -> bool:
    """Heuristic: a namespace object worth proxying, not a value or a callable."""
    if obj is None or isinstance(obj, type) or callable(obj):
        return False
    return hasattr(obj, "__dict__")


class GuardedClient:
    """A transparent proxy that records every LLM call onto a guard.

    Attribute access is forwarded to the wrapped client. Any callable named in
    ``record_on`` (``"create"`` by default) has its response extracted and
    accounted before it is returned to the caller, so the call site needs no
    changes at all.

    >>> from types import SimpleNamespace as NS
    >>> fake = NS(chat=NS(completions=NS(create=lambda **kw: {
    ...     "model": "gpt-4o",
    ...     "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
    ... })))
    >>> guard = Guard(max_usd=5.0)
    >>> client = GuardedClient(fake, guard)
    >>> _ = client.chat.completions.create(model="gpt-4o")
    >>> round(guard.spent_usd, 2)
    2.5
    """

    __slots__ = ("_target", "_guard", "_record_on", "_depth", "_preflight", "_chars_per_token")

    def __init__(
        self,
        target: Any,
        guard: Guard,
        *,
        record_on: Iterable[str] = ("create",),
        depth: int = _DEFAULT_DEPTH,
        preflight: bool = False,
        chars_per_token: float = 3.0,
    ) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_guard", guard)
        object.__setattr__(self, "_record_on", tuple(record_on))
        object.__setattr__(self, "_depth", max(0, int(depth)))
        object.__setattr__(self, "_preflight", bool(preflight))
        object.__setattr__(self, "_chars_per_token", float(chars_per_token))

    @property
    def guard(self) -> Guard:
        """The guard this client reports to."""
        return object.__getattribute__(self, "_guard")

    @property
    def target(self) -> Any:
        """The wrapped client, for anything the proxy gets in the way of."""
        return object.__getattribute__(self, "_target")

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        attr = getattr(target, name)  # AttributeError propagates untouched

        record_on: tuple[str, ...] = object.__getattribute__(self, "_record_on")
        if callable(attr) and name in record_on:
            return self._wrap(attr)

        depth: int = object.__getattribute__(self, "_depth")
        if depth > 0 and _is_resource(attr):
            return GuardedClient(
                attr,
                object.__getattribute__(self, "_guard"),
                record_on=record_on,
                depth=depth - 1,
                preflight=object.__getattribute__(self, "_preflight"),
                chars_per_token=object.__getattribute__(self, "_chars_per_token"),
            )
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes must land on the real client, or `client.timeout = 5` would
        # silently do nothing on the proxy.
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(object.__getattribute__(self, "_target"))))

    def __repr__(self) -> str:
        target = object.__getattribute__(self, "_target")
        return f"GuardedClient({type(target).__name__})"

    def __enter__(self) -> "GuardedClient":
        enter = getattr(object.__getattribute__(self, "_target"), "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        exit_ = getattr(object.__getattribute__(self, "_target"), "__exit__", None)
        return exit_(*exc) if exit_ is not None else False

    # -- internals -----------------------------------------------------------

    def _wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        guard = object.__getattribute__(self, "_guard")
        preflight = object.__getattribute__(self, "_preflight")
        chars_per_token = object.__getattribute__(self, "_chars_per_token")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if preflight:
                _preflight_estimate(guard, kwargs, chars_per_token)
            response = func(*args, **kwargs)
            guard.record(response)
            return response

        return wrapper


def _preflight_estimate(guard: Guard, kwargs: dict[str, Any], chars_per_token: float) -> None:
    """Best-effort budget check *before* an adapted call is issued.

    Input tokens are estimated from the serialised prompt, because counting them
    properly needs a provider tokenizer that agent-guard deliberately does not
    depend on. The estimate is a safety net for catastrophic calls (a 200k-token
    context with a 16k output cap against $0.02 of remaining budget), not an
    accounting figure. Output is bounded by ``max_tokens``/``max_completion_tokens``.
    """
    model = kwargs.get("model")
    payload = kwargs.get("messages", kwargs.get("input"))
    if not isinstance(model, str) or payload is None:
        return

    characters = len(stable_json(payload))
    input_tokens = int(characters / max(1.0, chars_per_token))
    raw_max = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 1024
    try:
        max_output_tokens = int(raw_max)
    except (TypeError, ValueError):
        max_output_tokens = 1024

    guard.preflight(model, input_tokens=input_tokens, max_output_tokens=max_output_tokens)


def guard_client(
    target: Any,
    guard: Guard,
    *,
    record_on: Sequence[str] = ("create",),
    depth: int = _DEFAULT_DEPTH,
    preflight: bool = False,
    chars_per_token: float = 3.0,
) -> GuardedClient:
    """Wrap any LLM client so calls are recorded onto ``guard``.

    Use :func:`agent_guard.adapters.openai.guard_openai` or
    :func:`agent_guard.adapters.anthropic.guard_anthropic` for the common cases;
    this is the generic escape hatch for LiteLLM, OpenRouter, vLLM and friends.
    """
    if guard is None:
        raise GuardConfigError("guard_client() requires a Guard instance")
    return GuardedClient(
        target,
        guard,
        record_on=record_on,
        depth=depth,
        preflight=preflight,
        chars_per_token=chars_per_token,
    )


def resolve_guard(
    guard: Guard | None,
    guard_kwargs: dict[str, Any],
    *,
    adapter: str,
) -> Guard:
    """Shared ``guard`` XOR ``guard_kwargs`` handling for the named adapters."""
    if guard is not None and guard_kwargs:
        raise GuardConfigError(
            f"{adapter}() takes either guard=<Guard> or Guard keyword arguments, not both"
        )
    if guard is not None:
        return guard
    if not guard_kwargs:
        raise GuardConfigError(
            f"{adapter}() needs a guard: pass guard=Guard(...) or Guard keyword "
            f"arguments such as max_usd=1.0"
        )
    return Guard(**guard_kwargs)
