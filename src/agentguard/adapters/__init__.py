"""Optional client adapters.

agent-guard never imports a provider SDK. These adapters rely only on the *shape*
of a client — attribute access down to a ``create()`` method — so they work with
OpenAI, Azure OpenAI, Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq and
anything else following the same convention.

.. code-block:: python

   from openai import OpenAI
   from agentguard import Guard
   from agentguard.adapters.openai import guard_openai

   guard = Guard(max_usd=1.0, max_steps=25)
   client = guard_openai(OpenAI(), guard)

   response = client.chat.completions.create(...)   # recorded automatically

Streaming responses carry no usage until the stream is drained, so a response
that looks like a stream is wrapped in :class:`GuardedStream` instead of being
recorded immediately. Iterating it to completion records exactly one call; an
abandoned stream records whatever usage it saw. A stream that reports nothing
warns rather than silently counting ``$0``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import warnings
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from .._util import stable_json
from ..exceptions import GuardConfigError
from ..guard import Guard
from ..tracker import extract_model, extract_usage

__all__ = ["GuardedClient", "GuardedStream", "guard_client"]


#: How many attribute levels below the root client are still treated as resources
#: (``client.chat.completions`` is two, ``client.beta.chat.completions`` is three).
_DEFAULT_DEPTH = 3


def _is_resource(obj: Any) -> bool:
    """Heuristic: a namespace object worth proxying, not a value or a callable."""
    if obj is None or isinstance(obj, type) or callable(obj):
        return False
    return hasattr(obj, "__dict__")


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or an object, without raising."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


class GuardedStream:
    """Wrap a streaming response so usage is recorded once, when the stream ends.

    Duck-typed across provider event shapes:

    * **OpenAI-style** chunks carry ``usage`` on the final chunk when the request
      sets ``stream_options={"include_usage": True}``.
    * **Anthropic-style** raw events carry ``usage`` on ``message_start`` (input
      tokens) and ``message_delta`` (cumulative output tokens).
    * **Anthropic's ``messages.stream()``** context manager is entered and exited
      transparently, and ``get_final_message()`` is honoured when iteration did
      not surface usage.

    Iterating to completion records exactly one call onto the guard. Abandoning
    the stream — ``close()``, or breaking out of the loop — records whatever
    usage was seen. A stream that reported nothing warns instead of silently
    counting ``$0``::

        stream = GuardedStream(client.chat.completions.create(..., stream=True), guard)
        for chunk in stream:
            print(chunk)
        # recorded on the guard here, exactly once

    Works on async streams too (``async for``, ``async with``).
    """

    __slots__ = (
        "_cached",
        "_entered",
        "_guard",
        "_input",
        "_model",
        "_model_hint",
        "_output",
        "_recorded",
        "_stream",
    )

    def __init__(self, stream: Any, guard: Guard, *, model_hint: str | None = None) -> None:
        self._stream = stream
        self._guard = guard
        self._model_hint = model_hint
        self._entered: Any = None
        self._model: str | None = None
        self._input = 0
        self._output = 0
        self._cached = 0
        self._recorded = False

    # -- accumulation --------------------------------------------------------

    def _accumulate(self, chunk: Any) -> None:
        for carrier in (chunk, _get(chunk, "message")):
            if carrier is None:
                continue
            model = extract_model(carrier)
            if model:
                self._model = model
            usage = extract_usage(carrier)
            if usage is not None:
                # Anthropic reports the input once on message_start and cumulative
                # output on every message_delta; OpenAI reports full usage once on
                # the final chunk. max() is the merge that is correct for both.
                self._input = max(self._input, usage.input_tokens)
                self._output = max(self._output, usage.output_tokens)
                self._cached = max(self._cached, usage.cached_input_tokens)

    def _finish(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        if self._input or self._output or self._cached:
            self._guard.record(
                model=self._model or self._model_hint or "unknown",
                input_tokens=self._input,
                output_tokens=self._output,
                cached_input_tokens=self._cached,
            )
            return
        warnings.warn(
            "agentguard: the stream finished without reporting token usage, so "
            "this call counts as $0 and will not move the budget. For "
            "OpenAI-compatible clients pass stream_options={'include_usage': True}; "
            "Anthropic streams carry usage on the message_start/message_delta events.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _finalize(self) -> None:
        """Best-effort usage grab for Anthropic's manager-style streams.

        Only used when iteration never surfaced usage: ``get_final_message()``
        consumes the remainder of the stream, which is the wrong thing to do to
        a stream that already accounted for itself.
        """
        if self._recorded:
            return
        if self._input or self._output or self._cached:
            self._finish()
            return
        target = self._entered if self._entered is not None else self._stream
        get_final = getattr(target, "get_final_message", None)
        if callable(get_final):
            # A stream that cannot finalize falls through to the warning.
            with contextlib.suppress(Exception):
                self._accumulate(get_final())
        self._finish()

    # -- sync protocol ---------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        target = self._entered if self._entered is not None else self._stream
        try:
            for chunk in target:
                self._accumulate(chunk)
                yield chunk
        finally:
            # Reached on exhaustion, on break (GeneratorExit) and on close().
            # Inside a context manager the authoritative end is __exit__, which
            # can still consult get_final_message() — do not finish early.
            if self._entered is None:
                self._finish()

    def __enter__(self) -> GuardedStream:
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            entered = enter()
            if entered is not self._stream:
                self._entered = entered
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finalize()
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if exit_ is not None else False

    def close(self) -> None:
        target = self._entered if self._entered is not None else self._stream
        close = getattr(target, "close", None)
        if callable(close):
            close()
        self._finish()

    # -- async protocol --------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Any]:
        target = self._entered if self._entered is not None else self._stream
        try:
            async for chunk in target:
                self._accumulate(chunk)
                yield chunk
        finally:
            if self._entered is None:
                self._finish()

    async def __aenter__(self) -> GuardedStream:
        aenter = getattr(self._stream, "__aenter__", None)
        if aenter is not None:
            entered = await aenter()
            if entered is not self._stream:
                self._entered = entered
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finalize()
        aexit = getattr(self._stream, "__aexit__", None)
        if aexit is not None:
            return await aexit(*exc)
        return False

    def __getattr__(self, name: str) -> Any:
        target = self._entered if self._entered is not None else self._stream
        return getattr(target, name)

    def __repr__(self) -> str:
        return f"GuardedStream({type(self._stream).__name__})"


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

    __slots__ = (
        "_chars_per_token",
        "_depth",
        "_guard",
        "_preflight",
        "_record_on",
        "_stream_on",
        "_target",
    )

    def __init__(
        self,
        target: Any,
        guard: Guard,
        *,
        record_on: Iterable[str] = ("create",),
        stream_on: Iterable[str] = (),
        depth: int = _DEFAULT_DEPTH,
        preflight: bool = False,
        chars_per_token: float = 3.0,
    ) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_guard", guard)
        object.__setattr__(self, "_record_on", tuple(record_on))
        object.__setattr__(self, "_stream_on", tuple(stream_on))
        object.__setattr__(self, "_depth", max(0, int(depth)))
        object.__setattr__(self, "_preflight", bool(preflight))
        object.__setattr__(self, "_chars_per_token", float(chars_per_token))

    @property
    def guard(self) -> Guard:
        """The guard this client reports to."""
        # Annotated rather than cast: object.__getattribute__ is typed as Any,
        # and mypy --strict rejects returning Any from a declared Guard.
        resolved: Guard = object.__getattribute__(self, "_guard")
        return resolved

    @property
    def target(self) -> Any:
        """The wrapped client, for anything the proxy gets in the way of."""
        return object.__getattribute__(self, "_target")

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        attr = getattr(target, name)  # AttributeError propagates untouched

        record_on: tuple[str, ...] = object.__getattribute__(self, "_record_on")
        stream_on: tuple[str, ...] = object.__getattribute__(self, "_stream_on")

        if callable(attr):
            if name in stream_on:
                return self._wrap_stream(attr)
            if name in record_on:
                return self._wrap(attr)

        depth: int = object.__getattribute__(self, "_depth")
        if depth > 0 and _is_resource(attr):
            return GuardedClient(
                attr,
                object.__getattribute__(self, "_guard"),
                record_on=record_on,
                stream_on=stream_on,
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

    def __enter__(self) -> GuardedClient:
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
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return _await_and_record(result, guard, kwargs)
            if kwargs.get("stream"):
                return GuardedStream(result, guard, model_hint=_model_hint(kwargs))
            guard.record(result)
            return result

        return wrapper

    def _wrap_stream(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a method whose *return value* is always a stream."""
        guard = object.__getattribute__(self, "_guard")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return _await_and_record(result, guard, kwargs, always_stream=True)
            return GuardedStream(result, guard, model_hint=_model_hint(kwargs))

        return wrapper


def _model_hint(kwargs: Mapping[str, Any]) -> str | None:
    model = kwargs.get("model")
    return model if isinstance(model, str) and model else None


async def _await_and_record(
    awaitable: Any,
    guard: Guard,
    kwargs: Mapping[str, Any],
    *,
    always_stream: bool = False,
) -> Any:
    """Async twin of the sync wrapper body: record once the coroutine resolves."""
    response = await awaitable
    if always_stream or kwargs.get("stream"):
        return GuardedStream(response, guard, model_hint=_model_hint(kwargs))
    guard.record(response)
    return response


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
    stream_on: Sequence[str] = (),
    depth: int = _DEFAULT_DEPTH,
    preflight: bool = False,
    chars_per_token: float = 3.0,
) -> GuardedClient:
    """Wrap any LLM client so calls are recorded onto ``guard``.

    Use :func:`agentguard.adapters.openai.guard_openai` or
    :func:`agentguard.adapters.anthropic.guard_anthropic` for the common cases;
    this is the generic escape hatch for LiteLLM, OpenRouter, vLLM and friends.

    :param record_on: Method names whose return value is recorded as one call.
        A truthy ``stream=True`` keyword argument makes the result a
        :class:`GuardedStream` instead.
    :param stream_on: Method names that *always* return a stream (for example
        Anthropic's ``messages.stream``), wrapped in :class:`GuardedStream`
        regardless of the call arguments.
    """
    if guard is None:
        raise GuardConfigError("guard_client() requires a Guard instance")
    return GuardedClient(
        target,
        guard,
        record_on=record_on,
        stream_on=stream_on,
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
