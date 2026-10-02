"""Decorator form of :class:`~agentguard.Guard`.

Useful when an agent is a single function and you would rather not thread a guard
object through it::

    from agentguard import guarded, current_guard

    @guarded(max_usd=0.50, max_steps=20)
    def summarise(url: str) -> str:
        guard = current_guard()      # the guard installed by the decorator
        ...

``async def`` functions are supported. The guard has to stay entered for as long as
the coroutine runs rather than merely until it is created, so an async function gets
an async wrapper — otherwise the body would execute after ``__exit__`` had already
run, and :func:`~agentguard.current_guard` would return ``None`` inside it::

    @guarded(max_usd=0.50)
    async def summarise(url: str) -> str:
        guard = current_guard()      # the guard is live across every await
        ...
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from .exceptions import GuardConfigError
from .guard import Guard

__all__ = ["guarded"]

F = TypeVar("F", bound=Callable[..., Any])


def guarded(
    *,
    guard: Guard | None = None,
    **guard_kwargs: Any,
) -> Callable[[F], F]:
    """Run the decorated function inside a :class:`~agentguard.Guard`.

    Pass either an existing ``guard`` or the keyword arguments for a new one —
    never both.

    * ``guard=`` reuses one guard, so spend accumulates across every call. Use it
      for a whole job, or for a worker that should have a lifetime budget.
    * keyword arguments create a **fresh guard per call**, which is the right
      default for request handlers: one caller exhausting a budget must not stop
      the next one.

    Works on plain functions, ``async def`` coroutines and ``async`` generators.
    Inside the function, :func:`~agentguard.current_guard` returns the active
    guard::

        @guarded(max_usd=1.0, max_steps=30)
        def run_agent(task: str) -> str:
            guard = current_guard()
            assert guard is not None
            with guard.step() as step:
                response = call_model(task)
                step.record(response)
            return "done"
    """
    if guard is not None and guard_kwargs:
        raise GuardConfigError(
            "guarded() takes either guard=<Guard> or guard keyword arguments, not both"
        )
    if guard is None:
        # Build one throwaway guard so a misconfiguration (`max_usd=0`, a typo in
        # a keyword) fails when the module is imported, not on the first request.
        # Guard construction does no I/O, so this is effectively free.
        Guard(**guard_kwargs)

    def decorator(func: F) -> F:
        # One guard per call, but which *kind* of wrapper is decided once here
        # rather than per call: the guard must stay entered for the whole of the
        # body, and for a coroutine that means across every ``await``.
        build = (lambda: guard) if guard is not None else (lambda: Guard(**guard_kwargs))
        if inspect.isasyncgenfunction(func):
            wrapper = _async_generator_wrapper(func, build)
        elif inspect.iscoroutinefunction(func):
            wrapper = _coroutine_wrapper(func, build)
        else:
            wrapper = _function_wrapper(func, build)

        # Give callers a handle for introspection and testing.
        wrapper.__agentguard_options__ = dict(guard_kwargs)  # type: ignore[attr-defined]
        wrapper.__agentguard_shared__ = guard  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def _function_wrapper(func: Callable[..., Any], build: Callable[[], Guard]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with build():
            return func(*args, **kwargs)

    return wrapper


def _coroutine_wrapper(func: Callable[..., Any], build: Callable[[], Guard]) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with build():
            return await func(*args, **kwargs)

    return wrapper


def _async_generator_wrapper(
    func: Callable[..., Any], build: Callable[[], Guard]
) -> Callable[..., Any]:
    """Keep the guard entered for the whole iteration, not just the first ``next()``.

    An async generator body does not run when it is called, so wrapping the call
    alone would exit the guard before a single item was produced — the
    :func:`~agentguard.current_guard` failure this whole module has to avoid.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        with build():
            async for item in func(*args, **kwargs):
                yield item

    return wrapper
