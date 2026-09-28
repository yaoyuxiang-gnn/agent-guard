"""Decorator form of :class:`~agent_guard.Guard`.

Useful when an agent is a single function and you would rather not thread a guard
object through it::

    from agent_guard import guarded, current_guard

    @guarded(max_usd=0.50, max_steps=20)
    def summarise(url: str) -> str:
        guard = current_guard()      # the guard installed by the decorator
        ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from .exceptions import GuardConfigError
from .guard import Guard

__all__ = ["guarded"]

F = TypeVar("F", bound=Callable[..., Any])


def guarded(
    *,
    guard: Guard | None = None,
    **guard_kwargs: Any,
) -> Callable[[F], F]:
    """Run the decorated function inside a :class:`~agent_guard.Guard`.

    Pass either an existing ``guard`` or the keyword arguments for a new one —
    never both.

    * ``guard=`` reuses one guard, so spend accumulates across every call. Use it
      for a whole job, or for a worker that should have a lifetime budget.
    * keyword arguments create a **fresh guard per call**, which is the right
      default for request handlers: one caller exhausting a budget must not stop
      the next one.

    Inside the function, :func:`~agent_guard.current_guard` returns the active
    guard, and :func:`~agent_guard.Guard.current` does the same::

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
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = guard if guard is not None else Guard(**guard_kwargs)
            with active:
                return func(*args, **kwargs)

        # Give callers a handle for introspection and testing.
        wrapper.__agent_guard_options__ = dict(guard_kwargs)  # type: ignore[attr-defined]
        wrapper.__agent_guard_shared__ = guard  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
