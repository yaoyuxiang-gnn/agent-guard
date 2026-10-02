"""Tests for the ``@guarded`` decorator."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from agentguard import (
    BudgetExceeded,
    Detector,
    Guard,
    GuardConfigError,
    LoopDetected,
    LoopVerdict,
    current_guard,
    guarded,
)


class AlwaysDetector(Detector):
    name = "always"

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        return LoopVerdict(self.name, "always trips", signature, 1, step)


class GuardLifecycleTests(unittest.TestCase):
    def test_a_fresh_guard_is_created_per_call(self) -> None:
        seen: list[Guard | None] = []

        @guarded(max_usd=1.0)
        def run() -> str:
            seen.append(current_guard())
            return "ok"

        self.assertEqual(run(), "ok")
        self.assertEqual(run(), "ok")
        self.assertIsNotNone(seen[0])
        self.assertIsNot(seen[0], seen[1])

    def test_fresh_guards_do_not_share_spend(self) -> None:
        @guarded(max_usd=0.003)
        def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=1000)  # $0.0025, under the cap

        # Two calls in a row must both succeed; a shared guard would trip on the
        # second because the spend would have accumulated.
        run()
        run()

    def test_a_shared_guard_accumulates_across_calls(self) -> None:
        shared = Guard(max_usd=100.0)

        @guarded(guard=shared)
        def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=1000)

        run()
        run()
        self.assertEqual(shared.calls, 2)

    def test_a_shared_guard_enforces_its_budget(self) -> None:
        shared = Guard(max_usd=0.001)

        @guarded(guard=shared)
        def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=100_000)

        with self.assertRaises(BudgetExceeded):
            run()


class PlumbingTests(unittest.TestCase):
    def test_the_guard_is_active_inside_and_cleared_afterwards(self) -> None:
        @guarded(max_usd=1.0)
        def run() -> Guard | None:
            return current_guard()

        self.assertIsNotNone(run())
        self.assertIsNone(current_guard())

    def test_arguments_and_return_values_pass_through(self) -> None:
        @guarded(max_usd=1.0)
        def add(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c

        self.assertEqual(add(1, 2, c=3), 6)

    def test_multiple_return_values(self) -> None:
        @guarded(max_usd=1.0)
        def pair() -> tuple[int, str]:
            return 1, "two"

        self.assertEqual(pair(), (1, "two"))

    def test_exceptions_propagate_untouched(self) -> None:
        @guarded(max_usd=1.0)
        def boom() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            boom()

    def test_the_guard_is_cleared_even_when_the_body_raises(self) -> None:
        @guarded(max_usd=1.0)
        def boom() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            boom()
        self.assertIsNone(current_guard())

    def test_functools_metadata_is_preserved(self) -> None:
        @guarded(max_usd=1.0)
        def documented() -> None:
            """A docstring."""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "A docstring.")

    def test_introspection_attributes(self) -> None:
        @guarded(max_usd=1.0, max_steps=5)
        def run() -> None:
            pass

        self.assertEqual(
            run.__agentguard_options__,  # type: ignore[attr-defined]
            {"max_usd": 1.0, "max_steps": 5},
        )

    def test_shared_guard_is_exposed_for_introspection(self) -> None:
        shared = Guard()

        @guarded(guard=shared)
        def run() -> None:
            pass

        self.assertIs(run.__agentguard_shared__, shared)  # type: ignore[attr-defined]


class AsyncDecoratorTests(unittest.TestCase):
    """An ``async def`` body runs after the decorator's wrapper returns.

    The guard therefore has to stay entered for the whole coroutine rather than
    just until the coroutine object is created — otherwise ``current_guard()`` is
    ``None`` inside the body and every call goes unrecorded. That was the bug.
    """

    def test_the_guard_is_live_inside_a_coroutine(self) -> None:
        @guarded(max_usd=1.0)
        async def run() -> Guard | None:
            return current_guard()

        active = asyncio.run(run())
        self.assertIsNotNone(active)

    def test_spend_is_recorded_across_an_await(self) -> None:
        @guarded(max_usd=1.0)
        async def run() -> float:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=1_000)
            await asyncio.sleep(0)
            guard.record("gpt-4o", input_tokens=1_000)
            return guard.spent_usd

        self.assertAlmostEqual(asyncio.run(run()), 0.005)

    def test_a_shared_guard_accumulates_across_awaits(self) -> None:
        shared = Guard(max_usd=100.0)

        @guarded(guard=shared)
        async def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=1_000)

        asyncio.run(run())
        asyncio.run(run())
        self.assertEqual(shared.calls, 2)

    def test_a_fresh_guard_is_created_per_await(self) -> None:
        seen: list[Guard | None] = []

        @guarded(max_usd=1.0)
        async def run() -> None:
            seen.append(current_guard())

        asyncio.run(run())
        asyncio.run(run())
        self.assertIsNotNone(seen[0])
        self.assertIsNot(seen[0], seen[1])

    def test_the_guard_is_cleared_after_the_coroutine_finishes(self) -> None:
        @guarded(max_usd=1.0)
        async def run() -> None:
            return None

        asyncio.run(run())
        self.assertIsNone(current_guard())

    def test_the_guard_is_cleared_when_the_coroutine_raises(self) -> None:
        @guarded(max_usd=1.0)
        async def boom() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            asyncio.run(boom())
        self.assertIsNone(current_guard())

    def test_a_budget_trip_propagates_out_of_a_coroutine(self) -> None:
        @guarded(max_usd=0.001)
        async def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.record("gpt-4o", input_tokens=100_000)

        with self.assertRaises(BudgetExceeded):
            asyncio.run(run())

    def test_loop_detection_works_through_an_async_decorator(self) -> None:
        @guarded(max_usd=1.0, detectors=[AlwaysDetector()])
        async def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.observe("anything")

        with self.assertRaises(LoopDetected):
            asyncio.run(run())

    def test_coroutine_metadata_and_introspection_survive(self) -> None:
        @guarded(max_usd=1.0, max_steps=5)
        async def documented() -> None:
            """An async docstring."""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "An async docstring.")
        self.assertEqual(
            documented.__agentguard_options__,  # type: ignore[attr-defined]
            {"max_usd": 1.0, "max_steps": 5},
        )

    def test_an_async_decorated_function_is_still_a_coroutine_function(self) -> None:
        # Callers inspect this, and so do frameworks that decide how to await.
        @guarded(max_usd=1.0)
        async def run() -> None:
            return None

        self.assertTrue(asyncio.iscoroutinefunction(run))


class AsyncGeneratorDecoratorTests(unittest.TestCase):
    """The generator body does not run when the function is called either."""

    def test_the_guard_stays_live_for_the_whole_iteration(self) -> None:
        seen: list[Guard | None] = []

        @guarded(max_usd=1.0)
        async def stream() -> Any:
            for _ in range(3):
                seen.append(current_guard())
                yield "chunk"

        async def collect() -> list[str]:
            return [chunk async for chunk in stream()]

        self.assertEqual(asyncio.run(collect()), ["chunk"] * 3)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(guard is not None for guard in seen))

    def test_spend_accumulates_while_iterating(self) -> None:
        shared = Guard(max_usd=100.0)

        @guarded(guard=shared)
        async def stream() -> Any:
            guard = current_guard()
            assert guard is not None
            for _ in range(4):
                guard.record("gpt-4o", input_tokens=1_000)
                yield "chunk"

        async def drain() -> int:
            return len([chunk async for chunk in stream()])

        self.assertEqual(asyncio.run(drain()), 4)
        self.assertEqual(shared.calls, 4)
        self.assertIsNone(current_guard())

    def test_abandoning_the_generator_still_exits_the_guard(self) -> None:
        @guarded(max_usd=1.0)
        async def stream() -> Any:
            while True:
                yield "chunk"

        async def take_one() -> str:
            agen = stream()
            first = await agen.__anext__()
            await agen.aclose()
            return first

        self.assertEqual(asyncio.run(take_one()), "chunk")
        self.assertIsNone(current_guard())


class ConfigurationTests(unittest.TestCase):
    def test_rejects_a_guard_and_keyword_arguments_together(self) -> None:
        with self.assertRaises(GuardConfigError):
            guarded(guard=Guard(), max_usd=1.0)

    def test_invalid_guard_options_fail_at_decoration_time(self) -> None:
        with self.assertRaises(GuardConfigError):
            guarded(max_usd=0)

    def test_loop_detection_works_through_the_decorator(self) -> None:
        @guarded(max_usd=1.0, detectors=[AlwaysDetector()])
        def run() -> None:
            guard = current_guard()
            assert guard is not None
            guard.observe("anything")

        with self.assertRaises(LoopDetected):
            run()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
