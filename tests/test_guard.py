"""Tests for :class:`agentguard.Guard`.

Everything here is deterministic: the clock is injected, so the time limit is
tested without sleeping, and nothing touches the network.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

from agentguard import (
    BudgetExceeded,
    BudgetScopeExceeded,
    Detector,
    Guard,
    GuardConfigError,
    GuardStopped,
    GuardTripped,
    LoopDetected,
    LoopVerdict,
    Price,
    RepeatDetector,
    Report,
    StepLimitExceeded,
    TimeLimitExceeded,
    TokenLimitExceeded,
    current_guard,
)
from agentguard.guard import Step
from agentguard.tracker import UNATTRIBUTED


class FakeClock:
    """A monotonic clock that only moves when a test tells it to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AlwaysDetector(Detector):
    name = "always"

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        return LoopVerdict(self.name, "always trips", signature, 1, step)


class GuardConfigTests(unittest.TestCase):
    def test_rejects_non_positive_limits(self) -> None:
        for kwargs in (
            {"max_usd": 0},
            {"max_usd": -1},
            {"max_steps": 0},
            {"max_tokens": -5},
            {"max_seconds": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(GuardConfigError):
                Guard(**kwargs)

    def test_rejects_non_numeric_limits(self) -> None:
        with self.assertRaises(GuardConfigError):
            Guard(max_usd="cheap")  # type: ignore[arg-type]
        with self.assertRaises(GuardConfigError):
            Guard(max_steps=True)

    def test_rejects_unknown_trip_mode(self) -> None:
        with self.assertRaises(GuardConfigError):
            Guard(on_trip="explode")

    def test_a_guard_with_no_limits_is_still_useful(self) -> None:
        guard = Guard()
        guard.record("gpt-4o", input_tokens=10, output_tokens=5)
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.report().limits, ())

    def test_repr_reports_state(self) -> None:
        guard = Guard(max_usd=1.0)
        self.assertIn("spent=$0.0000/$1.0000", repr(guard))


class BudgetTests(unittest.TestCase):
    def test_does_not_trip_at_exactly_the_limit(self) -> None:
        guard = Guard(max_usd=1.0)
        guard.record("gpt-4o", input_tokens=400_000, output_tokens=0)
        self.assertFalse(guard.stopped)
        self.assertAlmostEqual(guard.spent_usd, 1.0)

    def test_trips_when_exceeded(self) -> None:
        guard = Guard(max_usd=1.0)
        with self.assertRaises(BudgetExceeded) as ctx:
            guard.record("gpt-4o", input_tokens=500_000, output_tokens=0)
        self.assertEqual(ctx.exception.reason, "budget")
        self.assertAlmostEqual(ctx.exception.spent_usd, 1.25)
        self.assertEqual(ctx.exception.limit_usd, 1.0)

    def test_the_offending_call_is_still_recorded(self) -> None:
        # Losing the record of the call that broke the budget would make the
        # post-mortem useless.
        guard = Guard(max_usd=0.01)
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertEqual(guard.calls, 1)
        self.assertGreater(guard.spent_usd, 0.01)

    def test_further_calls_are_refused_after_a_trip(self) -> None:
        guard = Guard(max_usd=0.001)
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=1, output_tokens=0)
        with self.assertRaises(BudgetExceeded):
            guard.step()

    def test_remaining_usd(self) -> None:
        guard = Guard(max_usd=1.0)
        self.assertAlmostEqual(guard.remaining_usd or 0.0, 1.0)
        guard.record("gpt-4o", input_tokens=200_000, output_tokens=0)
        self.assertAlmostEqual(guard.remaining_usd or 0.0, 0.5)
        self.assertIsNone(Guard().remaining_usd)

    def test_unknown_model_cannot_silently_hide_an_overspend(self) -> None:
        guard = Guard(max_usd=0.01)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("mystery-model", input_tokens=10_000_000, output_tokens=0)
        self.assertFalse(guard.stopped)
        self.assertEqual(guard.report().unpriced_calls, 1)
        self.assertEqual(guard.report().unpriced_models, ("mystery-model",))
        self.assertTrue(any("no price" in str(w.message) for w in caught))


class TokenAndStepTests(unittest.TestCase):
    def test_token_limit(self) -> None:
        guard = Guard(max_tokens=100)
        with self.assertRaises(TokenLimitExceeded) as ctx:
            guard.record("gpt-4o", input_tokens=60, output_tokens=60)
        self.assertEqual(ctx.exception.used_tokens, 120)
        self.assertEqual(ctx.exception.limit_tokens, 100)

    def test_step_limit(self) -> None:
        guard = Guard(max_steps=2)
        with guard.step(), guard.step():
            pass
        self.assertEqual(guard.steps, 2)
        with self.assertRaises(StepLimitExceeded) as ctx:
            guard.step()
        self.assertEqual(ctx.exception.limit_steps, 2)

    def test_time_limit_uses_the_injected_clock(self) -> None:
        clock = FakeClock()
        guard = Guard(max_seconds=10.0, clock=clock)
        guard.check()
        clock.advance(5.0)
        guard.check()
        self.assertFalse(guard.stopped)
        clock.advance(6.0)
        with self.assertRaises(TimeLimitExceeded):
            guard.check()

    def test_elapsed_uses_the_injected_clock(self) -> None:
        clock = FakeClock()
        guard = Guard(clock=clock)
        clock.advance(2.5)
        self.assertAlmostEqual(guard.elapsed_s, 2.5)


class LoopDetectionTests(unittest.TestCase):
    def test_repeated_tool_call_trips(self) -> None:
        guard = Guard()
        with self.assertRaises(LoopDetected) as ctx:
            for _ in range(4):
                with guard.tool("search", {"q": "weather"}):
                    pass
        self.assertEqual(ctx.exception.kind, "repeat")
        self.assertIn("search", ctx.exception.detail)

    def test_arguments_in_a_different_key_order_still_count_as_the_same_call(self) -> None:
        guard = Guard()
        with self.assertRaises(LoopDetected):
            with guard.tool("s", {"a": 1, "b": 2}):
                pass
            with guard.tool("s", {"b": 2, "a": 1}):
                pass
            with guard.tool("s", {"a": 1, "b": 2}):
                pass

    def test_observe_accepts_raw_signatures(self) -> None:
        guard = Guard()
        with self.assertRaises(LoopDetected):
            for _ in range(3):
                guard.observe("my_tool(1)")

    def test_progress_stagnation_trips(self) -> None:
        guard = Guard()
        with self.assertRaises(LoopDetected) as ctx:
            for _ in range(6):
                guard.progress(42)
        self.assertEqual(ctx.exception.kind, "no-progress")

    def test_progress_that_moves_never_trips(self) -> None:
        guard = Guard()
        for value in range(20):
            guard.progress(value)
        self.assertFalse(guard.stopped)

    def test_loop_detection_can_be_disabled(self) -> None:
        guard = Guard(loop_detection=False)
        for _ in range(10):
            guard.observe("same()")
        self.assertFalse(guard.stopped)
        self.assertEqual(guard.detectors, ())

    def test_custom_detectors_replace_the_defaults(self) -> None:
        guard = Guard(detectors=[AlwaysDetector()])
        with self.assertRaises(LoopDetected) as ctx:
            guard.observe("anything")
        self.assertEqual(ctx.exception.kind, "always")

    def test_custom_progress_detectors(self) -> None:
        guard = Guard(progress_detectors=[AlwaysDetector()])
        with self.assertRaises(LoopDetected):
            guard.progress("first value")

    def test_call_signature_staticmethod_matches_the_function(self) -> None:
        self.assertEqual(Guard.call_signature("t", {"a": 1}), Guard.call_signature("t", {"a": 1}))

    def test_detector_configuration_errors_surface_at_construction(self) -> None:
        with self.assertRaises(GuardConfigError):
            Guard(detectors=[RepeatDetector(max_repeats=1)])


class TripModeTests(unittest.TestCase):
    def test_warn_mode_does_not_raise(self) -> None:
        guard = Guard(max_usd=0.001, on_trip="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertTrue(guard.stopped)
        self.assertIsInstance(guard.tripped, BudgetExceeded)
        self.assertTrue(any(isinstance(w.message, RuntimeWarning) for w in caught))

    def test_warn_mode_warns_only_once(self) -> None:
        guard = Guard(max_usd=0.001, on_trip="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertEqual(len(caught), 1)

    def test_stop_mode_records_without_raising_at_the_moment_it_trips(self) -> None:
        guard = Guard(max_usd=0.001, on_trip="stop")
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertTrue(guard.stopped)
        self.assertIsInstance(guard.tripped, BudgetExceeded)
        # The call that tripped the guard is still accounted for, and the caller's
        # current step is allowed to finish so it can clean up.
        self.assertEqual(guard.calls, 1)
        # From here on, every entry point refuses to do more work.
        with self.assertRaises(GuardStopped):
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)

    def test_callback_fires_exactly_once(self) -> None:
        seen: list[GuardTripped] = []
        guard = Guard(max_usd=0.001, on_trip="stop", on_trip_callback=seen.append)
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        with self.assertRaises(GuardStopped):
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], BudgetExceeded)

    def test_a_broken_callback_does_not_mask_the_trip(self) -> None:
        def explode(_: GuardTripped) -> None:
            raise RuntimeError("callback exploded")

        guard = Guard(max_usd=0.001, on_trip="stop", on_trip_callback=explode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertTrue(guard.stopped)
        self.assertTrue(any("callback" in str(w.message) for w in caught))


class StopModeTests(unittest.TestCase):
    """``on_trip="stop"`` used to depend on the caller remembering to check.

    "The caller's loop is responsible for breaking out" is a fine intention and a
    bad guarantee: a loop that forgets keeps spending, which is the failure this
    library exists to prevent. Now the trip is recorded, the current step finishes,
    and everything after it is refused.
    """

    def tripped_guard(self, **kwargs: object) -> Guard:
        guard = Guard(max_usd=0.001, on_trip="stop", **kwargs)
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        assert guard.stopped
        return guard

    def test_the_tripping_call_is_still_recorded(self) -> None:
        guard = self.tripped_guard()
        self.assertEqual(guard.calls, 1)
        self.assertAlmostEqual(guard.spent_usd, 0.25)

    def test_guard_stopped_carries_the_original_trip(self) -> None:
        guard = self.tripped_guard()
        with self.assertRaises(GuardStopped) as ctx:
            guard.step()
        self.assertIsInstance(ctx.exception.cause, BudgetExceeded)
        # `reason` mirrors the cause, so handlers that switch on it keep working.
        self.assertEqual(ctx.exception.reason, "budget")
        self.assertIs(ctx.exception.cause, guard.tripped)

    def test_guard_stopped_is_a_guard_tripped(self) -> None:
        guard = self.tripped_guard()
        with self.assertRaises(GuardTripped):
            guard.record("gpt-4o", input_tokens=1, output_tokens=0)

    def test_every_entry_point_refuses_after_a_stop(self) -> None:
        calls = {
            "step": lambda g: g.step(),
            "record": lambda g: g.record("gpt-4o", input_tokens=1, output_tokens=0),
            "observe": lambda g: g.observe("search(q=1)"),
            "progress": lambda g: g.progress(1),
            "check": lambda g: g.check(),
            "preflight": lambda g: g.preflight("gpt-4o", input_tokens=1),
        }
        for name, call in calls.items():
            with self.subTest(entry_point=name):
                guard = self.tripped_guard()
                with self.assertRaises(GuardStopped):
                    call(guard)

    def test_a_tool_body_never_runs_after_a_stop(self) -> None:
        guard = self.tripped_guard()
        ran = False
        with self.assertRaises(GuardStopped), guard.tool("search", {"q": "x"}):
            ran = True
        self.assertFalse(ran)

    def test_a_stopped_guard_still_calls_the_callback_once(self) -> None:
        seen: list[GuardTripped] = []
        guard = Guard(max_usd=0.001, on_trip="stop", on_trip_callback=seen.append)
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        with self.assertRaises(GuardStopped):
            guard.record("gpt-4o", input_tokens=1, output_tokens=0)
        self.assertEqual(len(seen), 1)

    def test_the_report_is_complete_for_a_stopped_run(self) -> None:
        guard = self.tripped_guard()
        report = guard.report()
        self.assertEqual(report.calls, 1)
        self.assertEqual(report.tripped_reason, "budget")
        self.assertIn("$0.25", report.render(ascii_only=True))

    def test_warn_mode_never_raises_and_keeps_counting(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            guard = Guard(max_usd=0.001, on_trip="warn")
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
            guard.record("gpt-4o", input_tokens=1, output_tokens=0)
            guard.check()
        self.assertEqual(guard.calls, 2)

    def test_raise_mode_accounts_for_a_call_made_after_the_trip(self) -> None:
        # The money was spent before the exception was raised, so the report has to
        # show it. Dropping the record would under-report the very overspend this
        # guard exists to make visible.
        guard = Guard(max_usd=1.0)
        guard.record("gpt-4o", input_tokens=390_000, output_tokens=0)  # $0.975
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=42_000, output_tokens=0)  # $0.105
        self.assertEqual(guard.calls, 2)
        self.assertAlmostEqual(guard.spent_usd, 0.975 + 0.105)

    def test_raise_mode_still_raises_the_original_trip_from_check(self) -> None:
        guard = Guard(max_usd=0.001)
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        with self.assertRaises(BudgetExceeded):
            guard.check()


class PreflightTests(unittest.TestCase):
    def test_refuses_a_call_that_would_overshoot(self) -> None:
        guard = Guard(max_usd=1.0)
        guard.record("gpt-4o", input_tokens=390_000, output_tokens=0)  # $0.975
        with self.assertRaises(BudgetExceeded) as ctx:
            guard.preflight("gpt-4o", input_tokens=10_000, max_output_tokens=10_000)
        self.assertIsNotNone(ctx.exception.projected_usd)
        self.assertAlmostEqual(ctx.exception.call_cost_usd or 0.0, 0.125)
        # Nothing was spent by the refused call.
        self.assertAlmostEqual(guard.spent_usd, 0.975)

    def test_allows_an_affordable_call_and_returns_its_worst_case(self) -> None:
        guard = Guard(max_usd=1.0)
        guard.record("gpt-4o", input_tokens=390_000, output_tokens=0)
        worst = guard.preflight("gpt-4o", input_tokens=100, max_output_tokens=100)
        self.assertAlmostEqual(worst, 0.00125)

    def test_unpriced_models_are_not_guessed_at(self) -> None:
        guard = Guard(max_usd=0.01)
        self.assertEqual(
            guard.preflight("mystery", input_tokens=10**9, max_output_tokens=10**9), 0.0
        )
        self.assertFalse(guard.stopped)

    def test_unknown_model_returns_zero_without_guessing(self) -> None:
        guard = Guard(max_usd=1.0)
        self.assertEqual(
            guard.preflight("anything", input_tokens=1_000_000, max_output_tokens=0), 0.0
        )

    def test_explicit_price_overrides_the_table(self) -> None:
        guard = Guard(max_usd=100.0)
        worst = guard.preflight(
            "anything",
            input_tokens=1_000_000,
            max_output_tokens=0,
            price=Price(2.0, 2.0),
        )
        self.assertAlmostEqual(worst, 2.0)
        self.assertFalse(guard.stopped)


class IntegrationDepthTests(unittest.TestCase):
    def test_record_accepts_a_bare_model_string(self) -> None:
        guard = Guard()
        record = guard.record("gpt-4o", input_tokens=1000, output_tokens=200)
        self.assertEqual(record.model, "gpt-4o")
        self.assertTrue(record.priced)

    def test_record_extracts_from_a_response_object(self) -> None:
        guard = Guard()
        response = NS(
            model="gpt-4o",
            usage=NS(prompt_tokens=1000, completion_tokens=200),
        )
        record = guard.record(response)
        self.assertEqual(record.model, "gpt-4o")
        self.assertEqual(record.usage.total_tokens, 1200)

    def test_explicit_counts_win_over_extracted_ones(self) -> None:
        guard = Guard()
        response = NS(model="gpt-4o", usage=NS(prompt_tokens=1, completion_tokens=1))
        record = guard.record(response, input_tokens=500, output_tokens=100)
        self.assertEqual(record.usage.total_tokens, 600)

    def test_explicit_model_overrides_the_extracted_one(self) -> None:
        guard = Guard()
        response = NS(model="gpt-4o", usage=NS(prompt_tokens=1, completion_tokens=1))
        record = guard.record(response, model="gpt-4o-mini")
        self.assertEqual(record.model, "gpt-4o-mini")

    def test_step_attribution(self) -> None:
        guard = Guard()
        with guard.step(tag="research") as step:
            record = step.record("gpt-4o", input_tokens=10)
        self.assertEqual(record.step, 1)
        self.assertEqual(record.tag, "research")

    def test_records_inside_a_step_block_pick_up_the_index(self) -> None:
        guard = Guard()
        with guard.step():
            record = guard.record("gpt-4o", input_tokens=10)
        self.assertEqual(record.step, 1)

    def test_step_tool_registers_for_loop_detection(self) -> None:
        guard = Guard()
        with self.assertRaises(LoopDetected), guard.step() as step:
            for _ in range(4):
                with step.tool("search", {"q": "same"}):
                    pass

    def test_step_returns_the_signature(self) -> None:
        guard = Guard()
        with guard.step() as step, step.tool("search", {"q": "x"}) as signature:
            self.assertIn("search", signature)

    def test_step_repr(self) -> None:
        guard = Guard()
        with guard.step(tag="t") as step:
            self.assertEqual(repr(step), "Step(index=1, tag='t')")

    def test_response_with_no_usage_warns(self) -> None:
        guard = Guard()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record(NS(model="gpt-4o"))
        self.assertTrue(any("token usage" in str(w.message) for w in caught))

    def test_the_no_usage_warning_fires_once(self) -> None:
        guard = Guard()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record(NS(model="gpt-4o"))
            guard.record(NS(model="gpt-4o"))
        self.assertEqual(len(caught), 1)

    def test_unrecognised_model_name_is_counted_as_unknown(self) -> None:
        guard = Guard(on_unknown_model="ignore")
        record = guard.record(input_tokens=10, output_tokens=10)
        self.assertEqual(record.model, "unknown")
        self.assertFalse(record.priced)


class ContextTests(unittest.TestCase):
    def test_current_guard_inside_a_with_block(self) -> None:
        guard = Guard()
        self.assertIsNone(current_guard())
        with guard:
            self.assertIs(current_guard(), guard)
        self.assertIsNone(current_guard())

    def test_nested_guards_restore_the_outer_one(self) -> None:
        outer = Guard(name="outer")
        inner = Guard(name="inner")
        with outer:
            with inner:
                self.assertIs(current_guard(), inner)
            self.assertIs(current_guard(), outer)
        self.assertIsNone(current_guard())

    def test_reentering_the_same_guard_restores_correctly(self) -> None:
        guard = Guard()
        with guard:
            with guard:
                self.assertIs(current_guard(), guard)
            self.assertIs(current_guard(), guard)
        self.assertIsNone(current_guard())

    def test_guard_does_not_swallow_exceptions(self) -> None:
        guard = Guard()
        with self.assertRaises(ValueError), guard:
            raise ValueError("from the agent")

    def test_one_shared_guard_entered_from_two_threads(self) -> None:
        # Regression: entry tokens live in a context-local stack, so two threads
        # inside ``with guard:`` at the same time each pop their own token.
        # A shared list pops the other thread's token and ``ContextVar.reset``
        # raises "Token was created in a different Context".
        guard = Guard()
        first_inside = threading.Event()
        second_inside = threading.Event()
        errors: list[BaseException] = []

        def first() -> None:
            try:
                with guard:
                    first_inside.set()
                    self.assertIs(current_guard(), guard)
                    # Stay inside until the second thread is in too, so the two
                    # entries genuinely overlap.
                    second_inside.wait(timeout=5)
            except BaseException as exc:  # reported to the test body
                errors.append(exc)

        def second() -> None:
            try:
                first_inside.wait(timeout=5)
                with guard:
                    self.assertIs(current_guard(), guard)
                    second_inside.set()
            except BaseException as exc:  # reported to the test body
                errors.append(exc)

        t1, t2 = threading.Thread(target=first), threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        self.assertIsNone(current_guard())


class LifecycleTests(unittest.TestCase):
    def test_reset_clears_everything(self) -> None:
        guard = Guard(max_usd=1.0)
        guard.record("gpt-4o", input_tokens=1000, output_tokens=100)
        guard.step()
        self.assertEqual((guard.calls, guard.steps), (1, 1))

        guard.reset()
        self.assertEqual((guard.calls, guard.steps), (0, 0))
        self.assertFalse(guard.stopped)
        self.assertEqual(guard.spent_usd, 0.0)
        self.assertIsNone(guard.tripped)

    def test_reset_after_a_trip_makes_the_guard_usable_again(self) -> None:
        guard = Guard(max_usd=0.001, on_trip="stop")
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertTrue(guard.stopped)
        with self.assertRaises(GuardStopped):
            guard.record("gpt-4o", input_tokens=10, output_tokens=0)
        guard.reset()
        guard.record("gpt-4o", input_tokens=10, output_tokens=0)
        self.assertFalse(guard.stopped)

    def test_reset_preserves_the_unknown_model_policy(self) -> None:
        guard = Guard(on_unknown_model="error")
        guard.reset()
        with self.assertRaises(GuardConfigError):
            guard.record("mystery", input_tokens=1)


class UnpricedModelPolicyTests(unittest.TestCase):
    """``on_unknown_model="error"`` must not cost the guard its bookkeeping.

    The call has already been made by the time agent-guard sees it, so it has
    already been paid for. An error raised *instead of* the record leaves the report
    claiming the call never happened — the one thing a cost report must not do, and
    by ``SECURITY.md``'s own definition a guard bypass.
    """

    def test_the_refused_call_is_still_counted(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="error")
        with self.assertRaises(GuardConfigError):
            guard.record("mystery-model", input_tokens=1_000, output_tokens=500)
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.spent_usd, 0.0)  # unknown price, not a guessed one

    def test_the_report_names_the_model_it_could_not_price(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="error")
        with self.assertRaises(GuardConfigError):
            guard.record("mystery-model", input_tokens=1_000, output_tokens=500)
        report = guard.report()
        self.assertEqual(report.calls, 1)
        self.assertEqual(report.unpriced_calls, 1)
        self.assertEqual(report.unpriced_models, ("mystery-model",))
        self.assertIn("mystery-model", report.render(ascii_only=True))

    def test_a_priced_call_recorded_alongside_is_still_billed(self) -> None:
        guard = Guard(max_usd=100.0, use_config=False, on_unknown_model="error")
        guard.record("gpt-4o", input_tokens=1_000_000)
        with self.assertRaises(GuardConfigError):
            guard.record("mystery-model", input_tokens=1_000)
        self.assertEqual(guard.calls, 2)
        self.assertAlmostEqual(guard.spent_usd, 2.5)
        self.assertEqual(guard.report().unpriced_calls, 1)

    def test_a_budget_trip_still_wins_when_the_priced_call_crosses_the_cap(self) -> None:
        # A priced call is recorded, the limit is re-evaluated, and *that* is what
        # surfaces — the budget failure is not masked by a later pricing failure.
        guard = Guard(max_usd=0.001, use_config=False, on_unknown_model="error")
        with self.assertRaises(BudgetExceeded):
            guard.record("gpt-4o", input_tokens=1_000_000)
        self.assertEqual(guard.calls, 1)

    def test_the_guard_still_refuses_the_next_call_after_the_error(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="error")
        with self.assertRaises(GuardConfigError):
            guard.record("mystery-model", input_tokens=1)
        with self.assertRaises(GuardConfigError):
            guard.record("another-mystery", input_tokens=1)
        self.assertEqual(guard.calls, 2)

    def test_warn_mode_records_without_raising(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("mystery-model", input_tokens=1_000)
        self.assertEqual(guard.calls, 1)
        self.assertEqual(len(caught), 1)

    def test_ignore_mode_is_silent_and_still_counts(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="ignore")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record("mystery-model", input_tokens=1_000)
        self.assertEqual(guard.calls, 1)
        self.assertEqual(caught, [])

    def test_a_call_with_no_identifiable_model_says_so(self) -> None:
        # A response carrying no model name at all lands on "unknown", and the
        # warning has to name the real problem — an extraction failure — rather than
        # telling the reader to price a model called "unknown".
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guard.record({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        self.assertEqual(guard.calls, 1)
        self.assertEqual(len(caught), 1)
        self.assertIn("could not determine the model", str(caught[0].message))


class ScopedBudgetTests(unittest.TestCase):
    """A whole-run cap cannot say *which* part of a run is too expensive."""

    def test_a_tool_cap_trips_and_names_the_tool(self) -> None:
        # The cap is checked as each call is recorded, so it fires on the call that
        # crosses it rather than at the end of the block. The limit is stated below
        # the first call's cost + the second's, so the boundary is not load-bearing.
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 0.004},
        )
        with self.assertRaises(BudgetScopeExceeded) as ctx, guard.tool("search"):
            guard.record("gpt-4o", input_tokens=1_000)  # $0.0025, under
            guard.record("gpt-4o", input_tokens=1_000)  # $0.0050, over
        self.assertEqual(ctx.exception.scope, "tool")
        self.assertEqual(ctx.exception.name, "search")
        self.assertAlmostEqual(ctx.exception.spent_usd, 0.005)
        self.assertAlmostEqual(ctx.exception.limit_usd, 0.004)

    def test_a_tag_cap_trips_and_names_the_tag(self) -> None:
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tag:summarise": 0.001},
        )
        with self.assertRaises(BudgetScopeExceeded) as ctx, guard.step(tag="summarise"):
            guard.record("gpt-4o", input_tokens=1_000)  # $0.0025, over $0.001
        self.assertEqual(ctx.exception.scope, "tag")
        self.assertEqual(ctx.exception.name, "summarise")

    def test_the_tripping_call_is_still_accounted(self) -> None:
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 0.001},
        )
        with self.assertRaises(BudgetScopeExceeded), guard.tool("search"):
            guard.record("gpt-4o", input_tokens=1_000)
        self.assertEqual(guard.calls, 1)
        self.assertAlmostEqual(guard.spent_usd, 0.0025)

    def test_another_scope_is_unaffected(self) -> None:
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 0.001},
        )
        with guard.tool("write"):
            guard.record("gpt-4o", input_tokens=1_000_000)
        self.assertFalse(guard.stopped)

    def test_the_run_budget_is_reported_before_the_scope_budget(self) -> None:
        # When both are over, "the run is over budget" is the more urgent fact.
        guard = Guard(
            max_usd=0.001,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 0.001},
        )
        with self.assertRaises(BudgetExceeded) as ctx, guard.tool("search"):
            guard.record("gpt-4o", input_tokens=1_000)
        self.assertNotIsInstance(ctx.exception, BudgetScopeExceeded)

    def test_scope_spend_is_available_without_a_cap(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        with guard.step(tag="search"):
            guard.record("gpt-4o", input_tokens=1_000_000)
        self.assertAlmostEqual(guard.scope_spend("tag", "search"), 2.5)
        self.assertAlmostEqual(guard.scope_spend("tool", "nothing"), 0.0)

    def test_report_renders_each_scope_as_its_own_limit_row(self) -> None:
        guard = Guard(
            max_usd=5.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 1.0, "tag:index": 0.5},
        )
        with guard.step(tag="index"), guard.tool("search"):
            guard.record("gpt-4o", input_tokens=100_000)
        rendered = guard.report().render(ascii_only=True)
        self.assertIn("tool:search", rendered)
        self.assertIn("tag:index", rendered)

    def test_scoped_limits_survive_the_report_round_trip(self) -> None:
        guard = Guard(
            max_usd=5.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 1.0},
        )
        with guard.tool("search"):
            guard.record("gpt-4o", input_tokens=100_000)
        rebuilt = Report.from_dict(guard.report().as_dict())
        row = next(item for item in rebuilt.limits if item.name == "tool:search")
        self.assertEqual(row.scope, "tool")
        self.assertEqual(row.scope_name, "search")
        self.assertAlmostEqual(row.used, 0.25)

    def test_a_scoped_cap_counts_spend_restored_from_a_checkpoint(self) -> None:
        original = Guard(max_usd=100.0, use_config=False, on_unknown_model="ignore")
        with original.tool("search"):
            original.record("gpt-4o", input_tokens=400_000)  # $1.00
        resumed = Guard.from_snapshot(
            original.snapshot(),
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 1.0},
        )
        self.assertAlmostEqual(resumed.scope_spend("tool", "search"), 1.0)
        with self.assertRaises(BudgetScopeExceeded), resumed.tool("search"):
            resumed.record("gpt-4o", input_tokens=1_000)

    def test_invalid_scope_keys_fail_at_construction(self) -> None:
        for bad in (
            {"search": 1.0},
            {"tool:": 1.0},
            {"model:gpt-4o": 1.0},
            {"tool:search": 0},
            {"tool:search": -1},
            {"tool:search": float("nan")},
            {"tool:search": float("inf")},
            {"tool:search": True},
            {"tool:search": "1.0"},
        ):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Guard(use_config=False, scoped_budgets=bad)

    def test_a_non_mapping_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            Guard(use_config=False, scoped_budgets=[("tool:search", 1.0)])  # type: ignore[arg-type]

    def test_an_unpriced_call_adds_nothing_to_a_scope(self) -> None:
        # Unpriced is not the same as free: counting it as $0.00 would put a bucket
        # in the totals that reads as "this scope has spent nothing".
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:search": 0.001},
        )
        with guard.tool("search"):
            guard.record("mystery-model", input_tokens=10_000_000)
        self.assertAlmostEqual(guard.scope_spend("tool", "search"), 0.0)
        self.assertFalse(guard.stopped)

    def test_scope_budgets_are_checked_in_a_stable_order(self) -> None:
        # Two scopes tripping on one call must always report the same one, or the
        # verdict changes between runs.
        reported = set()
        for _ in range(5):
            guard = Guard(
                max_usd=100.0,
                use_config=False,
                on_unknown_model="ignore",
                scoped_budgets={"tool:aaa": 0.001, "tool:zzz": 0.001},
            )
            with self.assertRaises(BudgetScopeExceeded) as ctx:
                with guard.tool("aaa"):
                    guard.record("gpt-4o", input_tokens=1_000)
                with guard.tool("zzz"):
                    guard.record("gpt-4o", input_tokens=1_000)
            reported.add(ctx.exception.name)
        self.assertEqual(len(reported), 1)


class WorkerContextTests(unittest.TestCase):
    """No thread inherits a contextvars context, so attribution needs carrying."""

    def test_a_worker_call_is_accounted_even_with_no_context(self) -> None:
        # The money is never lost — only the attribution is, which is exactly why
        # this is a documentation problem as much as a code one.
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        with guard.step(tag="outer"), ThreadPoolExecutor(1) as pool:
            record = pool.submit(guard.record, "gpt-4o", input_tokens=1_000).result()
        self.assertEqual(guard.calls, 1)
        self.assertIsNone(record.step)
        self.assertIsNone(record.tag)

    def test_neither_thread_kind_inherits_the_context(self) -> None:
        # Pinned because it is easy to assume otherwise, and because the fix
        # (bind/context) is only necessary if this is true.
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        from_pool: list[object] = []

        def worker() -> None:
            from_pool.append(guard.record("gpt-4o", input_tokens=10))

        with guard.step(tag="threaded"):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            with ThreadPoolExecutor(1) as pool:
                pooled = pool.submit(guard.record, "gpt-4o", input_tokens=10).result()
        self.assertIsNone(from_pool[0].tag)  # type: ignore[attr-defined]
        self.assertIsNone(pooled.tag)

    def test_bind_carries_the_step_tag_and_tool(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        with guard.step(tag="fan-out") as step, guard.tool("fetch"):
            bound = guard.bind(guard.record)
            with ThreadPoolExecutor(2) as pool:
                records = list(pool.map(lambda _: bound("gpt-4o", input_tokens=100), range(2)))
        for record in records:
            self.assertEqual(record.step, step.index)
            self.assertEqual(record.tag, "fan-out")
            self.assertEqual(record.tool, "fetch")

    def test_context_carries_the_attribution_through_a_worker(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")

        def worker(ctx):
            with ctx:
                return guard.record("gpt-4o", input_tokens=100)

        with guard.step(tag="ctx-tag") as step, guard.tool("ctx-tool"):
            ctx = guard.context()
            with ThreadPoolExecutor(1) as pool:
                record = pool.submit(worker, ctx).result()
        self.assertEqual(record.step, step.index)
        self.assertEqual(record.tag, "ctx-tag")
        self.assertEqual(record.tool, "ctx-tool")

    def test_attribution_reaches_by_tag_and_by_tool(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")

        def worker(ctx):
            with ctx:
                return guard.record("gpt-4o", input_tokens=1_000_000)

        with guard.step(tag="fan"), guard.tool("fetch"):
            ctx = guard.context()
            with ThreadPoolExecutor(1) as pool:
                pool.submit(worker, ctx).result()
        self.assertIn("fetch", guard.tracker.by_tool())
        self.assertIn("fan", guard.tracker.by_tag())

    def test_a_scoped_budget_fires_inside_a_worker(self) -> None:
        # The concrete consequence of the empty context: without it a tool cap
        # silently never fires, which is a limit that reports it is working while
        # doing nothing.
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="ignore",
            scoped_budgets={"tool:fetch": 0.005},
        )

        def worker(ctx):
            with ctx:
                return guard.record("gpt-4o", input_tokens=2_000)

        with guard.step(tag="fan"), guard.tool("fetch"):
            ctx = guard.context()
            with self.assertRaises(BudgetScopeExceeded), ThreadPoolExecutor(4) as pool:
                list(pool.map(worker, [ctx] * 4))

    def test_context_outside_any_block_is_a_no_op(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        with guard.context():
            record = guard.record("gpt-4o", input_tokens=10)
        self.assertIsNone(record.step)
        self.assertIsNone(record.tag)
        self.assertEqual(guard.calls, 1)

    def test_an_inner_step_restores_the_outer_one_on_exit(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")
        with guard.step(tag="outer") as outer:
            with guard.step(tag="inner"):
                pass
            record = guard.record("gpt-4o", input_tokens=10)
        self.assertEqual(record.tag, "outer")
        self.assertEqual(record.step, outer.index)

    def test_bind_preserves_the_wrapped_functions_metadata(self) -> None:
        guard = Guard(max_usd=10.0, use_config=False, on_unknown_model="ignore")

        def documented() -> None:
            """A docstring."""

        self.assertEqual(guard.bind(documented).__name__, "documented")
        self.assertEqual(guard.bind(documented).__doc__, "A docstring.")


class PreflightStrictTests(unittest.TestCase):
    """The unpriced hole in a gate that callers treat as a hard stop."""

    def test_an_unpriced_model_is_allowed_through_by_default(self) -> None:
        guard = Guard(max_usd=0.01, use_config=False, on_unknown_model="warn")
        self.assertEqual(guard.preflight("mystery", input_tokens=10_000_000), 0.0)
        self.assertFalse(guard.stopped)

    def test_strict_refuses_an_unpriced_model(self) -> None:
        guard = Guard(
            max_usd=0.01, use_config=False, on_unknown_model="warn", preflight_strict=True
        )
        with self.assertRaises(BudgetExceeded) as ctx:
            guard.preflight("mystery", input_tokens=10_000_000)
        self.assertTrue(ctx.exception.unpriced_model)
        self.assertEqual(ctx.exception.model, "mystery")

    def test_strict_defaults_to_on_for_on_unknown_model_error(self) -> None:
        # Someone who has already said "an unpriced model is an error" does not want
        # the pre-flight gate waving one through.
        guard = Guard(max_usd=0.01, use_config=False, on_unknown_model="error")
        with self.assertRaises(BudgetExceeded):
            guard.preflight("mystery", input_tokens=1)

    def test_strict_can_be_switched_off_explicitly(self) -> None:
        guard = Guard(
            max_usd=0.01, use_config=False, on_unknown_model="error", preflight_strict=False
        )
        self.assertEqual(guard.preflight("mystery", input_tokens=1), 0.0)

    def test_the_per_call_override_wins(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, on_unknown_model="warn")
        with self.assertRaises(BudgetExceeded):
            guard.preflight("mystery", input_tokens=1, strict=True)
        other = Guard(max_usd=1.0, use_config=False, on_unknown_model="warn", preflight_strict=True)
        self.assertEqual(other.preflight("mystery", input_tokens=1, strict=False), 0.0)

    def test_a_priced_model_is_checked_the_same_either_way(self) -> None:
        for strict in (True, False):
            with self.subTest(strict=strict):
                guard = Guard(
                    max_usd=0.01,
                    use_config=False,
                    preflight_strict=strict,
                    on_unknown_model="ignore",
                )
                with self.assertRaises(BudgetExceeded) as ctx:
                    guard.preflight("gpt-4o", input_tokens=1_000_000, max_output_tokens=1_000_000)
                self.assertFalse(ctx.exception.unpriced_model)

    def test_a_default_price_makes_strict_moot(self) -> None:
        guard = Guard(
            max_usd=100.0,
            use_config=False,
            on_unknown_model="warn",
            preflight_strict=True,
            default_price=Price(1.0, 1.0),
        )
        # 1M input at $1/1M, plus the default 1,024-token output bound.
        self.assertAlmostEqual(guard.preflight("mystery", input_tokens=1_000_000), 1.001024)
        self.assertAlmostEqual(
            guard.preflight("mystery", input_tokens=1_000_000, max_output_tokens=0), 1.0
        )
        self.assertFalse(guard.stopped)


class AttributionTests(unittest.TestCase):
    """Which tool is eating the budget? The report should answer it directly."""

    def build(self) -> Guard:
        guard = Guard(max_usd=5.0, name="attribution")
        with guard.step(tag="search") as step:
            step.record("gpt-4o", input_tokens=20_000, output_tokens=1_000)
        with guard.step(tag="summarise") as step:
            with step.tool("fetch", {"url": "x"}):
                step.record("gpt-4o", input_tokens=50_000, output_tokens=2_000)
            step.record("gpt-4o-mini", input_tokens=10_000, output_tokens=500)
        guard.record("gpt-4o-mini", input_tokens=1_000, output_tokens=100)
        return guard

    def test_a_call_inside_a_tool_block_is_attributed_to_it(self) -> None:
        guard = Guard(max_usd=5.0)
        with guard.tool("fetch", {"url": "x"}):
            guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        record = guard.tracker.records[0]
        self.assertEqual(record.tool, "fetch")
        self.assertAlmostEqual(guard.report().by_tool[0].cost_usd, 0.0025)

    def test_a_step_tool_block_attributes_too(self) -> None:
        guard = Guard(max_usd=5.0)
        with guard.step() as step, step.tool("search", {"q": "x"}):
            step.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        self.assertEqual(guard.tracker.records[0].tool, "search")

    def test_the_tool_attribution_ends_with_the_block(self) -> None:
        guard = Guard(max_usd=5.0)
        with guard.tool("fetch"):
            guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        self.assertIsNone(guard.tracker.records[1].tool)

    def test_nesting_restores_the_outer_tool(self) -> None:
        guard = Guard(max_usd=5.0)
        with guard.tool("outer"):
            with guard.tool("inner"):
                guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
            guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        tools = [record.tool for record in guard.tracker.records]
        self.assertEqual(tools, ["inner", "outer"])

    def test_an_explicit_tool_argument_wins(self) -> None:
        guard = Guard(max_usd=5.0)
        with guard.tool("outer"):
            guard.record("gpt-4o", input_tokens=1_000, output_tokens=0, tool="explicit")
        self.assertEqual(guard.tracker.records[0].tool, "explicit")

    def test_the_tool_is_restored_even_when_the_body_raises(self) -> None:
        guard = Guard(max_usd=5.0)
        with self.assertRaises(ValueError), guard.tool("fetch"):
            raise ValueError("tool blew up")
        guard.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        self.assertIsNone(guard.tracker.records[0].tool)

    def test_the_report_breaks_cost_down_by_tag_and_tool(self) -> None:
        report = self.build().report()
        tags = {s.name: s for s in report.by_tag}
        tools = {s.name: s for s in report.by_tool}
        self.assertEqual(set(tags), {"search", "summarise", UNATTRIBUTED})
        self.assertEqual(set(tools), {"fetch", UNATTRIBUTED})
        # Both breakdowns add up to the run total: nothing is silently dropped.
        for breakdown in (report.by_tag, report.by_tool):
            self.assertAlmostEqual(sum(s.cost_usd for s in breakdown), report.cost_usd)

    def test_the_text_report_shows_the_breakdowns(self) -> None:
        text = self.build().report().render(ascii_only=True)
        self.assertIn("by tag", text)
        self.assertIn("by tool", text)
        self.assertIn("fetch", text)
        self.assertIn(UNATTRIBUTED, text)

    def test_a_single_bucket_is_not_worth_printing(self) -> None:
        # One tag just repeats the total, so the section stays out of the way.
        guard = Guard(max_usd=5.0)
        with guard.step(tag="only") as step:
            step.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        report = guard.report()
        self.assertEqual(len(report.by_tag), 1)
        self.assertNotIn("by tag", report.render(ascii_only=True))

    def test_long_breakdowns_collapse_their_tail(self) -> None:
        guard = Guard(max_usd=5.0)
        for index in range(7):
            with guard.step(tag=f"tag-{index}") as step:
                step.record("gpt-4o", input_tokens=1_000, output_tokens=0)
        text = guard.report().render(ascii_only=True)
        self.assertIn("... 2 more", text)

    def test_the_breakdown_survives_a_json_round_trip(self) -> None:
        report = self.build().report()
        restored = Report.from_dict(json.loads(json.dumps(report.as_dict())))
        self.assertEqual(
            [(s.name, s.calls) for s in restored.by_tag],
            [(s.name, s.calls) for s in report.by_tag],
        )
        self.assertEqual(
            [(s.name, s.calls) for s in restored.by_tool],
            [(s.name, s.calls) for s in report.by_tool],
        )


class ReportingTests(unittest.TestCase):
    def build_guard(self) -> Guard:
        guard = Guard(max_usd=1.0, max_steps=10, name="agent")
        guard.record("gpt-4o", input_tokens=1000, output_tokens=100)
        guard.step()
        return guard

    def test_report_contents(self) -> None:
        report = self.build_guard().report()
        self.assertEqual(report.name, "agent")
        self.assertEqual(report.calls, 1)
        self.assertEqual(report.steps, 1)
        self.assertEqual(report.usage.total_tokens, 1100)
        self.assertGreater(report.cost_usd, 0)
        self.assertEqual([limit.name for limit in report.limits], ["budget", "steps"])
        self.assertEqual([m.model for m in report.by_model], ["gpt-4o"])

    def test_report_render_mentions_the_essentials(self) -> None:
        text = self.build_guard().report().render(ascii_only=True)
        # The tool names itself `agentguard` in output, never `agent-guard`: an
        # unrelated package owns the hyphenated name on PyPI, and a user who sees
        # it here would reasonably wonder whether they installed the wrong thing.
        self.assertIn("agentguard", text)
        self.assertIn("agent", text)
        self.assertIn("gpt-4o", text)
        self.assertIn("budget", text)

    def test_report_captures_a_loop_trip(self) -> None:
        guard = Guard(on_trip="stop")
        for _ in range(3):
            guard.observe("same()")
        with self.assertRaises(GuardStopped):
            guard.observe("same()")
        report = guard.report()
        assert report.trip is not None
        self.assertEqual(report.trip.kind, "repeat")

    def test_report_captures_a_non_loop_trip_reason(self) -> None:
        guard = Guard(max_usd=0.001, on_trip="stop")
        guard.record("gpt-4o", input_tokens=100_000, output_tokens=0)
        self.assertEqual(guard.report().tripped_reason, "budget")

    def test_as_dict_and_to_json_are_valid_json(self) -> None:
        guard = self.build_guard()
        self.assertEqual(json.loads(guard.to_json())["calls"], 1)
        self.assertEqual(guard.as_dict()["name"], "agent")

    def test_save_writes_a_loadable_report(self) -> None:
        guard = self.build_guard()
        with tempfile.TemporaryDirectory() as tmp:
            path = guard.save(Path(tmp) / "run.json")
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["calls"], 1)
        self.assertEqual(data["name"], "agent")

    def test_str_of_report_is_the_rendered_text(self) -> None:
        report = self.build_guard().report()
        self.assertEqual(str(report), report.render())


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_recording_is_lossless(self) -> None:
        guard = Guard(max_usd=1000.0, on_unknown_model="ignore")

        def work() -> None:
            for _ in range(100):
                guard.record("gpt-4o", input_tokens=1000, output_tokens=0)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(guard.calls, 800)
        self.assertAlmostEqual(guard.spent_usd, 800 * 1000 * 2.5 / 1_000_000, places=9)

    def test_step_indexes_are_unique_across_threads(self) -> None:
        guard = Guard(max_steps=1000)
        seen: list[int] = []
        lock = threading.Lock()

        def work() -> None:
            for _ in range(50):
                step = guard.step()
                with lock:
                    seen.append(step.index)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(seen), list(range(1, 201)))


class StepObjectTests(unittest.TestCase):
    def test_step_is_a_context_manager(self) -> None:
        guard = Guard()
        step = guard.step()
        self.assertIsInstance(step, Step)

    def test_step_exit_does_not_suppress_exceptions(self) -> None:
        guard = Guard()
        with self.assertRaises(ValueError), guard.step():
            raise ValueError("boom")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
