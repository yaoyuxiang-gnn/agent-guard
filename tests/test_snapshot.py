"""Tests for checkpointing a run: ``Guard.snapshot`` / ``restore``.

A checkpoint exists so a resumed run does not start with a fresh budget, which
makes these tests about one question: after a restore, is the arithmetic the
*same*? Anything that lets a restored guard forget money it already spent is a
budget cap that silently stops capping, so the round-trip cases below compare
every breakdown rather than just the total.
"""

from __future__ import annotations

import json
import unittest
import warnings

from agentguard import (
    BudgetExceeded,
    Detector,
    Guard,
    GuardConfigError,
    LoopDetected,
    LoopMonitor,
    LoopVerdict,
    NoProgressDetector,
    RepeatDetector,
    StepLimitExceeded,
    Usage,
)
from agentguard.tracker import SNAPSHOT_VERSION, CostTracker


class FakeClock:
    """Deterministic monotonic clock, so wall-clock assertions are exact."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StatefulDetector(Detector):
    """A custom detector that opts into checkpointing."""

    name = "stateful"

    def __init__(self, *, trip_at: int = 3) -> None:
        self.trip_at = trip_at
        self.seen: list[str] = []

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        self.seen.append(signature)
        if len(self.seen) >= self.trip_at:
            return LoopVerdict(
                self.name, f"saw {len(self.seen)} calls", signature, len(self.seen), step
            )
        return None

    def reset(self) -> None:
        self.seen.clear()

    def get_state(self) -> dict[str, object]:
        return {"seen": list(self.seen)}

    def set_state(self, state: dict[str, object]) -> None:
        self.seen = [str(item) for item in state.get("seen") or []]


class CluelessDetector(Detector):
    """A custom detector that does *not* implement state, like most will not."""

    name = "clueless"


def spent_guard(**kwargs: object) -> Guard:
    """A guard with a known, varied history: two models, a tag, a tool, a gap.

    ``on_unknown_model="ignore"`` because one call is deliberately unpriced, and
    the suite turns warnings into errors; the unpriced behaviour under test is
    that it stays unpriced, not that it warns.
    """
    guard = Guard(  # type: ignore[arg-type]
        max_usd=100.0, max_steps=50, use_config=False, on_unknown_model="ignore", **kwargs
    )
    guard.record(
        "gpt-4o", input_tokens=1_000_000, output_tokens=2_000, tag="retrieval", tool="search"
    )
    with guard.step(tag="summarise") as step:
        step.record("gpt-4o-mini", input_tokens=500_000, output_tokens=1_000)
    guard.record("unpriced-model", input_tokens=10_000, output_tokens=5, tag="retrieval")
    return guard


class TrackerStateTests(unittest.TestCase):
    def test_state_is_json_serialisable(self) -> None:
        state = spent_guard().tracker.get_state()
        json.dumps(state)  # must not raise

    def test_state_carries_no_records(self) -> None:
        # The point of the format is that it is small: no per-call log, no list of
        # records, nothing that grows linearly with the number of calls.
        state = spent_guard().tracker.get_state()
        self.assertEqual(set(state), {"version", "calls", "unpriced_calls", "groups"})
        for group in state["groups"]:
            self.assertNotIn("records", group)

    def test_groups_are_keyed_by_model_tag_and_tool(self) -> None:
        state = spent_guard().tracker.get_state()
        keys = {(g["model"], g["tag"], g["tool"]) for g in state["groups"]}
        self.assertEqual(
            keys,
            {
                ("gpt-4o", "retrieval", "search"),
                ("gpt-4o-mini", "summarise", "(unattributed)"),
                ("unpriced-model", "retrieval", "(unattributed)"),
            },
        )

    def test_round_trip_preserves_every_total(self) -> None:
        original = spent_guard().tracker
        restored = CostTracker.restore(
            json.loads(json.dumps(original.get_state())),
            original.price_table,
            on_unknown_model="ignore",
        )
        self.assertEqual(restored.calls, original.calls)
        self.assertEqual(restored.total_usd, original.total_usd)
        self.assertEqual(restored.usage.as_dict(), original.usage.as_dict())
        self.assertEqual(restored.unpriced_calls, original.unpriced_calls)
        self.assertEqual(restored.unpriced_models, original.unpriced_models)

    def test_round_trip_preserves_every_breakdown(self) -> None:
        original = spent_guard().tracker
        restored = CostTracker.restore(original.get_state(), original.price_table)
        for method in ("by_model", "by_tag", "by_tool"):
            with self.subTest(breakdown=method):
                before = {k: v.as_dict() for k, v in getattr(original, method)().items()}
                after = {k: v.as_dict() for k, v in getattr(restored, method)().items()}
                self.assertEqual(before, after)

    def test_a_group_mixing_priced_and_unpriced_calls_keeps_its_money(self) -> None:
        # An unknown model falls back to default_price, so one group can hold both
        # billed and unbilled calls. Labelling the whole group unpriced would hide
        # real spend from by_tag/by_tool.
        tracker = CostTracker(on_unknown_model="ignore")
        tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), tag="mixed")
        tracker.record(model="who-knows", usage=Usage(5, 5), tag="mixed")
        restored = CostTracker.restore(tracker.get_state(), tracker.price_table)
        self.assertAlmostEqual(restored.by_tag()["mixed"].cost_usd, 2.5)
        self.assertEqual(restored.by_tag()["mixed"].unpriced_calls, 1)
        self.assertEqual(restored.unpriced_calls, 1)

    def test_unpriced_stays_unpriced(self) -> None:
        # Restoring must never turn forgotten money into budget headroom.
        tracker = CostTracker(on_unknown_model="ignore")
        tracker.record(model="who-knows", usage=Usage(1_000, 1_000))
        restored = CostTracker.restore(tracker.get_state(), tracker.price_table)
        self.assertEqual(restored.total_usd, 0.0)
        self.assertEqual(restored.unpriced_calls, 1)
        self.assertIsNone(restored.records[0].cost_usd)

    def test_overlong_window_in_a_handwritten_state_cannot_trip_early(self) -> None:
        # A state file is input: a detector must not accept a history longer than
        # the window it was constructed with.
        detector = RepeatDetector(max_repeats=3, window=4)
        detector.set_state({"recent": ["a"] * 50})
        self.assertEqual(len(detector.get_state()["recent"]), 4)

    def test_wrong_version_is_refused(self) -> None:
        state = spent_guard().tracker.get_state()
        state["version"] = SNAPSHOT_VERSION + 1
        with self.assertRaises(GuardConfigError) as ctx:
            CostTracker.restore(state)
        self.assertIn("format version", str(ctx.exception))

    def test_missing_groups_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            CostTracker.restore({"version": SNAPSHOT_VERSION})

    def test_call_count_mismatch_is_refused(self) -> None:
        state = spent_guard().tracker.get_state()
        state["calls"] = 99
        with self.assertRaises(GuardConfigError) as ctx:
            CostTracker.restore(state)
        self.assertIn("claims 99", str(ctx.exception))

    def test_negative_counts_are_refused(self) -> None:
        bad = {
            "version": SNAPSHOT_VERSION,
            "groups": [{"model": "m", "calls": -1}],
        }
        with self.assertRaises(GuardConfigError):
            CostTracker.restore(bad)

    def test_unpriced_exceeding_calls_is_refused(self) -> None:
        bad = {
            "version": SNAPSHOT_VERSION,
            "groups": [{"model": "m", "calls": 1, "unpriced_calls": 5}],
        }
        with self.assertRaises(GuardConfigError):
            CostTracker.restore(bad)

    def test_cost_on_a_fully_unpriced_group_is_refused(self) -> None:
        bad = {
            "version": SNAPSHOT_VERSION,
            "groups": [{"model": "m", "calls": 2, "unpriced_calls": 2, "cost_usd": 5.0}],
        }
        with self.assertRaises(GuardConfigError) as ctx:
            CostTracker.restore(bad)
        self.assertIn("cannot have a cost", str(ctx.exception))

    def test_non_mapping_state_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            CostTracker.restore(["not", "a", "mapping"])  # type: ignore[arg-type]


class DetectorStateTests(unittest.TestCase):
    def test_repeat_detector_round_trips(self) -> None:
        detector = RepeatDetector(max_repeats=3, window=6)
        for signature in ("a", "b", "a"):
            detector.observe(signature, 1)
        restored = RepeatDetector(max_repeats=3, window=6)
        restored.set_state(detector.get_state())
        self.assertEqual(restored.get_state(), detector.get_state())

    def test_repeat_detector_trips_across_a_restore(self) -> None:
        detector = RepeatDetector(max_repeats=3, window=6)
        detector.observe("a", 1)
        detector.observe("a", 2)
        restored = RepeatDetector(max_repeats=3, window=6)
        restored.set_state(json.loads(json.dumps(detector.get_state())))
        # The third identical call is the one that trips, which only works if the
        # two observations from before the checkpoint survived.
        verdict = restored.observe("a", 3)
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertEqual(verdict.count, 3)

    def test_no_progress_detector_round_trips(self) -> None:
        detector = NoProgressDetector(max_stagnant=4)
        for value in ("a", "b", "b"):
            detector.observe(value, 1)
        restored = NoProgressDetector(max_stagnant=4)
        restored.set_state(json.loads(json.dumps(detector.get_state())))
        self.assertEqual(restored.get_state(), {"last": "b", "same": 2})
        # Two observations in, so two more unchanged markers reach the cap of 4.
        self.assertIsNone(restored.observe("b", 2))
        self.assertIsNotNone(restored.observe("b", 3))

    def test_no_progress_state_without_a_marker_resets_the_count(self) -> None:
        detector = NoProgressDetector(max_stagnant=4)
        detector.set_state({"last": None, "same": 7})
        self.assertEqual(detector.get_state(), {"last": None, "same": 0})

    def test_base_detector_refuses_to_pretend(self) -> None:
        # Returning {} would make an uncheckpointable detector look checkpointed.
        with self.assertRaises(NotImplementedError):
            Detector().get_state()
        with self.assertRaises(NotImplementedError):
            Detector().set_state({})

    def test_monitor_reports_none_for_detectors_that_cannot_checkpoint(self) -> None:
        monitor = LoopMonitor([RepeatDetector(), CluelessDetector(), NoProgressDetector()])
        states = monitor.get_states()
        self.assertEqual(len(states), 3)
        self.assertIsInstance(states[0], dict)
        self.assertIsNone(states[1])
        self.assertIsInstance(states[2], dict)

    def test_monitor_restores_by_position(self) -> None:
        source = LoopMonitor([StatefulDetector(trip_at=3)])
        source.observe("x", 1)
        target = LoopMonitor([StatefulDetector(trip_at=3)])
        target.set_states(source.get_states())
        # Two observations after the restore reaches this detector's cap of three,
        # which only happens if the one from before the checkpoint came back.
        self.assertIsNone(target.observe("x", 2))
        self.assertIsNotNone(target.observe("x", 3))

    def test_monitor_tolerates_a_different_detector_list(self) -> None:
        # A checkpoint may come from a guard with a different detector list; the
        # budget must still restore, so this cannot raise.
        monitor = LoopMonitor([RepeatDetector()])
        monitor.set_states([{"recent": ["a"]}, {"recent": ["b"]}, None])
        self.assertEqual(monitor.get_states(), [{"recent": ["a"]}])

    def test_monitor_skips_state_for_detectors_that_cannot_restore(self) -> None:
        monitor = LoopMonitor([CluelessDetector()])
        monitor.set_states([{"anything": 1}])  # must not raise


class GuardSnapshotTests(unittest.TestCase):
    def test_snapshot_is_json_serialisable(self) -> None:
        json.dumps(spent_guard().snapshot())

    def test_snapshot_has_a_version(self) -> None:
        self.assertEqual(spent_guard().snapshot()["version"], SNAPSHOT_VERSION)

    def test_round_trip_preserves_every_reported_number(self) -> None:
        guard = spent_guard()
        before = guard.report()
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=100.0, use_config=False)
        after = resumed.report()
        self.assertEqual(after.cost_usd, before.cost_usd)
        self.assertEqual(after.steps, before.steps)
        self.assertEqual(after.calls, before.calls)
        self.assertEqual(after.usage.as_dict(), before.usage.as_dict())
        self.assertEqual(
            [s.as_dict() for s in after.by_model], [s.as_dict() for s in before.by_model]
        )
        self.assertEqual([s.as_dict() for s in after.by_tag], [s.as_dict() for s in before.by_tag])
        self.assertEqual(
            [s.as_dict() for s in after.by_tool], [s.as_dict() for s in before.by_tool]
        )
        self.assertEqual(after.unpriced_models, before.unpriced_models)
        self.assertEqual(after.unpriced_calls, before.unpriced_calls)

    def test_round_trip_accepts_a_dict_or_json(self) -> None:
        guard = spent_guard()
        as_dict = Guard.from_snapshot(guard.snapshot(), max_usd=100.0, use_config=False)
        as_text = Guard.from_snapshot(guard.as_snapshot(), max_usd=100.0, use_config=False)
        self.assertEqual(as_dict.spent_usd, as_text.spent_usd)

    def test_restored_spend_counts_against_the_new_budget(self) -> None:
        # The whole reason the feature exists: a resumed run must not get a fresh
        # budget. $2.6 already spent, a $3 cap, so only ~$0.4 may remain.
        guard = spent_guard()
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=3.0, use_config=False)
        self.assertAlmostEqual(resumed.spent_usd, guard.spent_usd)
        self.assertLess(resumed.remaining_usd or 0.0, 0.5)
        self.assertGreater(resumed.remaining_usd or 0.0, 0.0)

    def test_a_restored_guard_trips_on_the_next_overspend(self) -> None:
        guard = spent_guard()
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=2.6, use_config=False)
        with self.assertRaises(BudgetExceeded):
            resumed.record("gpt-4o", input_tokens=10_000_000, output_tokens=0)

    def test_steps_carry_over_and_still_count(self) -> None:
        guard = Guard(max_usd=100.0, max_steps=3, use_config=False)
        with guard.step():
            pass
        resumed = Guard.from_snapshot(
            guard.as_snapshot(), max_usd=100.0, max_steps=3, use_config=False
        )
        self.assertEqual(resumed.steps, 1)
        with resumed.step():
            pass
        with resumed.step():
            pass
        self.assertEqual(resumed.steps, 3)
        # The fourth step exceeds max_steps=3.
        with self.assertRaises(StepLimitExceeded):
            resumed.step()

    def test_elapsed_time_is_deliberately_not_restored(self) -> None:
        # max_seconds caps *this process*; restoring an elapsed duration would make
        # a resumed run trip on time it never spent.
        clock = FakeClock()
        guard = Guard(max_usd=1.0, max_seconds=10.0, use_config=False, clock=clock)
        clock.advance(9.0)
        snapshot = guard.snapshot()
        resumed = Guard.from_snapshot(
            snapshot, max_usd=1.0, max_seconds=10.0, use_config=False, clock=clock
        )
        self.assertEqual(resumed.elapsed_s, 0.0)
        clock.advance(1.0)
        self.assertEqual(resumed.elapsed_s, 1.0)
        self.assertIsNone(resumed.tripped)  # 1s into a 10s cap, not 10s

    def test_a_loop_spanning_the_checkpoint_still_trips(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False)
        signature = Guard.call_signature("search", {"q": "same"})
        guard.observe(signature)
        guard.observe(signature)
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=1.0, use_config=False)
        # RepeatDetector needs three identical observations; the first two came
        # from before the checkpoint.
        with self.assertRaises(LoopDetected) as ctx:
            resumed.observe(signature)
        self.assertIn("repeat", str(ctx.exception))

    def test_progress_detector_state_survives(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False)
        for _ in range(3):
            guard.progress({"rows": 0})
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=1.0, use_config=False)
        # NoProgressDetector's default is 6 unchanged markers; 3 more must trip.
        with self.assertRaises(LoopDetected) as ctx:
            for _ in range(3):
                resumed.progress({"rows": 0})
        self.assertIn("no-progress", str(ctx.exception))

    def test_report_says_how_much_it_inherited(self) -> None:
        guard = spent_guard()
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=100.0, use_config=False)
        self.assertEqual(resumed.report().checkpointed_calls, guard.calls)

    def test_a_fresh_run_reports_nothing_inherited(self) -> None:
        self.assertEqual(spent_guard().report().checkpointed_calls, 0)

    def test_the_rendered_report_admits_the_checkpoint(self) -> None:
        resumed = Guard.from_snapshot(spent_guard().as_snapshot(), max_usd=100.0, use_config=False)
        text = resumed.report().render(ascii_only=True)
        self.assertIn("restored from a checkpoint", text)
        # And a run that never restored says nothing about one.
        self.assertNotIn("checkpoint", spent_guard().report().render(ascii_only=True))

    def test_report_dict_carries_the_checkpoint_count(self) -> None:
        resumed = Guard.from_snapshot(spent_guard().as_snapshot(), max_usd=100.0, use_config=False)
        self.assertEqual(resumed.as_dict()["checkpointed_calls"], resumed.calls)

    def test_report_round_trips_through_from_dict(self) -> None:
        from agentguard import Report

        resumed = Guard.from_snapshot(spent_guard().as_snapshot(), max_usd=100.0, use_config=False)
        data = resumed.as_dict()
        self.assertEqual(Report.from_dict(data).checkpointed_calls, data["checkpointed_calls"])

    def test_restore_replaces_whatever_the_guard_already_had(self) -> None:
        donor = spent_guard()
        target = Guard(max_usd=100.0, use_config=False)
        target.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        self.assertAlmostEqual(target.spent_usd, 2.5)
        target.restore(donor.snapshot())
        self.assertAlmostEqual(target.spent_usd, donor.spent_usd)
        self.assertEqual(target.calls, donor.calls)

    def test_reset_clears_an_inherited_checkpoint(self) -> None:
        guard = Guard.from_snapshot(spent_guard().as_snapshot(), max_usd=100.0, use_config=False)
        self.assertGreater(guard.spent_usd, 0.0)
        guard.reset()
        self.assertEqual(guard.spent_usd, 0.0)
        self.assertEqual(guard.report().checkpointed_calls, 0)

    def test_detector_without_state_warns_that_history_was_lost(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, detectors=[CluelessDetector()])
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["detectors"]["actions"], [None])
        with self.assertWarns(RuntimeWarning) as ctx:
            Guard.from_snapshot(
                snapshot, max_usd=1.0, use_config=False, detectors=[CluelessDetector()]
            )
        self.assertIn("history", str(ctx.warning))

    def test_detector_with_state_does_not_warn(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, detectors=[StatefulDetector()])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Guard.from_snapshot(
                guard.snapshot(), max_usd=1.0, use_config=False, detectors=[StatefulDetector()]
            )

    def test_stateful_custom_detector_resumes_its_window(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False, detectors=[StatefulDetector(trip_at=3)])
        guard.observe("call-1")
        guard.observe("call-2")
        resumed = Guard.from_snapshot(
            guard.as_snapshot(),
            max_usd=1.0,
            use_config=False,
            detectors=[StatefulDetector(trip_at=3)],
        )
        with self.assertRaises(LoopDetected) as ctx:
            resumed.observe("call-3")
        self.assertIn("stateful", str(ctx.exception))

    def test_loop_detection_disabled_round_trips(self) -> None:
        guard = Guard(
            max_usd=100.0, use_config=False, loop_detection=False, on_unknown_model="ignore"
        )
        guard.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        resumed = Guard.from_snapshot(
            guard.as_snapshot(), max_usd=100.0, use_config=False, loop_detection=False
        )
        self.assertAlmostEqual(resumed.spent_usd, 2.5)

    def test_snapshot_of_an_empty_run_restores_to_an_empty_run(self) -> None:
        guard = Guard(max_usd=1.0, use_config=False)
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=1.0, use_config=False)
        self.assertEqual(resumed.calls, 0)
        self.assertEqual(resumed.spent_usd, 0.0)
        self.assertEqual(resumed.report().checkpointed_calls, 0)

    def test_indent_keyword_pretty_prints(self) -> None:
        text = spent_guard().as_snapshot(indent=2)
        self.assertIn("\n", text)
        self.assertEqual(json.loads(text)["version"], SNAPSHOT_VERSION)


class GuardSnapshotRefusalTests(unittest.TestCase):
    """A snapshot it cannot read exactly must be refused, never half-applied."""

    def assertRefused(self, payload: object) -> GuardConfigError:
        with self.assertRaises(GuardConfigError) as ctx:
            Guard.from_snapshot(payload, use_config=False)  # type: ignore[arg-type]
        return ctx.exception

    def test_wrong_snapshot_version(self) -> None:
        exc = self.assertRefused({"version": 99, "guard": {"steps": 0}, "tracker": {}})
        self.assertIn("format version", str(exc))

    def test_wrong_tracker_version(self) -> None:
        exc = self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"version": 99, "groups": []},
            }
        )
        self.assertIn("format version", str(exc))

    def test_missing_tracker(self) -> None:
        exc = self.assertRefused({"version": SNAPSHOT_VERSION, "guard": {"steps": 0}})
        self.assertIn("tracker", str(exc))

    def test_negative_steps(self) -> None:
        self.assertRefused(
            {"version": SNAPSHOT_VERSION, "guard": {"steps": -1}, "tracker": {"groups": []}}
        )

    def test_boolean_steps_is_not_an_integer(self) -> None:
        # bool is an int subclass, so a naive isinstance check would accept True.
        self.assertRefused(
            {"version": SNAPSHOT_VERSION, "guard": {"steps": True}, "tracker": {"groups": []}}
        )

    def test_guard_state_must_be_a_mapping(self) -> None:
        self.assertRefused(
            {"version": SNAPSHOT_VERSION, "guard": "nope", "tracker": {"groups": []}}
        )

    def test_detector_states_must_be_a_list(self) -> None:
        self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"groups": []},
                "detectors": {"actions": "nope"},
            }
        )

    def test_detector_state_entries_must_be_mappings_or_null(self) -> None:
        self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"groups": []},
                "detectors": {"actions": [7]},
            }
        )

    def test_invalid_json(self) -> None:
        exc = self.assertRefused("{not json")
        self.assertIn("valid JSON", str(exc))

    def test_json_that_is_not_an_object(self) -> None:
        exc = self.assertRefused("[1, 2, 3]")
        self.assertIn("mapping or JSON object", str(exc))

    def test_group_missing_a_model(self) -> None:
        self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"groups": [{"calls": 1}]},
            }
        )

    def test_group_with_a_non_integer_call_count(self) -> None:
        self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"groups": [{"model": "m", "calls": "two"}]},
            }
        )

    def test_group_with_a_non_numeric_cost(self) -> None:
        self.assertRefused(
            {
                "version": SNAPSHOT_VERSION,
                "guard": {"steps": 0},
                "tracker": {"groups": [{"model": "m", "calls": 1, "cost_usd": "free"}]},
            }
        )

    def test_group_must_be_a_mapping(self) -> None:
        self.assertRefused(
            {"version": SNAPSHOT_VERSION, "guard": {"steps": 0}, "tracker": {"groups": ["nope"]}}
        )

    def test_a_refused_restore_leaves_the_guard_untouched(self) -> None:
        # The tracker is rebuilt before anything is assigned, so a bad snapshot
        # cannot leave a half-adopted budget behind.
        guard = Guard(max_usd=100.0, use_config=False)
        guard.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        before = guard.report()
        with self.assertRaises(GuardConfigError):
            guard.restore(
                {
                    "version": SNAPSHOT_VERSION,
                    "guard": {"steps": 0},
                    "tracker": {"groups": [{"model": "m", "calls": -1}]},
                }
            )
        after = guard.report()
        self.assertEqual(after.cost_usd, before.cost_usd)
        self.assertEqual(after.calls, before.calls)


class SnapshotDoesNotLeakRecordsTests(unittest.TestCase):
    def test_records_are_not_recoverable_after_a_restore(self) -> None:
        # Pinning the documented limitation rather than pretending otherwise: the
        # restored tracker has the right totals and no real call log.
        guard = spent_guard()
        resumed = Guard.from_snapshot(guard.as_snapshot(), max_usd=100.0, use_config=False)
        self.assertEqual(len(resumed.tracker.records), guard.calls)
        for record in resumed.tracker.records:
            self.assertEqual(record.at, 0.0)  # no real timestamp survived
            self.assertEqual(record.meta, {})

    def test_per_record_cost_is_an_average_not_a_measurement(self) -> None:
        tracker = CostTracker(on_unknown_model="ignore")
        tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0))
        tracker.record(model="gpt-4o", usage=Usage(1_000, 0))
        restored = CostTracker.restore(tracker.get_state(), tracker.price_table)
        # $2.50125 across two calls in one group: the group total is exact and the
        # per-call figure is its average, which is all the checkpoint knows.
        costs = [r.cost_usd for r in restored.records]
        self.assertEqual(costs, [1.25125, 1.25125])
        self.assertAlmostEqual(restored.total_usd, tracker.total_usd)
