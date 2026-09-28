"""Tests for the loop detectors.

The failure modes worth guarding against here are false positives: a detector
that kills healthy agents is worse than no detector at all. Every threshold is
therefore tested at the boundary, on both sides.
"""

from __future__ import annotations

import unittest

from agent_guard import GuardConfigError
from agent_guard.loop import (
    CycleDetector,
    LoopMonitor,
    LoopVerdict,
    NoProgressDetector,
    RepeatDetector,
    SimilarityDetector,
    call_signature,
    default_detectors,
    default_progress_detectors,
)


def feed(detector: object, signatures: list[str]) -> LoopVerdict | None:
    """Send each signature through a detector, returning the first verdict."""
    first = None
    for index, signature in enumerate(signatures, start=1):
        verdict = detector.observe(signature, index)  # type: ignore[attr-defined]
        if verdict is not None and first is None:
            first = verdict
    return first


class CallSignatureTests(unittest.TestCase):
    def test_key_order_does_not_change_the_signature(self) -> None:
        self.assertEqual(
            call_signature("search", {"q": "a", "n": 1}),
            call_signature("search", {"n": 1, "q": "a"}),
        )

    def test_none_args(self) -> None:
        self.assertEqual(call_signature("ping", None), "ping()")

    def test_different_args_differ(self) -> None:
        self.assertNotEqual(call_signature("s", {"q": 1}), call_signature("s", {"q": 2}))

    def test_long_signatures_are_truncated(self) -> None:
        signature = call_signature("big", {"blob": "x" * 5000}, max_len=100)
        self.assertTrue(signature.startswith("big("))
        self.assertIn("more chars>", signature)
        self.assertLess(len(signature), 200)

    def test_unserialisable_args_still_produce_a_signature(self) -> None:
        signature = call_signature("weird", {"obj": object()})
        self.assertTrue(signature.startswith("weird("))


class RepeatDetectorTests(unittest.TestCase):
    def test_does_not_trip_below_the_threshold(self) -> None:
        detector = RepeatDetector(max_repeats=3, window=6)
        self.assertIsNone(feed(detector, ["a", "a"]))

    def test_trips_at_the_threshold(self) -> None:
        verdict = feed(RepeatDetector(max_repeats=3, window=6), ["a", "a", "a"])
        assert verdict is not None
        self.assertEqual(verdict.kind, "repeat")
        self.assertEqual(verdict.count, 3)

    def test_counts_non_consecutive_repeats(self) -> None:
        verdict = feed(RepeatDetector(max_repeats=3, window=8), ["a", "b", "a", "c", "a"])
        assert verdict is not None
        self.assertEqual(verdict.count, 3)

    def test_window_eviction_prevents_a_trip(self) -> None:
        # max_repeats=3 but the window only holds 2 entries, so "a" can never
        # accumulate three sightings.
        self.assertIsNone(feed(RepeatDetector(max_repeats=3, window=3), ["a", "b", "c", "a"]))

    def test_reset_clears_history(self) -> None:
        detector = RepeatDetector(max_repeats=2, window=4)
        self.assertIsNotNone(feed(detector, ["a", "a"]))
        detector.reset()
        self.assertIsNone(detector.observe("a", 10))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(GuardConfigError):
            RepeatDetector(max_repeats=1)
        with self.assertRaises(GuardConfigError):
            RepeatDetector(max_repeats=5, window=3)


class CycleDetectorTests(unittest.TestCase):
    def test_trips_on_a_two_step_ping_pong(self) -> None:
        verdict = feed(CycleDetector(min_cycle=2, max_cycle=3, repeats=3), ["a", "b"] * 3)
        assert verdict is not None
        self.assertEqual(verdict.kind, "cycle")

    def test_does_not_trip_before_enough_repetitions(self) -> None:
        self.assertIsNone(feed(CycleDetector(min_cycle=2, max_cycle=3, repeats=3), ["a", "b"] * 2))

    def test_all_identical_is_left_to_the_repeat_detector(self) -> None:
        self.assertIsNone(feed(CycleDetector(repeats=2), ["a"] * 12))

    def test_three_step_cycle(self) -> None:
        verdict = feed(CycleDetector(min_cycle=2, max_cycle=3, repeats=3), ["a", "b", "c"] * 3)
        assert verdict is not None
        self.assertEqual(verdict.kind, "cycle")

    def test_a_changing_sequence_never_trips(self) -> None:
        self.assertIsNone(feed(CycleDetector(repeats=2), [f"step-{i}" for i in range(20)]))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(GuardConfigError):
            CycleDetector(min_cycle=1)
        with self.assertRaises(GuardConfigError):
            CycleDetector(min_cycle=3, max_cycle=2)
        with self.assertRaises(GuardConfigError):
            CycleDetector(repeats=1)


class SimilarityDetectorTests(unittest.TestCase):
    def test_trips_on_near_duplicates(self) -> None:
        detector = SimilarityDetector(threshold=0.9, window=8, max_similar=2)
        verdict = feed(
            detector,
            [
                "search(q=python asyncio)",
                "search(q=python asyncio )",
                "search(q=python asyncio  )",
            ],
        )
        assert verdict is not None
        self.assertEqual(verdict.kind, "similarity")

    def test_exact_repeats_count_as_similar(self) -> None:
        detector = SimilarityDetector(threshold=0.99, window=8, max_similar=2)
        self.assertIsNotNone(feed(detector, ["same"] * 3))

    def test_genuinely_different_calls_do_not_trip(self) -> None:
        detector = SimilarityDetector(threshold=0.9, window=8, max_similar=2)
        self.assertIsNone(
            feed(
                detector,
                [
                    "search(q=python asyncio)",
                    "search(q=weather in oslo)",
                    "write(path=notes.md)",
                ],
            )
        )

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(GuardConfigError):
            SimilarityDetector(threshold=0.0)
        with self.assertRaises(GuardConfigError):
            SimilarityDetector(threshold=1.5)
        with self.assertRaises(GuardConfigError):
            SimilarityDetector(window=0)
        with self.assertRaises(GuardConfigError):
            SimilarityDetector(max_similar=0)


class NoProgressDetectorTests(unittest.TestCase):
    def test_trips_only_after_enough_stagnation(self) -> None:
        detector = NoProgressDetector(max_stagnant=3)
        self.assertIsNone(feed(detector, ["5", "5"]))
        self.assertIsNotNone(feed(detector, ["5"]))

    def test_changing_markers_reset_the_counter(self) -> None:
        detector = NoProgressDetector(max_stagnant=3)
        self.assertIsNone(feed(detector, ["1", "2", "3", "4", "5", "6"]))

    def test_interleaved_stagnation_still_trips(self) -> None:
        detector = NoProgressDetector(max_stagnant=3)
        verdict = feed(detector, ["1", "2", "2", "2"])
        assert verdict is not None
        self.assertEqual(verdict.kind, "no-progress")

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(GuardConfigError):
            NoProgressDetector(max_stagnant=1)


class LoopMonitorTests(unittest.TestCase):
    def test_returns_the_first_verdict(self) -> None:
        monitor = LoopMonitor([RepeatDetector(max_repeats=2, window=4)])
        self.assertIsNone(monitor.observe("a", 1))
        verdict = monitor.observe("a", 2)
        assert verdict is not None
        self.assertEqual(verdict.kind, "repeat")

    def test_every_detector_sees_every_observation(self) -> None:
        # Once the fast detector trips, LoopMonitor must keep feeding the slow one,
        # or a guard running in warn/stop mode silently loses later detections.
        fast = RepeatDetector(max_repeats=2, window=4)
        slow = RepeatDetector(max_repeats=6, window=8)
        monitor = LoopMonitor([fast, slow])

        for step in range(1, 7):
            monitor.observe("a", step)

        # If the monitor had short-circuited on the first verdict, `slow` would
        # hold a single sighting and this would not fire.
        verdict = slow.observe("a", 7)
        assert verdict is not None
        self.assertEqual(verdict.count, 7)

    def test_empty_monitor_is_falsy(self) -> None:
        self.assertFalse(LoopMonitor([]))
        self.assertTrue(LoopMonitor(default_detectors()))
        self.assertEqual(len(LoopMonitor(default_detectors())), 3)

    def test_reset_clears_every_detector(self) -> None:
        monitor = LoopMonitor([RepeatDetector(max_repeats=2, window=4)])
        monitor.observe("a", 1)
        monitor.reset()
        self.assertIsNone(monitor.observe("a", 2))

    def test_default_progress_detectors(self) -> None:
        detectors = default_progress_detectors()
        self.assertEqual([d.name for d in detectors], ["no-progress"])

    def test_detectors_property_exposes_the_set(self) -> None:
        monitor = LoopMonitor([RepeatDetector()])
        self.assertEqual([d.name for d in monitor.detectors], ["repeat"])


class DefaultRegimeTests(unittest.TestCase):
    """Every shipped detector must own a reachable regime with the default config.

    This is a regression guard on the *defaults*, not on the detectors. Thresholds
    that drift can leave one detector permanently preempted by another: it still
    looks maintained, still has passing unit tests, and never fires in production.
    """

    def test_defaults_catch_an_exact_repeat(self) -> None:
        verdict = feed(LoopMonitor(default_detectors()), ["same()"] * 3)
        assert verdict is not None
        self.assertEqual(verdict.kind, "repeat")

    def test_defaults_report_a_two_step_cycle_as_a_cycle(self) -> None:
        # RepeatDetector would need a third sighting of "read()", i.e. five
        # observations, so a default CycleDetector must fire first at four.
        verdict = feed(LoopMonitor(default_detectors()), ["read()", "write()"] * 2)
        assert verdict is not None
        self.assertEqual(verdict.kind, "cycle")

    def test_defaults_report_paraphrasing_as_similarity(self) -> None:
        query = "how do i fix asyncio event loop is closed in python"
        calls = [call_signature("search", {"query": query + " " * n}) for n in range(4)]
        verdict = feed(LoopMonitor(default_detectors()), calls)
        assert verdict is not None
        self.assertEqual(verdict.kind, "similarity")

    def test_defaults_do_not_trip_on_two_paraphrases(self) -> None:
        query = "how do i fix asyncio event loop is closed in python"
        calls = [call_signature("search", {"query": query + " " * n}) for n in range(3)]
        self.assertIsNone(feed(LoopMonitor(default_detectors()), calls))

    def test_defaults_leave_genuine_work_alone(self) -> None:
        calls = [
            call_signature("search", {"query": "postgres index bloat"}),
            call_signature("read_file", {"path": "schema.sql"}),
            call_signature("run_sql", {"query": "SELECT count(*) FROM orders"}),
            call_signature("write_file", {"path": "report.md", "body": "index bloat"}),
            call_signature("search", {"query": "vacuum full vs reindex"}),
            call_signature("run_sql", {"query": "SELECT pg_size_pretty(1)"}),
        ]
        self.assertIsNone(feed(LoopMonitor(default_detectors()), calls))

    def test_defaults_leave_changing_progress_alone(self) -> None:
        monitor = LoopMonitor(default_progress_detectors())
        for index in range(1, 20):
            self.assertIsNone(monitor.observe(f'{{"rows": {index}}}', index))


class LoopVerdictTests(unittest.TestCase):
    def test_as_dict_round_trips(self) -> None:
        verdict = LoopVerdict(kind="repeat", detail="x", signature="s", count=3, step=7)
        data = verdict.as_dict()
        self.assertEqual(data["kind"], "repeat")
        self.assertEqual(data["step"], 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
