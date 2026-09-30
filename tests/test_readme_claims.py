"""Tests for the promises the README makes about the four detectors.

The README's detector table states, for each detector, the call count it fires on,
and its prompt claims the defaults are tuned so that every detector owns a distinct
reachable regime — its own scenario trips it, and healthy varied work does not. Those
are user-facing claims about tuned numbers, so they are pinned here rather than left
to the prose: a future retune that makes a detector unreachable, or noisy, fails a
test instead of quietly making the documentation wrong.

Each detector is exercised **in isolation**, because with the default three-detector
set a repeat also looks like a cycle candidate and the first verdict returned would
mask which detector actually owns the signal.
"""

from __future__ import annotations

import unittest

from agentguard import (
    CycleDetector,
    Detector,
    Guard,
    LoopMonitor,
    NoProgressDetector,
    RepeatDetector,
    SimilarityDetector,
)
from agentguard.loop import default_detectors, default_progress_detectors

#: One scenario per detector, at the documented threshold. The README's table says:
#: repeat on the 3rd identical call, cycle on the 2nd full pass, similarity on the
#: 4th near-identical call, no-progress on the 6th unchanged marker.
REPEAT_SCENARIO = ['search_web({"query":"weather in oslo"})'] * 3
CYCLE_SCENARIO = [
    'read_file({"path":"app.py"})',
    'write_file({"body":"print(1)"})',
    'read_file({"path":"app.py"})',
    'write_file({"body":"print(1)"})',
]
SIMILARITY_SCENARIO = [
    'search_web({"query":"python asyncio"})',
    'search_web({"query":"python asyncio "})',
    'search_web({"query":"python asyncio  "})',
    'search_web({"query":"python asyncio   "})',
]
NO_PROGRESS_SCENARIO = ['{"rows_written":0}'] * 6

#: Work that is genuinely varied — different tools, different arguments, a progress
#: marker that moves. Nothing here should trip anything.
HEALTHY_ACTIONS = [
    Guard.call_signature("search_web", {"query": "python asyncio tutorial"}),
    Guard.call_signature("read_file", {"path": "src/app.py"}),
    Guard.call_signature("write_file", {"path": "out.txt", "body": "hello"}),
    Guard.call_signature("run_tests", {"suite": "unit"}),
    Guard.call_signature("search_web", {"query": "postgres index bloat"}),
    Guard.call_signature("read_file", {"path": "README.md"}),
    Guard.call_signature("git_commit", {"message": "fix parser"}),
    Guard.call_signature("search_web", {"query": "seasonal flu statistics norway"}),
]
HEALTHY_PROGRESS = [f'{{"rows_written":{(i + 1) * 137}}}' for i in range(8)]


def trip_kind(
    detectors: list[Detector] | None,
    actions: list[str],
    progress: list[str] | None = None,
) -> str:
    """Feed observations to a guard; return the trip kind, or ``'clean'``.

    ``detectors=None`` means the shipping default set — ``LoopMonitor(None)`` builds
    it, whereas an empty list would mean *nothing can trip*, which is the opposite.
    """
    monitor = LoopMonitor(detectors)
    guard = Guard(use_config=False, on_unknown_model="ignore", detectors=monitor.detectors)
    try:
        for signature in actions:
            guard.observe(signature)
        for value in progress or ():
            guard.progress(value)
    except Exception as exc:
        # Deliberately broad: any trip exception is the outcome under test, and the
        # table in the README names a trip *kind*, not an exception class.
        return str(getattr(exc, "kind", type(exc).__name__))
    return "clean"


class DetectorOwnsItsScenarioTests(unittest.TestCase):
    """Each detector fires on its own scenario, alone, with default settings."""

    def test_repeat_detector(self) -> None:
        self.assertEqual(trip_kind([RepeatDetector()], REPEAT_SCENARIO), "repeat")

    def test_cycle_detector(self) -> None:
        self.assertEqual(trip_kind([CycleDetector()], CYCLE_SCENARIO), "cycle")

    def test_similarity_detector(self) -> None:
        self.assertEqual(trip_kind([SimilarityDetector()], SIMILARITY_SCENARIO), "similarity")

    def test_no_progress_detector(self) -> None:
        self.assertEqual(trip_kind([NoProgressDetector()], [], NO_PROGRESS_SCENARIO), "no-progress")

    def test_one_below_the_threshold_does_not_fire(self) -> None:
        # The documented number has to be the *first* one that fires, or the table
        # is off by one for everyone reading it.
        cases = [
            ("repeat", [RepeatDetector()], REPEAT_SCENARIO[:-1]),
            ("cycle", [CycleDetector()], CYCLE_SCENARIO[:-1]),
            ("similarity", [SimilarityDetector()], SIMILARITY_SCENARIO[:-1]),
        ]
        for label, detectors, actions in cases:
            with self.subTest(detector=label):
                self.assertEqual(trip_kind(detectors, actions), "clean")

    def test_five_unchanged_markers_do_not_trip_no_progress(self) -> None:
        self.assertEqual(trip_kind([NoProgressDetector()], [], NO_PROGRESS_SCENARIO[:-1]), "clean")


class DetectorsStaySilentOnHealthyWorkTests(unittest.TestCase):
    """The half that matters more: a detector that cries wolf gets switched off."""

    def test_each_detector_alone_is_clean_on_varied_work(self) -> None:
        for detector in (RepeatDetector(), CycleDetector(), SimilarityDetector()):
            with self.subTest(detector=detector.name):
                self.assertEqual(trip_kind([detector], HEALTHY_ACTIONS), "clean")

    def test_no_progress_is_clean_when_the_marker_moves(self) -> None:
        self.assertEqual(trip_kind([NoProgressDetector()], [], HEALTHY_PROGRESS), "clean")

    def test_the_shipping_default_set_is_clean_on_varied_work(self) -> None:
        self.assertEqual(trip_kind(None, HEALTHY_ACTIONS, HEALTHY_PROGRESS), "clean")

    def test_the_shipping_default_set_still_catches_a_repeat(self) -> None:
        self.assertEqual(trip_kind(None, REPEAT_SCENARIO), "repeat")


class READMEClaimsTests(unittest.TestCase):
    """The README quotes the exact detector names and defaults; keep them honest."""

    def test_the_four_detectors_are_the_documented_four(self) -> None:
        self.assertEqual([d.name for d in default_detectors()], ["repeat", "cycle", "similarity"])
        self.assertEqual([d.name for d in default_progress_detectors()], ["no-progress"])

    def test_documented_defaults(self) -> None:
        # "3rd identical call in a 12-call window", "2nd full pass of a 2-4 step
        # pattern", "4th call >= 95% similar in an 8-call window", "6th unchanged
        # marker" -- as tabulated in README.md and docs/DETAILS.md.
        repeat = RepeatDetector()
        self.assertEqual((repeat.max_repeats, repeat.window), (3, 12))

        cycle = CycleDetector()
        self.assertEqual((cycle.min_cycle, cycle.max_cycle, cycle.repeats), (2, 4, 2))

        similarity = SimilarityDetector()
        self.assertEqual(
            (similarity.max_similar, similarity.window, similarity.threshold), (3, 8, 0.95)
        )

        self.assertEqual(NoProgressDetector().max_stagnant, 6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
