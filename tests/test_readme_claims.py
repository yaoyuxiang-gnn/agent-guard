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
from pathlib import Path

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


class DocumentedNumbersTests(unittest.TestCase):
    """Numbers quoted in the docs that nothing else would catch going stale.

    ``tests/test_api_reference.py`` checks names and signatures, and this module
    checks the detector table, but the prose carries a third kind of claim: counts
    and limits a reader takes at face value. A renamed constant or a retuned bound
    used to leave them silently wrong — ``docs/DETAILS.md`` advertised 551 tests
    while the suite ran 588.
    """

    def test_loop_constants_match_what_the_reference_publishes(self) -> None:
        # docs/API.md: "Signatures are truncated to 512 characters" and
        # "SimilarityDetector(threshold=0.95, window=8, max_similar=3,
        # compare_chars=512)" were the pre-fix values.
        from agentguard.loop import _DEFAULT_COMPARE_CHARS, _MAX_SIGNATURE_LEN

        reference = (Path(__file__).resolve().parent.parent / "docs" / "API.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            f"call_signature(name, args=None, *, max_len={_MAX_SIGNATURE_LEN})", reference
        )
        self.assertIn(f"compare_chars={_DEFAULT_COMPARE_CHARS}", reference)

    def test_a_documented_signature_stays_within_its_documented_bound(self) -> None:
        from agentguard.loop import _MAX_SIGNATURE_LEN

        long_signature = Guard.call_signature("tool", {"blob": "x" * 100_000})
        self.assertLessEqual(len(long_signature), _MAX_SIGNATURE_LEN + 60)

    def test_the_readme_quickstart_claims_the_guard_it_describes(self) -> None:
        # The README's opening sample: a $1.00 cap, 25 steps, and three things
        # happening "on their own".
        guard = Guard(max_usd=1.00, max_steps=25)
        self.assertEqual(guard.remaining_usd, 1.00)
        with guard.step(tag="search") as step:
            step.record("gpt-4o", input_tokens=8_000, output_tokens=400)
        self.assertEqual(guard.steps, 1)
        self.assertEqual(guard.calls, 1)
        self.assertGreater(guard.spent_usd, 0.0)

    def test_the_report_prints_the_limits_the_readme_tabulates(self) -> None:
        # README's limit table, one row per documented name.
        guard = Guard(
            max_usd=1.00, max_tokens=500_000, max_steps=25, max_seconds=300, use_config=False
        )
        self.assertEqual(
            [status.name for status in guard.report().limits],
            ["budget", "tokens", "steps", "time"],
        )

    def test_the_documented_python_floor_is_still_declared(self) -> None:
        # README: "Python 3.10+". pyproject.toml is the declaration; this fails if
        # the two ever disagree.
        pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('requires-python = ">=3.10"', pyproject)

    def test_every_example_is_named_in_the_readme_and_run_in_ci(self) -> None:
        # CONTRIBUTING: "An example nothing runs is documentation that has already
        # drifted." The README's count and the files on disk are the same claim from
        # two directions, so each is checked against the list CI actually runs.
        root = Path(__file__).resolve().parent.parent
        on_disk = sorted(path.name for path in (root / "examples").glob("*.py"))
        self.assertTrue(on_disk, "no examples found")

        ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        makefile = (root / "Makefile").read_text(encoding="utf-8")
        for name in on_disk:
            with self.subTest(example=name):
                self.assertIn(f"examples/{name}", ci)
                self.assertIn(f"examples/{name}", makefile)

        numbers = {8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"{numbers[len(on_disk)]} runnable programs", readme)

    def test_every_documented_guard_parameter_is_current(self) -> None:
        # docs/API.md lists Guard's arguments in prose, and the list changes whenever
        # Guard gains a knob. A reader who cannot see `scoped_budgets` there has no
        # way to discover it exists.
        import inspect

        root = Path(__file__).resolve().parent.parent
        reference = (root / "docs" / "API.md").read_text(encoding="utf-8")
        parameters = [
            name for name in inspect.signature(Guard.__init__).parameters if name != "self"
        ]
        missing = [name for name in parameters if f"{name}=" not in reference]
        self.assertEqual(missing, [], f"docs/API.md never names these Guard parameters: {missing}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
