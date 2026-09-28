"""Tests for :class:`agentguard.Report` and its rendering."""

from __future__ import annotations

import json
import os
import unittest

from agentguard import LimitStatus, LoopVerdict, ModelSummary, Report, Usage
from agentguard import report as report_module
from agentguard.report import build_limits


def sample_report(**overrides: object) -> Report:
    base: dict[str, object] = {
        "name": "agent",
        "elapsed_s": 12.4,
        "steps": 8,
        "calls": 3,
        "usage": Usage(input_tokens=150_000, output_tokens=34_000, cached_input_tokens=20_000),
        "cost_usd": 0.4821,
        "limits": (
            LimitStatus("budget", 0.4821, 1.0, "usd"),
            LimitStatus("steps", 8.0, 25.0, "steps"),
        ),
        "by_model": (
            ModelSummary("gpt-4o", 2, 100_000, 20_000, 0.4500, 0),
            ModelSummary("gpt-4o-mini", 1, 50_000, 14_000, 0.0321, 0),
        ),
    }
    base.update(overrides)
    return Report(**base)  # type: ignore[arg-type]


class LimitStatusTests(unittest.TestCase):
    def test_fraction_and_remaining(self) -> None:
        status = LimitStatus("budget", 0.5, 1.0, "usd")
        self.assertAlmostEqual(status.fraction or 0.0, 0.5)
        self.assertAlmostEqual(status.remaining, 0.5)
        self.assertFalse(status.exceeded)

    def test_exceeded(self) -> None:
        status = LimitStatus("budget", 2.0, 1.0, "usd")
        self.assertTrue(status.exceeded)
        self.assertEqual(status.remaining, 0.0)

    def test_zero_limit_has_no_fraction(self) -> None:
        self.assertIsNone(LimitStatus("budget", 1.0, 0.0, "usd").fraction)

    def test_remaining_never_goes_negative(self) -> None:
        self.assertEqual(LimitStatus("steps", 99.0, 10.0, "steps").remaining, 0.0)

    def test_render_ascii(self) -> None:
        line = LimitStatus("budget", 0.5, 1.0, "usd").render(ascii_only=True)
        self.assertIn("budget", line)
        self.assertIn("$0.5 / $1", line)
        self.assertIn("50.0%", line)
        self.assertIn("#", line)
        self.assertIn(".", line)

    def test_render_unicode_uses_block_characters(self) -> None:
        line = LimitStatus("budget", 0.5, 1.0, "usd").render(ascii_only=False)
        self.assertIn("█", line)
        self.assertIn("░", line)

    def test_render_marks_an_exceeded_limit(self) -> None:
        self.assertTrue(
            LimitStatus("budget", 2.0, 1.0, "usd").render(ascii_only=True).startswith("!")
        )

    def test_unit_formatting(self) -> None:
        self.assertIn(
            "1,000", LimitStatus("tokens", 1000.0, 2000.0, "tokens").render(ascii_only=True)
        )
        self.assertIn(
            "12.4s", LimitStatus("time", 12.42, 60.0, "seconds").render(ascii_only=True)
        )
        self.assertIn(
            " / 25", LimitStatus("steps", 8.0, 25.0, "steps").render(ascii_only=True)
        )

    def test_as_dict(self) -> None:
        data = LimitStatus("budget", 0.5, 1.0, "usd").as_dict()
        self.assertEqual(data["name"], "budget")
        self.assertAlmostEqual(data["fraction"], 0.5)


class BuildLimitsTests(unittest.TestCase):
    def test_only_configured_limits_are_included(self) -> None:
        limits = build_limits(
            cost_usd=0.5,
            max_usd=1.0,
            total_tokens=100,
            max_tokens=None,
            steps=3,
            max_steps=None,
            elapsed_s=1.0,
            max_seconds=60.0,
        )
        self.assertEqual([limit.name for limit in limits], ["budget", "time"])

    def test_no_limits_configured(self) -> None:
        limits = build_limits(
            cost_usd=0.0,
            max_usd=None,
            total_tokens=0,
            max_tokens=None,
            steps=0,
            max_steps=None,
            elapsed_s=0.0,
            max_seconds=None,
        )
        self.assertEqual(limits, ())


class SupportsUnicodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = report_module._unicode_cache
        report_module._unicode_cache = None

    def tearDown(self) -> None:
        report_module._unicode_cache = self._saved
        os.environ.pop("AGENTGUARD_ASCII", None)

    def test_env_forces_ascii(self) -> None:
        os.environ["AGENTGUARD_ASCII"] = "1"
        self.assertFalse(report_module.supports_unicode())

    def test_env_forces_unicode(self) -> None:
        os.environ["AGENTGUARD_ASCII"] = "0"
        self.assertTrue(report_module.supports_unicode())

    def test_env_accepts_word_forms(self) -> None:
        os.environ["AGENTGUARD_ASCII"] = "true"
        self.assertFalse(report_module.supports_unicode())


class RenderTests(unittest.TestCase):
    def test_renders_the_headline_numbers(self) -> None:
        text = sample_report().render(ascii_only=True)
        self.assertIn("agentguard  agent", text)
        self.assertIn("12.4s", text)
        self.assertIn("184,000", text)  # 150k + 34k
        self.assertIn("cached", text)

    def test_renders_the_limit_bars(self) -> None:
        text = sample_report().render(ascii_only=True)
        self.assertIn("limits", text)
        self.assertIn("budget", text)
        self.assertIn("steps", text)

    def test_the_header_uses_the_installable_name(self) -> None:
        # `agent-guard` is the repository and brand name, but it is also somebody
        # else's PyPI package. Output must only ever say `agentguard`.
        text = sample_report().render(ascii_only=True)
        self.assertIn("agentguard", text)
        self.assertNotIn("agent-guard", text)

    def test_renders_the_per_model_breakdown(self) -> None:
        text = sample_report().render(ascii_only=True)
        self.assertIn("by model", text)
        self.assertIn("gpt-4o", text)
        self.assertIn("gpt-4o-mini", text)
        self.assertIn("2 calls", text)
        self.assertIn("1 call", text)

    def test_renders_the_unpriced_warning(self) -> None:
        text = sample_report(unpriced_models=("mystery",), unpriced_calls=2).render(
            ascii_only=True
        )
        self.assertIn("mystery", text)
        self.assertIn("excluded from the budget", text)

    def test_unpriced_models_show_as_unpriced_in_the_table(self) -> None:
        report = sample_report(
            by_model=(ModelSummary("mystery", 1, 10, 10, 0.0, 1),),
            unpriced_models=("mystery",),
            unpriced_calls=1,
        )
        self.assertIn("unpriced", report.render(ascii_only=True))

    def test_renders_a_loop_trip(self) -> None:
        report = sample_report(
            trip=LoopVerdict(kind="repeat", detail="same call 3x", signature="s", count=3, step=5)
        )
        text = report.render(ascii_only=True)
        self.assertIn("tripped: loop [repeat]", text)
        self.assertIn("same call 3x", text)

    def test_renders_a_non_loop_trip(self) -> None:
        text = sample_report(tripped_reason="budget").render(ascii_only=True)
        self.assertIn("tripped: budget", text)

    def test_always_mentions_that_prices_are_indicative(self) -> None:
        self.assertIn("indicative", sample_report().render(ascii_only=True))

    def test_unicode_and_ascii_differ(self) -> None:
        report = sample_report()
        self.assertNotEqual(report.render(ascii_only=True), report.render(ascii_only=False))

    def test_str_matches_render(self) -> None:
        report = sample_report()
        self.assertEqual(str(report), report.render())

    def test_render_is_deterministic(self) -> None:
        report = sample_report()
        self.assertEqual(report.render(ascii_only=True), report.render(ascii_only=True))


class SerialisationTests(unittest.TestCase):
    def test_as_dict_is_flat_and_json_friendly(self) -> None:
        data = sample_report().as_dict()
        self.assertEqual(data["name"], "agent")
        self.assertEqual(data["usage"]["total_tokens"], 184_000)
        self.assertEqual(len(data["limits"]), 2)
        json.dumps(data)  # must not raise

    def test_round_trip_preserves_every_field(self) -> None:
        original = sample_report(
            trip=LoopVerdict("cycle", "a -> b", "sig", 3, 9),
            unpriced_models=("mystery",),
            unpriced_calls=1,
        )
        restored = Report.from_dict(original.as_dict())

        self.assertEqual(restored.name, original.name)
        self.assertAlmostEqual(restored.elapsed_s, original.elapsed_s)
        self.assertEqual(restored.steps, original.steps)
        self.assertEqual(restored.calls, original.calls)
        self.assertAlmostEqual(restored.cost_usd, original.cost_usd)
        self.assertEqual(restored.usage.total_tokens, original.usage.total_tokens)
        self.assertEqual(restored.usage.cached_input_tokens, 20_000)
        self.assertEqual([s.name for s in restored.limits], ["budget", "steps"])
        self.assertEqual([s.model for s in restored.by_model], ["gpt-4o", "gpt-4o-mini"])
        self.assertEqual(restored.unpriced_models, ("mystery",))
        assert restored.trip is not None
        self.assertEqual(restored.trip.kind, "cycle")
        self.assertEqual(restored.trip.step, 9)
        self.assertEqual(restored.render(ascii_only=True), original.render(ascii_only=True))

    def test_from_dict_tolerates_a_minimal_payload(self) -> None:
        report = Report.from_dict({"calls": 0})
        self.assertEqual(report.calls, 0)
        self.assertIsNone(report.trip)
        self.assertEqual(report.limits, ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
