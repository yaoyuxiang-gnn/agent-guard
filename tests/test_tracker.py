"""Tests for usage extraction and cost accounting."""

from __future__ import annotations

import threading
import unittest
import warnings
from types import SimpleNamespace as NS

from agentguard import GuardConfigError, Price, PriceTable, Usage
from agentguard.tracker import CostTracker, extract_model, extract_usage


def openai_response(
    model: str = "gpt-4o",
    prompt: int = 100,
    completion: int = 50,
    cached: int = 0,
    reasoning: int = 0,
) -> NS:
    return NS(
        model=model,
        usage=NS(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=NS(cached_tokens=cached),
            completion_tokens_details=NS(reasoning_tokens=reasoning),
        ),
    )


def anthropic_response(
    model: str = "claude-sonnet-4",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
) -> NS:
    return NS(
        model=model,
        usage=NS(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


class ExtractUsageTests(unittest.TestCase):
    def test_openai_shape(self) -> None:
        usage = extract_usage(openai_response(prompt=120, completion=30))
        assert usage is not None
        self.assertEqual((usage.input_tokens, usage.output_tokens), (120, 30))

    def test_anthropic_shape(self) -> None:
        usage = extract_usage(anthropic_response(input_tokens=200, output_tokens=80))
        assert usage is not None
        self.assertEqual((usage.input_tokens, usage.output_tokens), (200, 80))

    def test_gemini_shape_via_usage_metadata(self) -> None:
        response = NS(usage_metadata=NS(prompt_token_count=11, candidates_token_count=7))
        usage = extract_usage(response)
        assert usage is not None
        self.assertEqual((usage.input_tokens, usage.output_tokens), (11, 7))

    def test_plain_dict_payload(self) -> None:
        usage = extract_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 6}})
        assert usage is not None
        self.assertEqual(usage.total_tokens, 11)

    def test_nested_cached_and_reasoning_details(self) -> None:
        usage = extract_usage(openai_response(cached=40, reasoning=25))
        assert usage is not None
        self.assertEqual(usage.cached_input_tokens, 40)
        self.assertEqual(usage.reasoning_tokens, 25)

    def test_anthropic_cache_read_is_a_discount(self) -> None:
        usage = extract_usage(anthropic_response(cache_read=90))
        assert usage is not None
        self.assertEqual(usage.cached_input_tokens, 90)

    def test_no_usage_returns_none(self) -> None:
        self.assertIsNone(extract_usage(NS(model="gpt-4o")))
        self.assertIsNone(extract_usage({}))
        self.assertIsNone(extract_usage(None))

    def test_all_zero_usage_returns_none(self) -> None:
        # Distinguishes "provider reported nothing" from "provider reported zero".
        self.assertIsNone(extract_usage(openai_response(prompt=0, completion=0)))

    def test_negative_counts_are_clamped(self) -> None:
        usage = extract_usage({"usage": {"prompt_tokens": -10, "completion_tokens": -5}})
        assert usage is not None
        self.assertEqual(usage.total_tokens, 0)

    def test_booleans_are_not_mistaken_for_counts(self) -> None:
        self.assertIsNone(extract_usage({"usage": {"prompt_tokens": True}}))


class ExtractModelTests(unittest.TestCase):
    def test_reads_model_attribute(self) -> None:
        self.assertEqual(extract_model(NS(model="gpt-4o")), "gpt-4o")

    def test_reads_model_from_dict(self) -> None:
        self.assertEqual(extract_model({"model": "gpt-4o-mini"}), "gpt-4o-mini")

    def test_falls_back_to_default(self) -> None:
        self.assertEqual(extract_model({}, default="fallback"), "fallback")
        self.assertIsNone(extract_model({}))


class UsageTests(unittest.TestCase):
    def test_total_and_add(self) -> None:
        total = Usage(input_tokens=10, output_tokens=5) + Usage(
            input_tokens=1, output_tokens=2, cached_input_tokens=3, reasoning_tokens=4
        )
        self.assertEqual(total.input_tokens, 11)
        self.assertEqual(total.output_tokens, 7)
        self.assertEqual(total.cached_input_tokens, 3)
        self.assertEqual(total.reasoning_tokens, 4)

    def test_is_empty(self) -> None:
        self.assertTrue(Usage().is_empty)
        self.assertFalse(Usage(input_tokens=1).is_empty)

    def test_as_dict(self) -> None:
        data = Usage(input_tokens=2, output_tokens=3).as_dict()
        self.assertEqual(data["total_tokens"], 5)


class CostTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = CostTracker(on_unknown_model="ignore")

    def test_priced_record(self) -> None:
        record = self.tracker.record(
            model="gpt-4o", usage=Usage(1_000_000, 0), elapsed_s=0.0
        )
        self.assertTrue(record.priced)
        self.assertEqual(record.canonical_model, "gpt-4o")
        self.assertAlmostEqual(self.tracker.total_usd, 2.50)

    def test_unpriced_record_is_counted_not_guessed(self) -> None:
        record = self.tracker.record(model="mystery", usage=Usage(1000, 1000), elapsed_s=0.0)
        self.assertFalse(record.priced)
        self.assertIsNone(record.cost_usd)
        self.assertEqual(self.tracker.unpriced_calls, 1)
        self.assertEqual(self.tracker.unpriced_models, ("mystery",))
        self.assertEqual(self.tracker.total_usd, 0.0)

    def test_explicit_price_argument_wins(self) -> None:
        record = self.tracker.record(
            model="anything",
            usage=Usage(1_000_000, 0),
            elapsed_s=0.0,
            price=Price(7.0, 7.0),
        )
        self.assertAlmostEqual(record.cost_usd or 0.0, 7.0)

    def test_default_price_covers_unknown_models(self) -> None:
        tracker = CostTracker(
            on_unknown_model="ignore", default_price=Price(1.0, 1.0)
        )
        record = tracker.record(model="whatever", usage=Usage(1_000_000, 0), elapsed_s=0.0)
        self.assertTrue(record.priced)
        self.assertAlmostEqual(tracker.total_usd, 1.0)

    def test_custom_table_is_used(self) -> None:
        tracker = CostTracker(
            PriceTable(base={"only": Price(3.0, 3.0)}), on_unknown_model="ignore"
        )
        tracker.record(model="only", usage=Usage(1_000_000, 0), elapsed_s=0.0)
        self.assertAlmostEqual(tracker.total_usd, 3.0)

    def test_unknown_model_warns_once_per_model(self) -> None:
        tracker = CostTracker()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                tracker.record(model="mystery", usage=Usage(1, 1), elapsed_s=0.0)
        self.assertEqual(len(caught), 1)
        self.assertIn("mystery", str(caught[0].message))
        self.assertEqual(tracker.unpriced_calls, 3)

    def test_unknown_model_can_raise(self) -> None:
        tracker = CostTracker(on_unknown_model="error")
        with self.assertRaises(GuardConfigError):
            tracker.record(model="mystery", usage=Usage(1, 1), elapsed_s=0.0)

    def test_invalid_unknown_model_mode(self) -> None:
        with self.assertRaises(ValueError):
            CostTracker(on_unknown_model="shrug")

    def test_token_totals(self) -> None:
        self.tracker.record(model="gpt-4o", usage=Usage(100, 20), elapsed_s=0.0)
        self.tracker.record(model="gpt-4o", usage=Usage(300, 40), elapsed_s=0.0)
        self.assertEqual(self.tracker.input_tokens, 400)
        self.assertEqual(self.tracker.output_tokens, 60)
        self.assertEqual(self.tracker.total_tokens, 460)
        self.assertEqual(self.tracker.calls, 2)

    def test_records_are_immutable_snapshots(self) -> None:
        self.tracker.record(model="gpt-4o", usage=Usage(1, 1), elapsed_s=0.0)
        snapshot = self.tracker.records
        self.tracker.record(model="gpt-4o", usage=Usage(1, 1), elapsed_s=0.0)
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(self.tracker.records), 2)

    def test_by_model_is_ordered_by_descending_cost(self) -> None:
        self.tracker.record(model="gpt-4o", usage=Usage(100_000, 0), elapsed_s=0.0)
        self.tracker.record(model="gpt-4o-mini", usage=Usage(100_000, 0), elapsed_s=0.0)
        self.tracker.record(model="gpt-4o", usage=Usage(100_000, 0), elapsed_s=0.0)
        summaries = list(self.tracker.by_model().values())
        self.assertEqual([s.model for s in summaries], ["gpt-4o", "gpt-4o-mini"])
        self.assertEqual(summaries[0].calls, 2)
        self.assertEqual(summaries[1].calls, 1)

    def test_by_model_groups_aliases_under_the_canonical_name(self) -> None:
        self.tracker.record(model="gpt-4o-2024-08-06", usage=Usage(10, 10), elapsed_s=0.0)
        self.tracker.record(model="gpt-4o", usage=Usage(10, 10), elapsed_s=0.0)
        self.assertEqual(list(self.tracker.by_model()), ["gpt-4o"])

    def test_burn_rate(self) -> None:
        self.assertIsNone(self.tracker.burn_rate_usd_per_step())
        self.tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), elapsed_s=0.0)
        self.tracker.record(model="gpt-4o", usage=Usage(1_000_000, 0), elapsed_s=0.0)
        self.assertAlmostEqual(self.tracker.burn_rate_usd_per_step() or 0.0, 2.50)

    def test_as_dict_shape(self) -> None:
        self.tracker.record(model="gpt-4o", usage=Usage(10, 5), elapsed_s=0.0)
        data = self.tracker.as_dict()
        self.assertEqual(data["calls"], 1)
        self.assertEqual(data["usage"]["total_tokens"], 15)
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(len(data["by_model"]), 1)

    def test_price_table_property(self) -> None:
        self.assertIsInstance(self.tracker.price_table, PriceTable)

    def test_concurrent_recording_is_lossless(self) -> None:
        tracker = CostTracker(on_unknown_model="ignore")

        def work() -> None:
            for _ in range(200):
                tracker.record(model="gpt-4o", usage=Usage(1_000, 0), elapsed_s=0.0)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(tracker.calls, 1600)
        self.assertAlmostEqual(tracker.total_usd, 1600 * 1_000 * 2.5 / 1_000_000, places=9)
        self.assertEqual([r.index for r in tracker.records], list(range(1600)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
