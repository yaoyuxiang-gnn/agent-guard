"""Tests for the price table and cost arithmetic.

Pricing is the one place where a bug means money: either a budget that never
trips, or a false alarm that kills a healthy agent. These tests pin down both the
arithmetic and the conservatism of model-name resolution.
"""

from __future__ import annotations

import unittest

from agent_guard import DEFAULT_PRICING, GuardConfigError, Price, PriceTable
from agent_guard.pricing import normalize_model_key


class NormalizeModelKeyTests(unittest.TestCase):
    def test_lowercases_and_strips_whitespace(self) -> None:
        self.assertEqual(normalize_model_key("  GPT-4o  "), "gpt-4o")

    def test_strips_gateway_namespace(self) -> None:
        self.assertEqual(normalize_model_key("openai/gpt-4o"), "gpt-4o")
        self.assertEqual(
            normalize_model_key("meta-llama/Llama-3.3-70B-Instruct"),
            "llama-3.3-70b-instruct",
        )

    def test_strips_variant_tag(self) -> None:
        self.assertEqual(normalize_model_key("deepseek/deepseek-chat:free"), "deepseek-chat")

    def test_strips_version_pin(self) -> None:
        self.assertEqual(normalize_model_key("claude-sonnet-4@20250514"), "claude-sonnet-4")

    def test_strips_stacked_bedrock_prefixes(self) -> None:
        self.assertEqual(
            normalize_model_key("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "claude-3-5-sonnet-20241022-v2",
        )

    def test_does_not_mangle_models_with_dots(self) -> None:
        # "gpt-3.5-turbo" must survive: a naive ``rsplit('.')`` would destroy it.
        self.assertEqual(normalize_model_key("gpt-3.5-turbo"), "gpt-3.5-turbo")

    def test_empty_input(self) -> None:
        self.assertEqual(normalize_model_key(""), "")
        self.assertEqual(normalize_model_key("   "), "")


class PriceArithmeticTests(unittest.TestCase):
    def test_input_only(self) -> None:
        price = Price(input_per_1m=2.50, output_per_1m=10.00)
        self.assertAlmostEqual(price.cost_usd(input_tokens=1_000_000), 2.50)

    def test_input_and_output(self) -> None:
        price = Price(input_per_1m=2.50, output_per_1m=10.00)
        self.assertAlmostEqual(price.cost_usd(input_tokens=1_000, output_tokens=500), 0.0075)

    def test_cached_input_is_billed_at_the_discount(self) -> None:
        price = Price(2.50, 10.00, cached_input_per_1m=1.25)
        # 600k fresh @ 2.50 + 400k cached @ 1.25 == 1.50 + 0.50
        self.assertAlmostEqual(
            price.cost_usd(input_tokens=1_000_000, cached_input_tokens=400_000), 2.00
        )

    def test_cached_input_is_clamped_to_input(self) -> None:
        price = Price(2.50, 10.00, cached_input_per_1m=1.25)
        self.assertAlmostEqual(
            price.cost_usd(input_tokens=100, cached_input_tokens=999_999),
            100 * 1.25 / 1_000_000,
        )

    def test_cached_input_without_a_cached_rate_bills_at_full_rate(self) -> None:
        price = Price(2.50, 10.00)
        self.assertAlmostEqual(
            price.cost_usd(input_tokens=1_000_000, cached_input_tokens=1_000_000), 2.50
        )

    def test_negative_and_zero_counts_are_safe(self) -> None:
        price = Price(2.50, 10.00)
        self.assertEqual(price.cost_usd(input_tokens=-5, output_tokens=-5), 0.0)
        self.assertEqual(price.cost_usd(), 0.0)

    def test_worst_case_bounds_the_output(self) -> None:
        price = Price(2.50, 10.00)
        self.assertAlmostEqual(
            price.worst_case_usd(input_tokens=1_000, max_output_tokens=2_000), 0.0225
        )

    def test_price_is_hashable_and_frozen(self) -> None:
        price = Price(1.0, 2.0)
        self.assertEqual(hash(price), hash(Price(1.0, 2.0)))
        with self.assertRaises(Exception):
            price.input_per_1m = 5.0  # type: ignore[misc]


class ModelResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = PriceTable()

    def test_exact_lookup(self) -> None:
        resolved = self.table.resolve("gpt-4o")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o")
        self.assertEqual(resolved[1].input_per_1m, 2.50)

    def test_dated_variant_falls_back_to_the_family(self) -> None:
        resolved = self.table.resolve("gpt-4o-2024-08-06")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o")

    def test_anthropic_dated_variant_falls_back(self) -> None:
        resolved = self.table.resolve("claude-3-5-sonnet-20241022")
        assert resolved is not None
        self.assertEqual(resolved[0], "claude-3-5-sonnet")

    def test_numeric_legacy_suffix_falls_back(self) -> None:
        resolved = self.table.resolve("gpt-4-0613")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4")

    def test_distinct_families_are_never_conflated(self) -> None:
        # "gpt-4-turbo" exists in the table, but even if it did not it must not
        # silently become "gpt-4" at a tenth of the price.
        self.assertIsNone(self.table.resolve("gpt-4-something-weird"))
        self.assertIsNone(self.table.resolve("totally-made-up-model"))

    def test_get_never_falls_back_but_resolve_does(self) -> None:
        self.assertIsNone(self.table.get("gpt-4o-2024-08-06"))
        self.assertIsNotNone(self.table.resolve("gpt-4o-2024-08-06"))

    def test_contains_uses_resolution(self) -> None:
        self.assertIn("gpt-4o-2024-08-06", self.table)
        self.assertNotIn("definitely-not-a-model", self.table)

    def test_gateway_prefixed_name_resolves(self) -> None:
        resolved = self.table.resolve("openai/gpt-4o-mini")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o-mini")

    def test_empty_name_is_not_resolved(self) -> None:
        self.assertIsNone(self.table.resolve(""))
        self.assertIsNone(self.table.resolve_price(""))


class PriceTableOverrideTests(unittest.TestCase):
    def test_price_instance_override(self) -> None:
        table = PriceTable(overrides={"mine": Price(1.0, 2.0)})
        self.assertEqual(table.resolve_price("mine").output_per_1m, 2.0)

    def test_two_tuple_override(self) -> None:
        table = PriceTable(overrides={"mine": (1.0, 2.0)})
        price = table.resolve_price("mine")
        assert price is not None
        self.assertEqual((price.input_per_1m, price.output_per_1m), (1.0, 2.0))
        self.assertIsNone(price.cached_input_per_1m)

    def test_three_tuple_override(self) -> None:
        table = PriceTable(overrides={"mine": (1.0, 2.0, 0.5)})
        price = table.resolve_price("mine")
        assert price is not None
        self.assertEqual(price.cached_input_per_1m, 0.5)

    def test_override_keys_are_normalized(self) -> None:
        table = PriceTable(overrides={"OpenAI/My-Model": (1.0, 2.0)})
        self.assertIsNotNone(table.resolve_price("my-model"))

    def test_override_wins_over_bundled_price(self) -> None:
        table = PriceTable(overrides={"gpt-4o": (99.0, 99.0)})
        self.assertEqual(table.resolve_price("gpt-4o").input_per_1m, 99.0)

    def test_bad_override_shapes_raise(self) -> None:
        for bad in ({"m": (1.0,)}, {"m": (1.0, 2.0, 3.0, 4.0)}, {"m": "cheap"}, {"m": 1.0}):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                PriceTable(overrides=bad)  # type: ignore[arg-type]

    def test_non_numeric_override_raises(self) -> None:
        with self.assertRaises(GuardConfigError):
            PriceTable(overrides={"m": ("a", "b")})  # type: ignore[dict-item]

    def test_empty_override_name_raises(self) -> None:
        with self.assertRaises(GuardConfigError):
            PriceTable(overrides={"": (1.0, 2.0)})

    def test_custom_base_replaces_the_bundled_table(self) -> None:
        table = PriceTable(base={"only": Price(1.0, 1.0)})
        self.assertEqual(len(table), 1)
        self.assertIsNone(table.resolve_price("gpt-4o"))

    def test_len_and_iteration(self) -> None:
        table = PriceTable()
        self.assertEqual(len(table), len(DEFAULT_PRICING))
        self.assertEqual(len(list(iter(table))), len(DEFAULT_PRICING))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
