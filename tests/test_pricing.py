"""Tests for the price table and cost arithmetic.

Pricing is the one place where a bug means money: either a budget that never
trips, or a false alarm that kills a healthy agent. These tests pin down both the
arithmetic and the conservatism of model-name resolution.
"""

from __future__ import annotations

import dataclasses
import unittest

from agentguard import DEFAULT_PRICING, GuardConfigError, Price, PriceTable
from agentguard.pricing import normalize_model_key


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
        # Named rather than a blind `Exception`: if the dataclass ever stops being
        # frozen, this test must fail for that reason and not for some other one.
        with self.assertRaises(dataclasses.FrozenInstanceError):
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


class PriceValidationTests(unittest.TestCase):
    """A price is money: an impossible one must fail where it was written.

    A ``NaN`` rate is the dangerous case. ``NaN > budget`` is false, so a single
    bad number would turn every budget check into a no-op — a guard that silently
    stops guarding.
    """

    def test_negative_rates_are_rejected(self) -> None:
        for bad in (-0.01, -1.0):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Price(bad, 1.0)
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Price(1.0, bad)
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Price(1.0, 1.0, bad)

    def test_non_finite_rates_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Price(bad, 1.0)

    def test_booleans_and_strings_are_not_rates(self) -> None:
        for bad in (True, "2.50", None):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                Price(bad, 1.0)  # type: ignore[arg-type]

    def test_zero_is_a_legitimate_price(self) -> None:
        self.assertEqual(Price(0.0, 0.0).cost_usd(input_tokens=1_000_000), 0.0)

    def test_integers_are_coerced_to_floats(self) -> None:
        price = Price(3, 12)
        self.assertIsInstance(price.input_per_1m, float)
        self.assertEqual(price, Price(3.0, 12.0))

    def test_as_dict_matches_the_config_key_names(self) -> None:
        self.assertEqual(
            Price(1.0, 2.0, 0.5).as_dict(),
            {"input_per_1m": 1.0, "output_per_1m": 2.0, "cached_input_per_1m": 0.5},
        )
        self.assertIsNone(Price(1.0, 2.0).as_dict()["cached_input_per_1m"])


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

    def test_mapping_override(self) -> None:
        table = PriceTable(overrides={"mine": {"input": 1.0, "output": 2.0, "cached": 0.5}})
        self.assertEqual(table.resolve_price("mine"), Price(1.0, 2.0, 0.5))

    def test_mapping_override_accepts_the_field_names_from_repr(self) -> None:
        table = PriceTable(overrides={"mine": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
        self.assertEqual(table.resolve_price("mine"), Price(1.0, 2.0))

    def test_mapping_override_needs_both_directions(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            PriceTable(overrides={"mine": {"input": 1.0}})
        self.assertIn("output", str(caught.exception))

    def test_mapping_override_rejects_unknown_and_duplicate_keys(self) -> None:
        for bad in ({"in": 1.0, "output": 2.0}, {"input": 1, "input_per_1m": 1, "output": 2}):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                PriceTable(overrides={"mine": bad})

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

    def test_impossible_override_rates_raise(self) -> None:
        for bad in ((-1.0, 2.0), (float("nan"), 2.0), (1.0, float("inf"))):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                PriceTable(overrides={"m": bad})

    def test_empty_override_name_raises(self) -> None:
        with self.assertRaises(GuardConfigError):
            PriceTable(overrides={"": (1.0, 2.0)})


class BundledPriceTableTests(unittest.TestCase):
    """The bundled table is data, and data rots. These are the checks a refresh
    has to keep passing — a wrong rate here is a budget that never fires."""

    def test_every_entry_is_reachable_by_its_own_name(self) -> None:
        table = PriceTable()
        for name in DEFAULT_PRICING:
            with self.subTest(model=name):
                self.assertIsNotNone(
                    table.resolve(name),
                    f"{name!r} is in the bundled table but does not resolve to itself",
                )

    def test_no_two_names_collapse_onto_one_key(self) -> None:
        # Normalization is aggressive, so two keys that differ only in punctuation
        # would silently shadow each other and one price would be unreachable.
        keys = [normalize_model_key(name) for name in DEFAULT_PRICING]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        self.assertEqual(duplicates, [], f"these names collide after normalization: {duplicates}")

    def test_cached_input_is_never_dearer_than_fresh_input(self) -> None:
        # A cached-input rate above the input rate would bill a cache hit as a
        # penalty, which no provider does; it means a typo in the table.
        for name, price in DEFAULT_PRICING.items():
            if price.cached_input_per_1m is None:
                continue
            with self.subTest(model=name):
                self.assertLessEqual(price.cached_input_per_1m, price.input_per_1m)

    def test_output_is_never_cheaper_than_input(self) -> None:
        # Holds for every model priced here and would catch a swapped pair.
        for name, price in DEFAULT_PRICING.items():
            with self.subTest(model=name):
                self.assertGreaterEqual(price.output_per_1m, price.input_per_1m)

    def test_no_price_is_zero(self) -> None:
        # A zero rate is how a "free" model would be entered, but it is also what a
        # missing column looks like. Real free tiers are reported as unpriced.
        for name, price in DEFAULT_PRICING.items():
            with self.subTest(model=name):
                self.assertGreater(price.input_per_1m, 0.0)

    def test_expected_model_families_are_present(self) -> None:
        # A refresh that drops a family silently turns every call to it unpriced,
        # which is the failure this whole table exists to prevent.
        table = PriceTable()
        for model in (
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-5",
            "gpt-5-mini",
            "o3",
            "o4-mini",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-sonnet-4",
            "claude-opus-4",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "deepseek-chat",
            "deepseek-reasoner",
            "mistral-large",
            "codestral",
            "grok-3",
            "grok-4",
            "llama-3.3-70b",
            "qwen-max",
        ):
            with self.subTest(model=model):
                self.assertIsNotNone(table.resolve_price(model), f"{model} lost its price")

    def test_current_flagships_are_priced(self) -> None:
        # The refresh that motivated this class: a model newer than the snapshot is
        # reported unpriced and excluded from the budget, so the cap a user
        # configured never fires. One current model per provider, at minimum.
        table = PriceTable()
        for model in (
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-5.5",
            "gpt-5.1",
            "claude-opus-5",
            "claude-opus-5.5",
            "claude-sonnet-5",
            "claude-haiku-4.5",
            "claude-fable-5.1",
            "gemini-3.8-flash",
            "gemini-3.1-pro",
            "grok-4.7",
            "deepseek-v4-pro",
            "mistral-medium-3.5",
            "qwen3.8-max",
        ):
            with self.subTest(model=model):
                self.assertIsNotNone(table.resolve_price(model), f"{model} has no bundled price")

    def test_a_dated_variant_resolves_to_its_family_price(self) -> None:
        table = PriceTable()
        # The shapes providers actually report, including the dash-dated ones.
        for reported, expected in (
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-5.1-20260301", "gpt-5.1"),
            ("claude-opus-5-20260601", "claude-opus-5"),
            ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
            ("gemini-3.1-pro-preview", "gemini-3.1-pro"),
        ):
            with self.subTest(reported=reported):
                resolved = table.resolve(reported)
                self.assertIsNotNone(resolved, f"{reported} resolved to nothing")
                assert resolved is not None
                self.assertEqual(resolved[0], expected)

    def test_both_spellings_of_a_dotted_version_are_priced(self) -> None:
        # Anthropic writes "claude-haiku-4-5" and the rate catalogue writes
        # "claude-haiku-4.5"; either spelling must find the same rate rather than
        # one of them silently counting as unpriced.
        table = PriceTable()
        for dotted, dashed in (
            ("claude-haiku-4.5", "claude-haiku-4-5"),
            ("claude-opus-5.5", "claude-opus-5-5"),
            ("claude-sonnet-5.5", "claude-sonnet-5-5"),
            ("claude-fable-5.1", "claude-fable-5-1"),
            ("gemini-3.1-pro", "gemini-3-1-pro"),
        ):
            with self.subTest(dotted=dotted, dashed=dashed):
                self.assertEqual(table.resolve_price(dotted), table.resolve_price(dashed))

    def test_a_different_family_is_never_inherited_from_a_prefix(self) -> None:
        # The conservative half of version stripping: gpt-5.6 must not be billed at
        # the gpt-5 rate, because "6" is part of the model, not a build number.
        table = PriceTable()
        resolved = table.resolve("gpt-5.6-sol")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-5.6-sol")
        self.assertNotEqual(resolved[0], "gpt-5")

    def test_retired_models_keep_a_price_rather_than_being_dropped(self) -> None:
        # Dropping a name is not neutral: it turns every call to that model
        # unpriced, which excludes the spend from the budget. Someone still running
        # a withdrawn model is better served by its last published rate.
        table = PriceTable()
        for model in (
            "grok-3",
            "grok-4",
            "deepseek-chat",
            "deepseek-reasoner",
            "gemini-2.0-flash",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "gpt-4o",
            "o4-mini",
        ):
            with self.subTest(model=model):
                self.assertIsNotNone(
                    table.resolve_price(model),
                    f"{model} was dropped; its spend would now leave the budget",
                )

    def test_unknown_models_are_still_unpriced(self) -> None:
        self.assertIsNone(PriceTable().resolve_price("definitely-not-a-real-model-9000"))


class PriceTableTests(unittest.TestCase):
    def test_custom_base_replaces_the_bundled_table(self) -> None:
        table = PriceTable(base={"only": Price(1.0, 1.0)})
        self.assertEqual(len(table), 1)
        self.assertIsNone(table.resolve_price("gpt-4o"))

    def test_len_and_iteration(self) -> None:
        table = PriceTable()
        self.assertEqual(len(table), len(DEFAULT_PRICING))
        self.assertEqual(len(list(iter(table))), len(DEFAULT_PRICING))


class AliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = PriceTable(aliases={"acme/fast": "claude-3-5-haiku"})

    def test_alias_resolves_to_the_target_price(self) -> None:
        resolved = self.table.resolve("acme/fast")
        assert resolved is not None
        self.assertEqual(resolved[0], "claude-3-5-haiku")
        self.assertEqual(resolved[1], DEFAULT_PRICING["claude-3-5-haiku"])

    def test_alias_matching_ignores_case_and_surrounding_space(self) -> None:
        self.assertIsNotNone(self.table.resolve("  ACME/Fast "))

    def test_alias_is_matched_before_normalization(self) -> None:
        # "acme/gpt-4o" normalizes to "gpt-4o"; the alias must still win, which is
        # the whole point of naming a gateway model explicitly.
        table = PriceTable(aliases={"acme/gpt-4o": "claude-3-5-haiku"})
        resolved = table.resolve("acme/gpt-4o")
        assert resolved is not None
        self.assertEqual(resolved[0], "claude-3-5-haiku")

    def test_alias_may_point_at_a_dated_variant(self) -> None:
        table = PriceTable(aliases={"legacy": "gpt-4o-2024-08-06"})
        resolved = table.resolve("legacy")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o")

    def test_alias_chains_resolve(self) -> None:
        table = PriceTable(
            overrides={"mine": (1.0, 2.0)},
            aliases={"proxy": "fast", "fast": "mine"},
        )
        resolved = table.resolve("proxy")
        assert resolved is not None
        self.assertEqual(resolved[0], "mine")

    def test_alias_cycles_are_rejected(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            PriceTable(aliases={"a": "b", "b": "a"})
        self.assertIn("cycle", str(caught.exception))

    def test_alias_to_a_model_with_no_price_is_rejected(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            PriceTable(aliases={"a": "not-a-model"})
        self.assertIn("no price", str(caught.exception))

    def test_alias_name_cannot_also_carry_a_price(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            PriceTable(overrides={"mine": (1.0, 2.0)}, aliases={"mine": "gpt-4o"})
        self.assertIn("both a price and an alias", str(caught.exception))

    def test_alias_needs_a_non_empty_target(self) -> None:
        for bad in ("", "   ", None):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                PriceTable(aliases={"a": bad})  # type: ignore[dict-item]

    def test_alias_of_reports_the_target(self) -> None:
        self.assertEqual(self.table.alias_of("ACME/FAST"), "claude-3-5-haiku")
        self.assertIsNone(self.table.alias_of("gpt-4o"))

    def test_aliases_property_is_a_copy(self) -> None:
        aliases = self.table.aliases
        aliases["sneaky"] = "gpt-4o"
        self.assertIsNone(self.table.alias_of("sneaky"))

    def test_get_ignores_aliases(self) -> None:
        self.assertIsNone(self.table.get("acme/fast"))
        self.assertIsNotNone(self.table.resolve_price("acme/fast"))


class DisableTests(unittest.TestCase):
    def test_a_disabled_model_has_no_price(self) -> None:
        table = PriceTable(disable=["gpt-4"])
        self.assertIsNone(table.resolve_price("gpt-4"))
        self.assertNotIn("gpt-4", table)

    def test_a_disabled_family_takes_its_dated_variants_with_it(self) -> None:
        table = PriceTable(disable=["gpt-4o"])
        self.assertIsNone(table.resolve_price("gpt-4o-2024-08-06"))

    def test_disabling_an_unbundled_model_is_an_error(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            PriceTable(disable=["my-own-model"])
        self.assertIn("cannot disable", str(caught.exception))

    def test_disabled_names_are_normalized(self) -> None:
        self.assertEqual(PriceTable(disable=["OpenAI/GPT-4o"]).disabled, frozenset({"gpt-4o"}))

    def test_an_explicit_override_beats_a_disable(self) -> None:
        # Disable means "I do not trust the *bundled* price", not "never price this".
        table = PriceTable(overrides={"gpt-4": (1.0, 2.0)}, disable=["gpt-4"])
        self.assertEqual(table.resolve_price("gpt-4"), Price(1.0, 2.0))

    def test_disabling_shrinks_the_table(self) -> None:
        self.assertEqual(len(PriceTable(disable=["gpt-4"])), len(DEFAULT_PRICING) - 1)


class OriginTests(unittest.TestCase):
    def test_bundled_prices_are_marked_builtin(self) -> None:
        self.assertEqual(PriceTable().origin("gpt-4o"), "builtin")

    def test_overrides_are_marked_override(self) -> None:
        self.assertEqual(PriceTable(overrides={"mine": (1.0, 2.0)}).origin("mine"), "override")

    def test_unknown_and_disabled_models_have_no_origin(self) -> None:
        table = PriceTable(disable=["gpt-4"])
        self.assertIsNone(table.origin("who-knows"))
        self.assertIsNone(table.origin("gpt-4"))

    def test_an_alias_reports_the_origin_of_the_price_it_found(self) -> None:
        table = PriceTable(overrides={"mine": (1.0, 2.0)}, aliases={"fast": "mine"})
        self.assertEqual(table.origin("fast"), "override")


class FromConfigTests(unittest.TestCase):
    """`from_config` is where file, code and bundle are stitched together."""

    def config(self, **document: object):
        from agentguard.config import parse_config

        return parse_config(document)

    def test_config_models_are_marked_config(self) -> None:
        table = PriceTable.from_config(self.config(models={"mine": [1.0, 2.0]}))
        self.assertEqual(table.origin("mine"), "config")
        self.assertEqual(table.resolve_price("mine"), Price(1.0, 2.0))

    def test_code_overrides_beat_config_models(self) -> None:
        table = PriceTable.from_config(
            self.config(models={"mine": [1.0, 2.0]}), overrides={"mine": (3.0, 4.0)}
        )
        self.assertEqual(table.origin("mine"), "override")

    def test_config_can_reprice_a_bundled_model(self) -> None:
        table = PriceTable.from_config(self.config(models={"gpt-4o": [1.0, 1.0]}))
        self.assertEqual(table.origin("gpt-4o"), "config")

    def test_config_aliases_and_disables_are_applied(self) -> None:
        table = PriceTable.from_config(
            self.config(aliases={"fast": "gpt-4o-mini"}, disable=["gpt-4"])
        )
        self.assertIsNone(table.resolve_price("gpt-4"))
        resolved = table.resolve("fast")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o-mini")

    def test_none_builds_the_plain_bundled_table(self) -> None:
        table = PriceTable.from_config(None)
        self.assertEqual(len(table), len(DEFAULT_PRICING))
        self.assertEqual(table.sources, ())

    def test_sources_come_from_the_config(self) -> None:
        from pathlib import Path

        from agentguard.config import parse_config

        config = parse_config({"models": {"mine": [1.0, 2.0]}}, source=Path("x.json"))
        self.assertEqual(PriceTable.from_config(config).sources, (Path("x.json"),))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
