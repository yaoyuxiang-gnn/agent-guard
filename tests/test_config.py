"""Tests for the user pricing configuration.

A config file is a user's own statement about what their models cost, so the
things worth pinning down are the ones that decide *whose* number wins, what
happens when the file is wrong, and whether a typo can quietly change a budget.
The rule this suite enforces: anything unrecognised is an error, never a shrug.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from agentguard import DEFAULT_PRICING, Guard, GuardConfigError, Price, PriceTable, Report
from agentguard import config as config_module
from agentguard.config import (
    CONFIG_ENV_VAR,
    CONFIG_TRUST_ENV_VAR,
    CONFIG_VERSION,
    Removal,
    config_paths,
    ignored_project_config,
    initialize_config,
    load_config,
    merge_configs,
    parse_config,
    project_config_path,
    project_config_trusted,
    read_config_file,
    remove_entry,
    set_alias,
    set_disabled,
    set_model_price,
    user_config_path,
    write_config_file,
)

USER_FILE = Path("agentguard") / "pricing.json"
PROJECT_FILE = "agentguard.json"


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ConfigTestCase(unittest.TestCase):
    """Shared scaffolding: a temporary directory that acts as the whole world."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    @property
    def user_file(self) -> Path:
        return self.root / USER_FILE

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    def env(self, **extra: str) -> dict[str, str]:
        """The environment discovery sees, with the world confined to ``root``.

        ``windows=True`` keeps the per-user path deterministic on any platform: it
        resolves to ``%APPDATA%``, which is pinned to the temporary directory. The
        project file is trusted here because most of this file is about *merging*;
        :class:`ProjectConfigTrustTests` covers what happens when it is not.
        """
        return {"APPDATA": str(self.root), CONFIG_TRUST_ENV_VAR: "1", **extra}

    def discover(self, *, start: Path | None = None, **env: str):
        """Load config the way Guard does, with the project file trusted."""
        return load_config(start=start, env=self.env(**env), windows=True)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


class ParseConfigTests(unittest.TestCase):
    def test_empty_document_is_valid_and_empty(self) -> None:
        config = parse_config({})
        self.assertTrue(config.is_empty)
        self.assertEqual(config.sources, ())

    def test_models_accept_mapping_and_array_forms(self) -> None:
        config = parse_config(
            {
                "models": {
                    "with-object": {"input": 3.0, "output": 12.0, "cached_input": 0.3},
                    "with-array": [0.05, 0.08],
                    "with-array-and-cache": [1.0, 2.0, 0.5],
                }
            }
        )
        self.assertEqual(config.models["with-object"].cached_input_per_1m, 0.3)
        self.assertEqual(config.models["with-array"].output_per_1m, 0.08)
        self.assertIsNone(config.models["with-array"].cached_input_per_1m)
        self.assertEqual(config.models["with-array-and-cache"].cached_input_per_1m, 0.5)

    def test_field_names_from_price_repr_are_accepted(self) -> None:
        config = parse_config({"models": {"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}})
        self.assertEqual(config.models["m"], Price(1.0, 2.0))

    def test_aliases_and_disable(self) -> None:
        config = parse_config({"aliases": {"acme/fast": "claude-3-5-haiku"}, "disable": ["gpt-4"]})
        self.assertEqual(config.aliases, {"acme/fast": "claude-3-5-haiku"})
        self.assertEqual(config.disable, frozenset({"gpt-4"}))

    def test_version_is_optional_and_checked(self) -> None:
        self.assertTrue(parse_config({"version": CONFIG_VERSION}).is_empty)
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"version": CONFIG_VERSION + 1})
        self.assertIn("newer agentguard", str(caught.exception))

    def test_document_must_be_an_object(self) -> None:
        for bad in ([], "models", 3, None):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                parse_config(bad)

    def test_unknown_top_level_key_is_an_error(self) -> None:
        # "modal" instead of "models" must not silently configure nothing.
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"modal": {"m": [1, 2]}})
        self.assertIn("unknown key", str(caught.exception))
        self.assertIn("models", str(caught.exception))

    def test_unknown_key_inside_a_model_is_an_error(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"models": {"m": {"input": 1, "outpt": 2}}})
        self.assertIn("outpt", str(caught.exception))

    def test_a_model_needs_both_directions(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"models": {"m": {"input": 1.0}}})
        self.assertIn("missing 'output'", str(caught.exception))

    def test_bad_rates_are_rejected(self) -> None:
        for bad in (-1.0, float("nan"), float("inf"), "cheap", True, None):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                parse_config({"models": {"m": [bad, 2.0]}})

    def test_one_name_cannot_be_both_a_model_and_an_alias(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"models": {"m": [1, 2]}, "aliases": {"m": "gpt-4o"}})
        self.assertIn("both a price and an alias", str(caught.exception))

    def test_two_names_resolving_to_one_model_are_rejected(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"models": {"gpt-4o": [1, 2], "OpenAI/GPT-4o": [3, 4]}})
        self.assertIn("both resolve to", str(caught.exception))

    def test_disable_must_be_an_array_of_names(self) -> None:
        for bad in ("gpt-4", 3, {"gpt-4": True}, [""]):
            with self.subTest(bad=bad), self.assertRaises(GuardConfigError):
                parse_config({"disable": bad})

    def test_disable_only_accepts_bundled_models(self) -> None:
        # Disabling a name that is not in the bundle would do nothing at all, so
        # it is an error rather than a no-op the user never hears about.
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"disable": ["my-own-model"]})
        self.assertIn("no such entry", str(caught.exception))

    def test_an_alias_must_point_at_a_model_with_a_price(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"aliases": {"fast": "nope"}})
        self.assertIn("no price", str(caught.exception))

    def test_errors_name_the_file(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            parse_config({"nope": 1}, source=Path("C:/etc/agentguard.json"))
        self.assertIn("agentguard.json", str(caught.exception))

    def test_as_dict_round_trips(self) -> None:
        config = parse_config({"models": {"m": [1.0, 2.0]}, "aliases": {"a": "m"}})
        payload = config.as_dict()
        self.assertEqual(payload["aliases"], {"a": "m"})
        self.assertEqual(payload["models"]["m"]["input_per_1m"], 1.0)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


class PathTests(unittest.TestCase):
    def test_windows_user_path_uses_appdata(self) -> None:
        path = user_config_path(env={"APPDATA": "C:/Users/x/AppData/Roaming"}, windows=True)
        self.assertEqual(path.as_posix(), "C:/Users/x/AppData/Roaming/agentguard/pricing.json")

    def test_posix_user_path_prefers_xdg_config_home(self) -> None:
        path = user_config_path(env={"XDG_CONFIG_HOME": "/tmp/cfg"}, windows=False)
        self.assertEqual(path.as_posix(), "/tmp/cfg/agentguard/pricing.json")

    def test_posix_user_path_falls_back_to_dot_config(self) -> None:
        path = user_config_path(env={"HOME": "/home/x"}, windows=False)
        self.assertEqual(path.as_posix(), "/home/x/.config/agentguard/pricing.json")


class DiscoveryTests(ConfigTestCase):
    def test_nothing_configured_yields_nothing(self) -> None:
        self.assertEqual(config_paths(start=self.root, env={"APPDATA": str(self.root)}), ())
        self.assertTrue(self.discover(start=self.root).is_empty)

    def test_project_file_is_found_by_walking_up(self) -> None:
        nested = self.root / "packages" / "agent" / "src"
        nested.mkdir(parents=True)
        write_json(self.project_file, {"models": {"m": [1.0, 2.0]}})
        self.assertEqual(project_config_path(nested), self.project_file)
        self.assertIn("m", self.discover(start=nested).models)

    def test_hidden_project_filename_is_recognised(self) -> None:
        nested = self.root / "sub"
        nested.mkdir()
        hidden = write_json(self.root / ".agentguard.json", {"models": {"m": [1.0, 2.0]}})
        self.assertEqual(project_config_path(nested), hidden)

    def test_user_and_project_files_are_both_read(self) -> None:
        write_json(self.user_file, {"models": {"from-user": [1.0, 1.0]}})
        write_json(self.project_file, {"models": {"from-project": [2.0, 2.0]}})
        config = self.discover(start=self.root)
        self.assertEqual(set(config.models), {"from-user", "from-project"})
        self.assertEqual(config.sources, (self.user_file, self.project_file))

    def test_project_file_wins_on_the_same_model(self) -> None:
        write_json(self.user_file, {"models": {"m": [1.0, 2.0]}})
        write_json(self.project_file, {"models": {"m": [9.0, 9.0]}})
        self.assertEqual(self.discover(start=self.root).models["m"].input_per_1m, 9.0)

    def test_project_file_may_alias_a_model_from_the_user_file(self) -> None:
        write_json(self.user_file, {"models": {"mine": [1.0, 2.0]}})
        write_json(self.project_file, {"aliases": {"fast": "mine"}})
        config = self.discover(start=self.root)
        self.assertEqual(config.aliases, {"fast": "mine"})
        self.assertIsNotNone(PriceTable.from_config(config).resolve("fast"))

    def test_two_files_defining_one_model_name_are_a_config_error(self) -> None:
        write_json(self.user_file, {"models": {"My-Model": [1.0, 2.0]}})
        write_json(self.project_file, {"models": {"OpenAI/My-Model": [3.0, 4.0]}})
        with self.assertRaises(GuardConfigError) as caught:
            self.discover(start=self.root)
        self.assertIn("both resolve to", str(caught.exception))

    def test_env_var_replaces_discovery(self) -> None:
        write_json(self.project_file, {"models": {"project-model": [1.0, 1.0]}})
        other = write_json(self.root / "elsewhere" / "picked.json", {"models": {"m": [5.0, 5.0]}})
        config = self.discover(start=self.root, **{CONFIG_ENV_VAR: str(other)})
        self.assertEqual(set(config.models), {"m"})
        self.assertEqual(config.sources, (other,))

    def test_env_var_can_disable_config_entirely(self) -> None:
        write_json(self.project_file, {"models": {"m": [1.0, 1.0]}})
        for off in ("none", "off", "0", "false", "no", "NONE"):
            with self.subTest(off=off):
                config = self.discover(start=self.root, **{CONFIG_ENV_VAR: off})
                self.assertTrue(config.is_empty)
                self.assertEqual(config.sources, ())

    def test_env_var_pointing_at_a_missing_file_is_an_error(self) -> None:
        # Silently ignoring an explicitly requested file would mean billing at
        # prices the user believes they replaced.
        missing = self.root / "nope.json"
        with self.assertRaises(GuardConfigError) as caught:
            self.discover(start=self.root, **{CONFIG_ENV_VAR: str(missing)})
        self.assertIn("not found", str(caught.exception))


class LoadConfigTests(ConfigTestCase):
    def test_missing_file_raises_when_explicitly_requested(self) -> None:
        with self.assertRaises(GuardConfigError):
            load_config(self.root / "nope.json")

    def test_an_empty_file_is_an_empty_config(self) -> None:
        path = self.root / "agentguard.json"
        path.write_text("   \n", encoding="utf-8")
        self.assertTrue(load_config(path).is_empty)
        self.assertEqual(load_config(path).sources, (path,))

    def test_invalid_json_names_the_file(self) -> None:
        path = self.root / "agentguard.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(GuardConfigError) as caught:
            load_config(path)
        self.assertIn("not valid JSON", str(caught.exception))
        self.assertIn("agentguard.json", str(caught.exception))

    def test_env_mapping_is_not_required_to_be_os_environ(self) -> None:
        # Passing env= is what makes discovery testable; it must not read the
        # real environment behind the caller's back.
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: "none"}):
            self.assertTrue(load_config(env={}).is_empty)


class MergeConfigTests(unittest.TestCase):
    def test_disable_entries_are_combined(self) -> None:
        merged = merge_configs(
            [
                parse_config({"disable": ["gpt-4"]}),
                parse_config({"disable": ["claude-3-opus"]}),
            ]
        )
        self.assertEqual(merged.disable, frozenset({"gpt-4", "claude-3-opus"}))

    def test_sources_are_kept_in_load_order_without_duplicates(self) -> None:
        first = parse_config({}, source=Path("/a.json"))
        second = parse_config({}, source=Path("/b.json"))
        self.assertEqual(
            merge_configs([first, second, first]).sources, (Path("/a.json"), Path("/b.json"))
        )


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #


class EditTests(ConfigTestCase):
    def test_initialize_writes_a_valid_starter_file(self) -> None:
        path = initialize_config(self.project_file)
        self.assertEqual(
            read_config_file(path),
            {"version": CONFIG_VERSION, "models": {}, "aliases": {}, "disable": []},
        )

    def test_initialize_refuses_to_clobber_without_force(self) -> None:
        write_json(self.project_file, {"models": {"keep-me": [1.0, 2.0]}})
        with self.assertRaises(GuardConfigError):
            initialize_config(self.project_file)
        self.assertIn("keep-me", read_config_file(self.project_file)["models"])
        initialize_config(self.project_file, force=True)
        self.assertEqual(read_config_file(self.project_file)["models"], {})

    def test_set_model_price_creates_then_updates(self) -> None:
        set_model_price(self.project_file, "my-ft", input_per_1m=3.0, output_per_1m=12.0)
        price = set_model_price(
            self.project_file,
            "my-ft",
            input_per_1m=4.0,
            output_per_1m=16.0,
            cached_input_per_1m=0.4,
        )
        self.assertEqual(price, Price(4.0, 16.0, 0.4))
        config = load_config(self.project_file)
        self.assertEqual(config.models["my-ft"], Price(4.0, 16.0, 0.4))

    def test_set_model_price_rejects_a_name_used_as_an_alias(self) -> None:
        set_alias(self.project_file, "fast", "gpt-4o")
        with self.assertRaises(GuardConfigError) as caught:
            set_model_price(self.project_file, "fast", input_per_1m=1.0, output_per_1m=1.0)
        self.assertIn("alias", str(caught.exception))

    def test_set_alias_points_a_reported_name_at_a_price(self) -> None:
        set_alias(self.project_file, "acme/fast", "claude-3-5-haiku")
        table = PriceTable.from_config(load_config(self.project_file))
        resolved = table.resolve("acme/fast")
        assert resolved is not None
        self.assertEqual(resolved[0], "claude-3-5-haiku")

    def test_set_alias_refuses_a_target_with_no_price(self) -> None:
        with self.assertRaises(GuardConfigError):
            set_alias(self.project_file, "fast", "who-knows")
        # The rejected edit must not have been written at all.
        self.assertEqual(read_config_file(self.project_file)["aliases"], {})

    def test_disable_and_enable_a_bundled_model(self) -> None:
        self.assertTrue(set_disabled(self.project_file, "gpt-4"))
        self.assertEqual(load_config(self.project_file).disable, frozenset({"gpt-4"}))
        self.assertFalse(set_disabled(self.project_file, "gpt-4"))  # already disabled
        self.assertTrue(set_disabled(self.project_file, "gpt-4", False))
        self.assertFalse(set_disabled(self.project_file, "gpt-4", False))
        self.assertEqual(load_config(self.project_file).disable, frozenset())

    def test_disable_refuses_a_model_that_is_not_bundled(self) -> None:
        with self.assertRaises(GuardConfigError) as caught:
            set_disabled(self.project_file, "my-own-model")
        self.assertIn("not a bundled model", str(caught.exception))

    def test_disable_refuses_to_break_an_alias(self) -> None:
        set_alias(self.project_file, "fast", "gpt-4")
        with self.assertRaises(GuardConfigError) as caught:
            set_disabled(self.project_file, "gpt-4")
        self.assertIn("fast", str(caught.exception))

    def test_disable_refuses_a_model_the_same_file_prices(self) -> None:
        # A price beats a disable, so writing one would silently do nothing.
        set_model_price(self.project_file, "gpt-4o", input_per_1m=1.0, output_per_1m=1.0)
        with self.assertRaises(GuardConfigError) as caught:
            set_disabled(self.project_file, "gpt-4o")
        self.assertIn("beats a disable", str(caught.exception))

    def test_disable_matches_a_dated_variant_of_a_bundled_model(self) -> None:
        set_disabled(self.project_file, "gpt-4o-2024-08-06")
        self.assertEqual(load_config(self.project_file).disable, frozenset({"gpt-4o"}))

    def test_remove_entry_deletes_a_model_and_the_aliases_pointing_at_it(self) -> None:
        set_model_price(self.project_file, "my-ft", input_per_1m=3.0, output_per_1m=12.0)
        set_alias(self.project_file, "acme/fast", "my-ft")
        removal = remove_entry(self.project_file, "my-ft")
        self.assertEqual(removal.models, ("my-ft",))
        self.assertEqual(removal.aliases, ("acme/fast",))
        self.assertTrue(removal.changed)
        document = read_config_file(self.project_file)
        self.assertEqual(document["models"], {})
        self.assertEqual(document["aliases"], {})

    def test_remove_entry_reports_nothing_found(self) -> None:
        removal = remove_entry(self.project_file, "not-there")
        self.assertFalse(removal.changed)
        self.assertEqual(removal.describe(), "nothing")

    def test_remove_entry_clears_a_disable_entry(self) -> None:
        set_disabled(self.project_file, "gpt-4")
        removal = remove_entry(self.project_file, "gpt-4")
        self.assertEqual(removal.disabled, ("gpt-4",))
        self.assertEqual(load_config(self.project_file).disable, frozenset())

    def test_removal_describes_what_it_did(self) -> None:
        removal = Removal(models=("a",), aliases=("b",))
        self.assertEqual(removal.describe(), "model 'a' and alias 'b'")

    def test_a_stale_disable_entry_can_be_repaired(self) -> None:
        # A name that the bundled table no longer has: Guard must refuse to load
        # it, but `config remove` has to be able to clean it up.
        write_json(self.project_file, {"version": 1, "disable": ["gone-in-v2"]})
        with self.assertRaises(GuardConfigError):
            load_config(self.project_file)
        removal = remove_entry(self.project_file, "gone-in-v2")
        self.assertTrue(removal.changed)
        self.assertTrue(load_config(self.project_file).is_empty)

    def test_edits_never_leave_a_broken_file_behind(self) -> None:
        set_model_price(self.project_file, "my-ft", input_per_1m=3.0, output_per_1m=12.0)
        before = read_config_file(self.project_file)
        with self.assertRaises(GuardConfigError):
            write_config_file(self.project_file, {"models": {"m": [-1.0, 2.0]}})
        self.assertEqual(read_config_file(self.project_file), before)

    def test_writing_creates_missing_directories(self) -> None:
        path = self.root / "deep" / "nested" / "agentguard.json"
        set_model_price(path, "m", input_per_1m=1.0, output_per_1m=2.0)
        self.assertTrue(path.is_file())

    def test_read_of_an_invalid_file_is_reported_not_silently_replaced(self) -> None:
        path = self.root / "agentguard.json"
        path.write_text("{oops", encoding="utf-8")
        with self.assertRaises(GuardConfigError):
            read_config_file(path)


class ProjectConfigTrustTests(ConfigTestCase):
    """A config that travels with a repository is data from someone else.

    Reading it by default would make "check out this repo and run your agent in
    it" a documented way to reprice every model or disable the expensive ones,
    which is a budget bypass performed with a JSON file.
    """

    def setUp(self) -> None:
        super().setUp()
        write_json(self.project_file, {"models": {"from-project": [1.0, 2.0]}})
        # The once-per-process dedupe would hide the warning from later tests.
        self.addCleanup(setattr, config_module, "_warned_untrusted", set())
        config_module._warned_untrusted.clear()

    def test_project_file_is_skipped_and_reported(self) -> None:
        with self.assertWarns(RuntimeWarning) as caught:
            config = load_config(
                start=self.root, env=self.env(**{CONFIG_TRUST_ENV_VAR: ""}), windows=True
            )
        self.assertTrue(config.is_empty)
        self.assertIn(str(self.project_file), str(caught.warning))
        self.assertIn(CONFIG_TRUST_ENV_VAR, str(caught.warning))

    def test_the_warning_fires_once_per_file(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                load_config(
                    start=self.root, env=self.env(**{CONFIG_TRUST_ENV_VAR: ""}), windows=True
                )
        self.assertEqual(len(caught), 1)

    def test_trusted_project_file_is_read(self) -> None:
        config = load_config(start=self.root, env=self.env(), windows=True)
        self.assertIn("from-project", config.models)

    def test_user_file_is_still_read_while_the_project_file_is_ignored(self) -> None:
        write_json(self.user_file, {"models": {"from-user": [3.0, 3.0]}})
        with self.assertWarns(RuntimeWarning):
            config = load_config(
                start=self.root, env=self.env(**{CONFIG_TRUST_ENV_VAR: ""}), windows=True
            )
        self.assertEqual(set(config.models), {"from-user"})

    def test_no_warning_when_there_is_no_project_file(self) -> None:
        self.project_file.unlink()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(start=self.root, env=self.env(**{CONFIG_TRUST_ENV_VAR: ""}), windows=True)
        self.assertEqual(caught, [])

    def test_env_var_pointing_at_the_project_file_needs_no_trust(self) -> None:
        config = load_config(
            start=self.root,
            env=self.env(**{CONFIG_TRUST_ENV_VAR: "", CONFIG_ENV_VAR: str(self.project_file)}),
            windows=True,
        )
        self.assertIn("from-project", config.models)

    def test_guard_can_be_pointed_at_it_explicitly(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        self.assertEqual(guard.price_table.origin("from-project"), "config")

    def test_guard_ignores_it_by_default(self) -> None:
        # Guard() reads os.environ, so the whole environment is pinned here rather
        # than passed in: cwd inside the project, no explicit path, no trust.
        nested = self.root / "sub"
        nested.mkdir()
        original = Path.cwd()
        os.chdir(nested)
        self.addCleanup(os.chdir, original)
        with (
            mock.patch.dict(
                os.environ,
                {"APPDATA": str(self.root), CONFIG_ENV_VAR: "", CONFIG_TRUST_ENV_VAR: ""},
            ),
            self.assertWarns(RuntimeWarning),
        ):
            guard = Guard(max_usd=1.0)
        self.assertIsNone(guard.price_table.origin("from-project"))

    def test_ignored_project_config_helper(self) -> None:
        self.assertIsNotNone(
            ignored_project_config(start=self.root, env=self.env(**{CONFIG_TRUST_ENV_VAR: ""}))
        )
        self.assertIsNone(ignored_project_config(start=self.root, env=self.env()))

    def test_truthy_spellings_are_accepted(self) -> None:
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value):
                self.assertTrue(project_config_trusted(env={CONFIG_TRUST_ENV_VAR: value}))
        for value in ("", "0", "no", "maybe"):
            with self.subTest(value=value):
                self.assertFalse(project_config_trusted(env={CONFIG_TRUST_ENV_VAR: value}))


# --------------------------------------------------------------------------- #
# Guard integration
# --------------------------------------------------------------------------- #


class GuardConfigTests(ConfigTestCase):
    def setUp(self) -> None:
        super().setUp()
        write_json(
            self.project_file,
            {
                "version": 1,
                "models": {"my-ft": {"input": 3.0, "output": 12.0, "cached_input": 0.3}},
                "aliases": {"acme/fast": "my-ft"},
                "disable": ["gpt-4"],
            },
        )

    def test_guard_loads_the_config_it_is_pointed_at(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        self.assertEqual(guard.price_table.origin("my-ft"), "config")
        self.assertEqual(guard.pricing_config.sources, (self.project_file,))

    def test_configured_model_is_billed_at_the_configured_price(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        record = guard.record("my-ft", input_tokens=1_000, output_tokens=1_000)
        self.assertAlmostEqual(record.cost_usd or 0.0, (1_000 * 3.0 + 1_000 * 12.0) / 1_000_000)

    def test_alias_bills_at_the_target_price(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        record = guard.record("acme/fast", input_tokens=1_000, output_tokens=1_000)
        self.assertAlmostEqual(record.cost_usd or 0.0, 0.015)
        self.assertEqual(record.canonical_model, "my-ft")

    def test_disabled_model_counts_as_unpriced_not_as_a_guess(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        with self.assertWarns(RuntimeWarning):
            record = guard.record("gpt-4", input_tokens=1_000, output_tokens=1_000)
        self.assertFalse(record.priced)
        self.assertEqual(guard.spent_usd, 0.0)
        self.assertEqual(guard.report().unpriced_calls, 1)

    def test_env_var_is_discovered_without_being_asked(self) -> None:
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(self.project_file)}):
            guard = Guard(max_usd=1.0)
            self.assertEqual(guard.price_table.origin("my-ft"), "config")
            self.assertIn("my-ft", guard.price_table)

    def test_use_config_false_ignores_every_file(self) -> None:
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(self.project_file)}):
            guard = Guard(max_usd=1.0, use_config=False)
        self.assertEqual(len(guard.price_table), len(DEFAULT_PRICING))
        self.assertIsNone(guard.price_table.origin("my-ft"))
        self.assertTrue(guard.pricing_config.is_empty)

    def test_code_overrides_beat_the_config_file(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file, pricing={"my-ft": (99.0, 99.0)})
        self.assertEqual(guard.price_table.origin("my-ft"), "override")
        self.assertEqual(guard.price_table.resolve_price("my-ft").input_per_1m, 99.0)

    def test_code_can_add_aliases_and_disables_too(self) -> None:
        guard = Guard(
            max_usd=1.0,
            config_path=self.project_file,
            aliases={"internal-llm": "gpt-4o"},
            disable=["claude-3-opus"],
        )
        resolved = guard.price_table.resolve("internal-llm")
        assert resolved is not None
        self.assertEqual(resolved[0], "gpt-4o")
        self.assertIsNone(guard.price_table.resolve_price("claude-3-opus"))

    def test_a_prebuilt_table_skips_config_discovery(self) -> None:
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(self.project_file)}):
            guard = Guard(max_usd=1.0, price_table=PriceTable())
        self.assertIsNone(guard.price_table.origin("my-ft"))

    def test_a_prebuilt_config_object_bypasses_the_filesystem(self) -> None:
        guard = Guard(max_usd=1.0, config=parse_config({"models": {"inline": [1.0, 2.0]}}))
        self.assertEqual(guard.price_table.origin("inline"), "config")

    def test_a_broken_config_fails_at_construction_not_at_the_first_call(self) -> None:
        write_json(self.project_file, {"models": {"m": {"input": 1.0}}})
        with self.assertRaises(GuardConfigError):
            Guard(max_usd=1.0, config_path=self.project_file)

    def test_report_names_the_config_that_priced_the_run(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        guard.record("my-ft", input_tokens=1_000, output_tokens=100)
        report = guard.report()
        self.assertEqual(report.pricing_sources, (str(self.project_file),))
        self.assertIn("pricing config", report.render(ascii_only=True))
        self.assertEqual(report.as_dict()["pricing_sources"], [str(self.project_file)])
        # A report round-trips through JSON, so the config must survive as_dict.
        self.assertEqual(Report.from_dict(report.as_dict()).pricing_sources, report.pricing_sources)

    def test_reset_keeps_the_configured_prices(self) -> None:
        guard = Guard(max_usd=1.0, config_path=self.project_file)
        guard.record("my-ft", input_tokens=1_000, output_tokens=0)
        guard.reset()
        self.assertEqual(guard.price_table.origin("my-ft"), "config")
        self.assertEqual(guard.spent_usd, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
