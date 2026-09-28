"""Tests for the ``agent-guard`` command line interface."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agentguard import Guard
from agentguard.cli import main
from agentguard.config import CONFIG_ENV_VAR, CONFIG_TRUST_ENV_VAR, user_config_path


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Invoke the CLI, capturing stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ReportCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

        guard = Guard(max_usd=1.0, max_steps=10, name="cli-agent")
        guard.record("gpt-4o", input_tokens=1000, output_tokens=100)
        self.path = guard.save(self.dir / "run.json")

    def test_renders_a_saved_report(self) -> None:
        code, out, _ = run_cli("report", str(self.path), "--ascii")
        self.assertEqual(code, 0)
        self.assertIn("cli-agent", out)
        self.assertIn("gpt-4o", out)
        self.assertIn("budget", out)

    def test_json_output_is_valid_json(self) -> None:
        code, out, _ = run_cli("report", str(self.path), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["calls"], 1)
        self.assertEqual(payload["name"], "cli-agent")

    def test_missing_file_is_a_clean_error(self) -> None:
        code, _, err = run_cli("report", str(self.dir / "nope.json"))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_invalid_json_is_a_clean_error(self) -> None:
        bad = self.dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = run_cli("report", str(bad))
        self.assertEqual(code, 2)
        self.assertIn("not valid JSON", err)

    def test_non_object_json_is_a_clean_error(self) -> None:
        bad = self.dir / "list.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        code, _, err = run_cli("report", str(bad))
        self.assertEqual(code, 2)
        self.assertIn("JSON object", err)


class PricingCommandTests(unittest.TestCase):
    def test_lists_the_table(self) -> None:
        code, out, _ = run_cli("pricing")
        self.assertEqual(code, 0)
        self.assertIn("models bundled", out)
        self.assertIn("gpt-4o", out)
        self.assertIn("input", out)

    def test_looks_up_a_known_model(self) -> None:
        code, out, _ = run_cli("pricing", "gpt-4o")
        self.assertEqual(code, 0)
        self.assertIn("gpt-4o", out)
        self.assertIn("example costs", out)
        self.assertIn("$12.5", out)  # 1M in + 1M out at 2.50 / 10.00

    def test_looks_up_a_dated_variant(self) -> None:
        code, out, _ = run_cli("pricing", "gpt-4o-2024-08-06")
        self.assertEqual(code, 0)
        self.assertIn("gpt-4o", out)

    def test_unknown_model_is_a_clean_error(self) -> None:
        code, _, err = run_cli("pricing", "not-a-real-model")
        self.assertEqual(code, 1)
        self.assertIn("no bundled price", err)
        # The suggested fix must be copy-pasteable Python: balanced braces.
        self.assertIn("Guard(pricing={'not-a-real-model': (input_per_1m, output_per_1m)})", err)


class GeneralTests(unittest.TestCase):
    def test_no_arguments_prints_help(self) -> None:
        code, out, _ = run_cli()
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)
        self.assertIn("report", out)
        self.assertIn("pricing", out)
        self.assertIn("config", out)

    def test_version_flag(self) -> None:
        out = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, redirect_stdout(out):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("agentguard", out.getvalue())

    def test_unknown_command_exits_non_zero(self) -> None:
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(err):
            main(["frobnicate"])
        self.assertNotEqual(ctx.exception.code, 0)


class ConfigCommandTests(unittest.TestCase):
    """The CLI is how a user configures prices without touching Python.

    Every path a command could write to is redirected into a temporary directory:
    a test must never be able to leave an ``agentguard.json`` in the developer's
    real home directory.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.env = mock.patch.dict(
            os.environ,
            {
                "APPDATA": str(self.dir),
                "XDG_CONFIG_HOME": str(self.dir),
                "HOME": str(self.dir),
                CONFIG_ENV_VAR: "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        # Writing with no target flag goes here, and discovery reads it back.
        self.path = user_config_path()
        self.other = self.dir / "elsewhere.json"

    def contents(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_path_shows_the_locations_and_what_is_read(self) -> None:
        code, out, _ = run_cli("config", "path")
        self.assertEqual(code, 0)
        self.assertIn("config locations", out)
        self.assertIn(CONFIG_ENV_VAR, out)
        self.assertIn(str(self.path), out)
        self.assertIn("bundled prices only", out)

    def test_init_writes_a_starter_file(self) -> None:
        code, out, _ = run_cli("config", "init")
        self.assertEqual(code, 0)
        self.assertIn(str(self.path), out)
        self.assertTrue(self.path.is_file())
        self.assertEqual(self.contents()["version"], 1)

    def test_init_refuses_to_clobber_without_force(self) -> None:
        run_cli("config", "init")
        code, _, err = run_cli("config", "init")
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        code, _, _ = run_cli("config", "init", "--force")
        self.assertEqual(code, 0)

    def test_set_adds_a_model_that_guard_then_uses(self) -> None:
        code, out, _ = run_cli("config", "set", "my-ft", "3", "12", "--cached", "0.3")
        self.assertEqual(code, 0)
        self.assertIn("my-ft", out)
        self.assertEqual(
            self.contents()["models"]["my-ft"],
            {"input": 3.0, "output": 12.0, "cached_input": 0.3},
        )
        guard = Guard(max_usd=100.0)
        record = guard.record("my-ft", input_tokens=1_000_000, output_tokens=0)
        self.assertAlmostEqual(record.cost_usd or 0.0, 3.0)

    def test_set_rejects_a_non_numeric_price(self) -> None:
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(err):
            main(["config", "set", "my-ft", "cheap", "12"])
        self.assertEqual(ctx.exception.code, 2)

    def test_set_rejects_an_impossible_price(self) -> None:
        code, _, err = run_cli("config", "set", "my-ft", "-3", "12")
        self.assertEqual(code, 2)
        self.assertIn("negative", err)

    def test_one_target_at_a_time(self) -> None:
        code, _, err = run_cli("config", "set", "m", "1", "2", "--user", "--file", str(self.other))
        self.assertEqual(code, 2)
        self.assertIn("choose one target", err)

    def test_file_flag_writes_where_it_is_told(self) -> None:
        code, _, _ = run_cli("config", "init", "--file", str(self.other))
        self.assertEqual(code, 0)
        self.assertTrue(self.other.is_file())

    def test_alias_makes_a_reported_name_billable(self) -> None:
        run_cli("config", "set", "my-ft", "3", "12")
        code, out, _ = run_cli("config", "alias", "acme/fast", "my-ft")
        self.assertEqual(code, 0)
        self.assertIn("acme/fast", out)
        guard = Guard(max_usd=100.0)
        self.assertAlmostEqual(
            guard.record("acme/fast", input_tokens=1_000_000, output_tokens=0).cost_usd or 0.0,
            3.0,
        )

    def test_alias_to_an_unknown_target_is_reported(self) -> None:
        code, _, err = run_cli("config", "alias", "fast", "who-knows")
        self.assertEqual(code, 2)
        self.assertIn("no price", err)

    def test_remove_drops_the_model_and_says_so(self) -> None:
        run_cli("config", "set", "my-ft", "3", "12")
        code, out, _ = run_cli("config", "remove", "my-ft")
        self.assertEqual(code, 0)
        self.assertIn("removed", out)
        self.assertEqual(self.contents()["models"], {})

    def test_remove_of_a_missing_name_exits_non_zero(self) -> None:
        code, _, err = run_cli("config", "remove", "not-there")
        self.assertEqual(code, 1)
        self.assertIn("not in", err)

    def test_disable_then_enable_a_bundled_model(self) -> None:
        code, out, _ = run_cli("config", "disable", "gpt-4")
        self.assertEqual(code, 0)
        self.assertIn("disabled gpt-4", out)
        self.assertEqual(self.contents()["disable"], ["gpt-4"])

        code, _, err = run_cli("config", "disable", "gpt-4")
        self.assertEqual(code, 1)
        self.assertIn("already disabled", err)

        code, _, _ = run_cli("config", "enable", "gpt-4")
        self.assertEqual(code, 0)
        self.assertEqual(self.contents()["disable"], [])

    def test_disable_refuses_a_model_that_is_not_bundled(self) -> None:
        code, _, err = run_cli("config", "disable", "my-own-model")
        self.assertEqual(code, 2)
        self.assertIn("not a bundled model", err)

    def test_list_shows_what_the_files_contain(self) -> None:
        run_cli("config", "set", "my-ft", "3", "12")
        run_cli("config", "alias", "acme/fast", "my-ft")
        run_cli("config", "disable", "gpt-4")
        code, out, _ = run_cli("config", "list")
        self.assertEqual(code, 0)
        self.assertIn("my-ft", out)
        self.assertIn("acme/fast -> my-ft", out)
        self.assertIn("disabled", out)

    def test_list_explains_a_config_file_that_cannot_be_loaded(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"models": {"m": {"input": 1}}}', encoding="utf-8")
        code, out, _ = run_cli("config", "list")
        self.assertEqual(code, 1)
        self.assertIn("missing 'output'", out)

    def test_a_write_to_a_file_that_is_not_in_effect_is_flagged(self) -> None:
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: "none"}):
            code, _, err = run_cli("config", "set", "my-ft", "3", "12")
        self.assertEqual(code, 0)
        self.assertIn("not in effect", err)


class ProjectConfigTrustCliTests(unittest.TestCase):
    """A project config is somebody else's data until you say otherwise."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.env = mock.patch.dict(
            os.environ,
            {
                "APPDATA": str(self.dir),
                "XDG_CONFIG_HOME": str(self.dir),
                "HOME": str(self.dir),
                CONFIG_ENV_VAR: "",
                CONFIG_TRUST_ENV_VAR: "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.original_cwd = Path.cwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, self.original_cwd)
        self.project = self.dir / "agentguard.json"
        self.project.write_text(
            json.dumps({"version": 1, "models": {"from-project": [1.0, 2.0]}}), encoding="utf-8"
        )

    def assert_same_file(self, reported: object, expected: Path) -> None:
        """Assert the CLI named *this* file, whatever way it spelled it.

        macOS reaches its temporary directory through ``/var``, a symlink to
        ``/private/var``, so ``os.getcwd()`` — what discovery walks up from — hands
        back a different string than ``tempfile`` gave the test. Comparing raw
        strings made these tests pass on Linux and Windows and fail on every macOS
        Python, which is exactly what the CI badge showed.
        """
        self.assertIsNotNone(reported)
        self.assertEqual(Path(str(reported)).resolve(), expected.resolve())

    def test_path_marks_the_project_file_as_ignored(self) -> None:
        code, out, _ = run_cli("config", "path")
        self.assertEqual(code, 0)
        self.assertIn("ignored: not trusted", out)
        self.assertIn(CONFIG_TRUST_ENV_VAR, out)

    def test_pricing_does_not_use_an_untrusted_project_file(self) -> None:
        code, out, _ = run_cli("pricing", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertNotIn("from-project", {row["model"] for row in payload["models"]})
        self.assertEqual(payload["config_files"], [])
        self.assert_same_file(payload["ignored_config"], self.project)

    def test_pricing_prints_a_note_instead_of_a_warning(self) -> None:
        code, out, err = run_cli("pricing")
        self.assertEqual(code, 0)
        self.assertIn("not trusted", out)
        self.assertIn(CONFIG_TRUST_ENV_VAR, out)
        self.assertEqual(err, "")

    def test_list_mentions_the_file_it_is_not_reading(self) -> None:
        code, out, _ = run_cli("config", "list")
        self.assertEqual(code, 0)
        self.assertIn("not trusted", out)

    def test_writing_a_project_config_says_it_needs_trust(self) -> None:
        code, _, err = run_cli("config", "set", "--project", "my-ft", "3", "12")
        self.assertEqual(code, 0)
        self.assertIn("not trusted", err)
        self.assertIn(CONFIG_TRUST_ENV_VAR, err)
        self.assertNotIn("RuntimeWarning", err)

    def test_a_relative_file_target_is_not_called_missing(self) -> None:
        # `--file agentguard.json` writes the project config, so the note has to be
        # the trust one. Comparing the relative path against the absolute path
        # discovery reports made this say "not in effect (no config file)".
        code, out, err = run_cli("config", "set", "--file", "agentguard.json", "my-ft", "3", "12")
        self.assertEqual(code, 0)
        self.assertIn("agentguard.json", out)
        self.assertIn("not trusted", err)
        self.assertNotIn("not in effect", err)

    def test_trusting_it_makes_it_effective(self) -> None:
        with mock.patch.dict(os.environ, {CONFIG_TRUST_ENV_VAR: "1"}):
            code, out, _ = run_cli("pricing", "--json")
            self.assertEqual(code, 0)
            payload = json.loads(out)
            models = {row["model"]: row for row in payload["models"]}
            self.assertEqual(models["from-project"]["source"], "config")
            reported = list(payload["config_files"])
            self.assertEqual(len(reported), 1)
            self.assert_same_file(reported[0], self.project)


class ConfiguredPricingTests(unittest.TestCase):
    """`agentguard pricing` must show what Guard will actually bill."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "agentguard.json"
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": {"my-ft": {"input": 3.0, "output": 12.0}},
                    "aliases": {"acme/fast": "my-ft"},
                    "disable": ["gpt-4"],
                }
            ),
            encoding="utf-8",
        )
        self.env = mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_the_table_marks_where_each_price_came_from(self) -> None:
        code, out, _ = run_cli("pricing")
        self.assertEqual(code, 0)
        self.assertIn("models bundled, 1 configured", out)
        self.assertIn("my-ft", out)
        self.assertIn("config", out)
        self.assertIn("acme/fast -> my-ft", out)
        self.assertIn("disabled", out)
        self.assertIn(str(self.path), out)

    def test_detail_reports_the_source_of_a_configured_price(self) -> None:
        code, out, _ = run_cli("pricing", "my-ft")
        self.assertEqual(code, 0)
        self.assertIn("source", out)
        self.assertIn("config", out)

    def test_detail_follows_an_alias(self) -> None:
        code, out, _ = run_cli("pricing", "acme/fast")
        self.assertEqual(code, 0)
        self.assertIn("acme/fast -> my-ft", out)

    def test_a_disabled_model_is_explained_rather_than_unknown(self) -> None:
        code, _, err = run_cli("pricing", "gpt-4")
        self.assertEqual(code, 1)
        self.assertIn("disabled", err)
        self.assertIn("config enable gpt-4", err)

    def test_no_config_shows_only_the_bundled_table(self) -> None:
        code, out, _ = run_cli("pricing", "--no-config")
        self.assertEqual(code, 0)
        self.assertNotIn("my-ft", out)
        self.assertIn("41 models bundled", out)

    def test_json_output_is_machine_readable(self) -> None:
        code, out, _ = run_cli("pricing", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        models = {row["model"]: row for row in payload["models"]}
        self.assertEqual(models["my-ft"]["source"], "config")
        self.assertEqual(models["gpt-4o"]["source"], "builtin")
        self.assertEqual(payload["aliases"], {"acme/fast": "my-ft"})
        self.assertEqual(payload["disabled"], ["gpt-4"])
        self.assertEqual(payload["config_files"], [str(self.path)])

    def test_json_can_be_narrowed_to_one_model(self) -> None:
        code, out, _ = run_cli("pricing", "acme/fast", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual([row["model"] for row in payload["models"]], ["my-ft"])
        self.assertEqual(payload["requested"], "acme/fast")
        self.assertEqual(payload["alias_of"], "my-ft")

    def test_json_for_an_unknown_model_exits_non_zero(self) -> None:
        code, _, err = run_cli("pricing", "not-a-model", "--json")
        self.assertEqual(code, 1)
        self.assertIn("no price", err)

    def test_a_broken_config_file_is_a_clean_error(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        code, _, err = run_cli("pricing")
        self.assertEqual(code, 2)
        self.assertIn("not valid JSON", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
