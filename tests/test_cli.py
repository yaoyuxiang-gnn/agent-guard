"""Tests for the ``agent-guard`` command line interface."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agentguard import Guard
from agentguard.cli import main


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
        self.assertIn("pricing=", err)


class GeneralTests(unittest.TestCase):
    def test_no_arguments_prints_help(self) -> None:
        code, out, _ = run_cli()
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)
        self.assertIn("report", out)
        self.assertIn("pricing", out)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
