"""Run every docstring example in the package as a test.

Docstring examples are the first code anyone copies out of the repository, so they
are part of the test suite rather than decoration. The same applies to the examples
in ``docs/``: an API reference whose examples are wrong is worse than one with none.

The cases below are generated as ordinary :class:`unittest.TestCase` subclasses
rather than through the ``load_tests`` protocol, because pytest does not implement
``load_tests`` — with it, these tests would silently collect as zero under pytest
while passing under ``python -m unittest``.
"""

from __future__ import annotations

import doctest
import importlib
import io
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_MODULE_NAMES = (
    "agentguard",
    "agentguard._util",
    "agentguard._version",
    "agentguard.adapters",
    "agentguard.adapters.anthropic",
    "agentguard.adapters.openai",
    "agentguard.cli",
    "agentguard.config",
    "agentguard.decorators",
    "agentguard.exceptions",
    "agentguard.guard",
    "agentguard.loop",
    "agentguard.pricing",
    "agentguard.report",
    "agentguard.tracker",
)

# Guard against a refactor quietly deleting most examples: if this number drops,
# someone removed documentation without noticing.
_MINIMUM_EXAMPLES = 20


class _DocstringChecks:
    """Mixin holding the actual check.

    Deliberately *not* a :class:`unittest.TestCase`: both unittest and pytest
    collect every TestCase subclass in a module, so a TestCase base class would
    itself be run (with an empty ``module_name``) and fail.
    """

    module_name: str = ""

    def test_docstring_examples(self) -> None:
        module = importlib.import_module(self.module_name)
        captured = io.StringIO()
        with redirect_stdout(captured):
            results = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
        self.assertEqual(
            results.failed,
            0,
            f"{results.failed} of {results.attempted} examples failed in "
            f"{self.module_name}:\n{captured.getvalue()}",
        )


def _install_cases() -> None:
    for name in _MODULE_NAMES:
        class_name = "TestDocstrings_" + name.replace(".", "_")
        globals()[class_name] = type(
            class_name,
            (_DocstringChecks, unittest.TestCase),
            {"module_name": name},
        )


_install_cases()


class DocstringCoverageTests(unittest.TestCase):
    def test_enough_examples_are_still_being_collected(self) -> None:
        total = 0
        for name in _MODULE_NAMES:
            module = importlib.import_module(name)
            with redirect_stdout(io.StringIO()):
                total += doctest.testmod(module, optionflags=doctest.ELLIPSIS).attempted
        self.assertGreaterEqual(
            total,
            _MINIMUM_EXAMPLES,
            f"only {total} docstring examples were collected; expected at least "
            f"{_MINIMUM_EXAMPLES}",
        )


class DocumentedExampleTests(unittest.TestCase):
    """Examples in ``docs/`` carry expected output, so they can be executed too."""

    #: If the block-extraction regex stops matching, the test below would pass by
    #: finding nothing — so the count is asserted as well.
    MINIMUM_BLOCKS = 4
    MINIMUM_EXAMPLES = 8

    def _blocks(self) -> list[str]:
        text = (_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
        return [
            body for body in re.findall(r"```python\n(.*?)```", text, re.DOTALL) if ">>>" in body
        ]

    @staticmethod
    def _namespace(blocks: list[str]) -> dict[str, object]:
        """What a reader would have in scope after following the document.

        The blocks are fragments — only the first carries its imports — so the
        globals are assembled from every ``>>> import`` in the file plus the whole
        public API. Without this the examples would fail on missing names, which
        says nothing about whether their claims are true.
        """
        namespace: dict[str, object] = {}
        import agentguard

        for name in agentguard.__all__:
            namespace[name] = getattr(agentguard, name)

        statements: list[str] = []
        for block in blocks:
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith(">>> "):
                    stripped = stripped[4:]
                if stripped.startswith(("import ", "from ")):
                    statements.append(stripped)
        if statements:
            # Only ``import`` statements collected from our own document.
            exec("\n".join(statements), namespace)
        return namespace

    def test_api_reference_examples_execute_as_documented(self) -> None:
        parser = doctest.DocTestParser()
        blocks = self._blocks()
        self.assertGreaterEqual(
            len(blocks),
            self.MINIMUM_BLOCKS,
            f"only {len(blocks)} example block(s) found in docs/API.md; the "
            f"extraction or the document changed shape",
        )
        namespace = self._namespace(blocks)

        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        attempted = 0
        for index, body in enumerate(blocks):
            test = parser.get_doctest(
                body, dict(namespace), f"docs/API.md block {index + 1}", "API.md", 0
            )
            runner.run(test, out=lambda _: None)
            attempted += len(test.examples)

        self.assertEqual(
            runner.failures,
            0,
            f"{runner.failures} example(s) in docs/API.md do not produce the output "
            f"the reference claims",
        )
        self.assertGreaterEqual(
            attempted,
            self.MINIMUM_EXAMPLES,
            f"only {attempted} example(s) were executed across {len(blocks)} block(s)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
