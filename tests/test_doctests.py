"""Run every docstring example in the package as a test.

Docstring examples are the first code anyone copies out of the repository, so they
are part of the test suite rather than decoration.

The cases below are generated as ordinary :class:`unittest.TestCase` subclasses
rather than through the ``load_tests`` protocol, because pytest does not implement
``load_tests`` — with it, these tests would silently collect as zero under pytest
while passing under ``python -m unittest``.
"""

from __future__ import annotations

import doctest
import importlib
import io
import unittest
from contextlib import redirect_stdout

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
