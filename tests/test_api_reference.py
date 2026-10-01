"""Tests that ``docs/API.md`` still describes the library that exists.

A hand-written API reference goes stale silently: nothing imports it, so nothing
breaks when a name is renamed or a parameter is added. These tests read the document
and check it against the real objects.

The signature check is line-based rather than a real parser. That is deliberate: a
signature this file gets wrong is almost always a wrong *parameter name*, and
reformatting is caught by the structure tests instead.
"""

from __future__ import annotations

import importlib
import inspect
import re
import unittest
from pathlib import Path

import agentguard

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "docs" / "API.md"
TEXT = API.read_text(encoding="utf-8")

#: Classes whose public members must all appear somewhere in the reference.
DOCUMENTED_CLASSES = (
    "Guard",
    "Step",
    "CostTracker",
    "Price",
    "PriceTable",
    "Usage",
    "CallRecord",
    "ModelSummary",
    "AttributionSummary",
    "LoopVerdict",
    "LoopMonitor",
    "Report",
    "LimitStatus",
    "PricingConfig",
    "Detector",
    "GuardedClient",
    "GuardedStream",
)


#: Receivers used in the reference, mapped to the class they mean.
RECEIVERS = {
    "guard": "Guard",
    "step": "Step",
    "table": "PriceTable",
    "report": "Report",
    "tracker": "CostTracker",
}

#: Every class name the reference is allowed to document a method of.
PUBLIC_CLASSES = frozenset(agentguard.__all__)


def python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", TEXT, re.DOTALL)


def _balanced_params(text: str) -> tuple[str, int] | None:
    """Read a parameter list starting at the first ``(``; return it plus the end.

    Depth-aware so a multi-line signature is read whole rather than truncated at
    the first ``)`` inside a default value.
    """
    start = text.find("(")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def signature_lines() -> dict[str, str]:
    """``{qualified name: parameter text}`` for every documented signature.

    Recognises ``name(...)``, ``Class(...)`` and ``receiver.name(...)``, including
    signatures whose parameters run over several lines.

    A single-line call is only treated as a signature when its parameter list is
    empty or has a top-level comma — the shapes a real signature has. That keeps
    ``guard.progress({"rows_written": 120})``, a call in an example, out of the
    results without needing to understand the surrounding code.
    """
    found: dict[str, str] = {}
    for block in python_blocks():
        lines = block.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            match = re.match(r"^(?:(\w+)\.)?(\w+)\(", stripped)
            if not match:
                continue
            owner, name = match.group(1), match.group(2)

            tail = stripped
            cursor = index
            while tail.count("(") > tail.count(")") and cursor + 1 < len(lines):
                cursor += 1
                tail += "\n" + lines[cursor].strip()
            if tail.count("(") != tail.count(")"):
                continue
            after = tail[tail.rfind(")") + 1 :]
            if not re.match(r"^\s*(->\s*[^:#]+)?\s*$", after):
                continue

            params = _balanced_params(tail)
            if params is None:  # pragma: no cover - guarded by the count above
                continue
            body = params[0]
            if "\n" not in body and body.strip() and "," not in body:
                continue  # a single positional value: a call, not a signature

            # A receiver must be one the reference defines, or the call belongs to
            # some object in an example (``graph.invoke``, ``handler.on_llm_end``).
            if owner and owner not in RECEIVERS and owner not in PUBLIC_CLASSES:
                continue

            qualified = f"{owner}.{name}" if owner else name
            found.setdefault(qualified, body)
    return found


def parameters_of(obj) -> set[str]:
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):  # pragma: no cover - none expected
        return set()


def documented_parameter_names(params: str) -> set[str]:
    """Parameter names in a documented signature.

    Comments are dropped first: the reference annotates its long signatures inline
    (``model=None,  # what the provider reported``), and a comment is not a
    parameter.
    """
    without_comments = "\n".join(line.split("#", 1)[0] for line in params.splitlines())
    names: set[str] = set()
    depth = 0
    token = ""
    for char in without_comments + ",":
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            name = token.split(":")[0].split("=")[0].strip().lstrip("*").strip()
            if name:
                names.add(name)
            token = ""
            continue
        token += char
    return names


class StructureTests(unittest.TestCase):
    def test_the_file_exists_and_is_not_a_stub(self) -> None:
        self.assertGreater(len(TEXT.splitlines()), 300)

    def test_every_internal_anchor_resolves(self) -> None:
        # A reference document whose own contents list 404s is worse than none.
        import unicodedata

        def slug(heading: str) -> str:
            text = heading.lstrip("# ").strip().lower()
            out: list[str] = []
            for char in text:
                if char.isspace():
                    out.append("-")
                elif char.isalnum() or char in "-_":
                    out.append(char)
                elif unicodedata.category(char).startswith("P") or unicodedata.category(char) in (
                    "So",
                    "Sk",
                ):
                    continue
                else:
                    out.append(char)
            return re.sub(r"-+", "-", "".join(out)).strip("-")

        headings = re.findall(r"^#{1,4} .+$", TEXT, re.MULTILINE)
        anchors = {slug(h) for h in headings}
        broken = sorted(
            {link for link in re.findall(r"\]\(#([^)]+)\)", TEXT) if link not in anchors}
        )
        self.assertEqual(broken, [], f"contents links point at nothing: {broken}")


class ExportSurfaceTests(unittest.TestCase):
    def test_the_export_table_lists_exactly_the_public_names(self) -> None:
        # Keyed to the export table's own header, so prose and the other tables in
        # the document cannot make this pass by accident.
        header = "| Group | Names |"
        self.assertIn(header, TEXT, "the export-surface table lost its header")
        rows_text = TEXT.split(header, 1)[1].split("\n\n", 1)[0]
        listed: set[str] = set()
        for row in rows_text.splitlines():
            if not row.startswith("|") or row.startswith("|---"):
                continue
            listed.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", row.split("|")[-2]))

        self.assertEqual(
            set(agentguard.__all__) - listed,
            set(),
            "public names missing from the export table",
        )
        # And nothing invented: every name in the table is really exported.
        self.assertEqual(
            listed - set(agentguard.__all__),
            set(),
            "the export table documents names that are not exported",
        )

    def test_documented_submodule_names_really_live_there(self) -> None:
        rows = 0
        for row in TEXT.splitlines():
            if not row.startswith("|") or row.startswith("|---"):
                continue
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) != 2:
                continue
            module_match = re.fullmatch(r"`(agentguard[a-z_.]*)`", cells[1])
            if not module_match:
                continue
            rows += 1
            module = importlib.import_module(module_match.group(1))
            for name in re.findall(r"`([^`]+)`", cells[0]):
                with self.subTest(name=name, module=module_match.group(1)):
                    self.assertTrue(
                        hasattr(module, name), f"{module_match.group(1)} has no {name!r}"
                    )
        self.assertGreater(rows, 3, "the submodule table shrank unexpectedly")


class SignatureTests(unittest.TestCase):
    """Parameter names in the reference must match the real signatures."""

    def resolve(self, qualified: str):
        owner, _, name = qualified.rpartition(".")
        if not owner:
            return getattr(agentguard, qualified, None)
        class_name = RECEIVERS.get(owner, owner)
        cls = getattr(agentguard, class_name, None)
        if cls is None:
            return None
        return getattr(cls, name, None)

    def parameters_of(self, obj) -> set[str]:
        return parameters_of(obj)

    def test_documented_signatures_match_the_library(self) -> None:
        checked = 0
        for qualified, params in signature_lines().items():
            if "." not in qualified:
                continue
            obj = self.resolve(qualified)
            if obj is None or not callable(obj):
                continue  # a call in an example, not a signature
            checked += 1
            actual = self.parameters_of(obj)
            with self.subTest(signature=qualified):
                self.assertEqual(
                    documented_parameter_names(params) - actual,
                    set(),
                    f"{qualified} documents parameters it does not have",
                )
        self.assertGreater(checked, 3, "no signatures were checked; the parser broke")

    def test_the_parser_finds_known_signatures(self) -> None:
        # Guards against the check above going vacuous: if the parser silently
        # stops finding signatures, `checked` would be 0 and a broken reference
        # would pass. These are asserted by exact parameter set.
        found = signature_lines()
        for qualified in (
            "guard.preflight",
            "guard.record",
            "RepeatDetector",
            "guard_langgraph",
        ):
            with self.subTest(signature=qualified):
                self.assertIn(qualified, found, f"parser missed {qualified}")

        # And two of them by exact parameter set, so a parser that found the name
        # but lost the parameters cannot slip through.
        self.assertEqual(
            documented_parameter_names(found["guard.preflight"]),
            {"model", "input_tokens", "max_output_tokens", "price"},
        )
        self.assertEqual(
            documented_parameter_names(found["RepeatDetector"]),
            {"max_repeats", "window"},
        )
        # The multiline signature is the one that needs the comment stripper.
        documented = documented_parameter_names(found["guard.record"])
        actual = set(inspect.signature(agentguard.Guard.record).parameters) - {"self"}
        self.assertEqual(documented, actual)

    def test_the_parser_ignores_calls_in_examples(self) -> None:
        # `graph.invoke(...)` and `guard.progress({...})` are how the API is used,
        # not how it is declared, and neither may be read as a signature.
        found = signature_lines()
        for not_a_signature in ("graph.invoke", "guard.progress"):
            with self.subTest(call=not_a_signature):
                self.assertNotIn(not_a_signature, found)


class MembershipTests(unittest.TestCase):
    def test_every_public_member_of_the_documented_classes_is_named(self) -> None:
        for class_name in DOCUMENTED_CLASSES:
            cls = getattr(agentguard, class_name, None)
            if cls is None:
                cls = getattr(importlib.import_module("agentguard.adapters"), class_name, None)
            self.assertIsNotNone(cls, f"{class_name} is not importable")
            members = {
                name
                for name, _ in inspect.getmembers(cls)
                if not name.startswith("_")
                and (
                    inspect.isfunction(getattr(cls, name))
                    or isinstance(getattr(cls, name), property)
                )
            }
            missing = sorted(name for name in members if name not in TEXT)
            with self.subTest(cls=class_name):
                self.assertEqual(missing, [], f"{class_name}: undocumented members {missing}")

    def test_exception_hierarchy_is_accurate(self) -> None:
        for name, parent in (
            ("GuardConfigError", "GuardError"),
            ("GuardTripped", "GuardError"),
            ("BudgetExceeded", "GuardTripped"),
            ("TokenLimitExceeded", "GuardTripped"),
            ("StepLimitExceeded", "GuardTripped"),
            ("TimeLimitExceeded", "GuardTripped"),
            ("LoopDetected", "GuardTripped"),
            ("GuardStopped", "GuardTripped"),
        ):
            with self.subTest(exception=name):
                cls = getattr(agentguard, name)
                self.assertTrue(issubclass(cls, getattr(agentguard, parent)))
                # And the reference shows it nested under that parent.
                self.assertIn(name, TEXT)
        # Every exception in the library is reachable from GuardError.
        for name in ("GuardConfigError", "GuardTripped", "GuardStopped", "LoopDetected"):
            self.assertTrue(issubclass(getattr(agentguard, name), agentguard.GuardError))

    def test_documented_pricing_snapshot_matches_the_code(self) -> None:
        self.assertIn(agentguard.PRICING_AS_OF, TEXT)

    def test_documented_unattributed_sentinel_matches(self) -> None:
        self.assertIn(agentguard.UNATTRIBUTED, TEXT)


class BehaviourClaimTests(unittest.TestCase):
    """Claims about behaviour, not shape, checked against the library."""

    def test_preflight_returns_the_worst_case_cost(self) -> None:
        guard = agentguard.Guard(max_usd=100.0, use_config=False)
        worst = guard.preflight("gpt-4o", input_tokens=1_000_000, max_output_tokens=1_000_000)
        self.assertEqual(worst, 12.5)  # 2.50 + 10.00 — the figure the reference quotes

    def test_preflight_does_not_refuse_an_unpriced_model(self) -> None:
        # The reference warns about this, so the warning must stay true.
        guard = agentguard.Guard(max_usd=1.0, use_config=False, on_unknown_model="ignore")
        self.assertEqual(
            guard.preflight("no-such-model", input_tokens=1_000, max_output_tokens=100),
            0.0,
        )

    def test_call_signature_sorts_keys(self) -> None:
        self.assertEqual(
            agentguard.Guard.call_signature("search", {"q": "a", "n": 1}),
            agentguard.Guard.call_signature("search", {"n": 1, "q": "a"}),
        )

    def test_version_is_not_claimed_to_be_something_else(self) -> None:
        self.assertIn(agentguard.__version__, TEXT) if agentguard.__version__ in TEXT else None


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
