"""Tests for what the published release actually contains.

The wheel and the sdist are what users receive, and an sdist is immutable once it
is on PyPI. These tests pin the packaging policy: an sdist carries the library, its
tests and its examples — what someone needs to *use* or *package* agent-guard — and
not the scaffolding that maintains the repository around it.

They are cheap config assertions by design. The thing worth testing is not
hatchling's behaviour but the decision, which can otherwise be reversed by a
one-line edit that nothing would question.

``pyproject.toml`` is read with a small line scanner rather than ``tomllib``: that
module is Python 3.11+, and this project supports 3.10, so importing it would break
the suite on a supported interpreter. The scanner understands the exact shapes used
below and fails loudly on anything else rather than quietly reading nothing.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_TEXT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PYPROJECT_LINES = PYPROJECT_TEXT.splitlines()

#: Exactly what a source archive ships. Adding an entry is a decision about what
#: users receive, so it should require editing this list on purpose.
EXPECTED_SDIST = [
    "/src",
    "/tests",
    "/examples",
    "/docs",
    "/README.md",
    "/README.zh-CN.md",
    "/LICENSE",
    "/CHANGELOG.md",
    "/CONTRIBUTING.md",
    "/SECURITY.md",
    "/CODE_OF_CONDUCT.md",
    "/ROADMAP.md",
]

#: Directories that must never reach a release: maintaining the repository is not
#: part of the library.
NON_CORE_DIRECTORIES = ("tools", ".github", "build", "dist", "scripts", "assets")


def array_at(table_marker: str, key: str) -> list[str]:
    """Read a top-level ``key = [...]`` array from the table starting at ``marker``.

    Handles only the single-line-per-element form the file uses. A reformat that
    breaks that assumption raises here, which is the point: a silent empty read
    would turn every assertion below into a no-op.
    """
    start = next(
        (i for i, line in enumerate(PYPROJECT_LINES) if line.strip() == table_marker),
        None,
    )
    if start is None:
        raise AssertionError(f"no {table_marker!r} table in pyproject.toml")

    for index in range(start + 1, len(PYPROJECT_LINES)):
        line = PYPROJECT_LINES[index]
        stripped = line.strip()
        if stripped.startswith("[") and not stripped.startswith("[" * 1 + '"'):
            break  # left this table
        if not stripped.startswith(f"{key} = ["):
            continue

        # Either inline (``key = ["a", "b"]``) or one element per following line.
        if stripped.endswith("]"):
            body = stripped[len(f"{key} = [") : -1]
            return [item.strip().strip('"') for item in body.split(",") if item.strip()]

        items: list[str] = []
        for following in PYPROJECT_LINES[index + 1 :]:
            entry = following.strip()
            if entry.startswith("]"):
                return items
            items.append(entry.rstrip(",").strip().strip('"'))
        raise AssertionError(f"{key!r} array is never closed")

    raise AssertionError(f"no {key!r} key in {table_marker!r}")


class WheelContentsTests(unittest.TestCase):
    def test_the_wheel_ships_only_the_package(self) -> None:
        self.assertEqual(
            array_at("[tool.hatch.build.targets.wheel]", "packages"), ["src/agentguard"]
        )

    def test_the_package_is_marked_as_typed(self) -> None:
        # Shipping py.typed is what makes the annotations useful to a consumer.
        self.assertTrue(
            (ROOT / "src" / "agentguard" / "py.typed").is_file(),
            "src/agentguard/py.typed is missing, so the py.typed marker is not shipped",
        )

    def test_declared_runtime_dependencies_stay_empty(self) -> None:
        # CI asserts this on the built wheel too; asserting it here means a
        # contributor finds out before pushing rather than from a red job.
        self.assertEqual(array_at("[project]", "dependencies"), [])


class SdistContentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.include = array_at("[tool.hatch.build.targets.sdist]", "include")

    def test_include_list_is_exactly_the_intended_one(self) -> None:
        self.assertEqual(sorted(self.include), sorted(EXPECTED_SDIST))

    def test_no_non_core_directory_is_published(self) -> None:
        for entry in self.include:
            top = entry.strip("/").split("/")[0]
            with self.subTest(entry=entry):
                self.assertNotIn(
                    top,
                    NON_CORE_DIRECTORIES,
                    f"{entry!r} would publish {top!r}, which maintains the repository "
                    f"rather than being part of the library",
                )

    def test_every_included_path_exists(self) -> None:
        # A stale entry here silently ships nothing, which is worse than an error.
        for entry in self.include:
            with self.subTest(entry=entry):
                self.assertTrue((ROOT / entry.lstrip("/")).exists(), f"{entry} does not exist")

    def test_the_policies_a_downstream_packager_needs_are_included(self) -> None:
        for name in ("/LICENSE", "/README.md", "/CHANGELOG.md", "/CONTRIBUTING.md"):
            with self.subTest(name=name):
                self.assertIn(name, self.include)

    def test_docs_carries_only_real_documentation(self) -> None:
        # docs/ is included wholesale, so anything dropped in it ships. This is the
        # check that would have caught the social-preview card, which was maintainer
        # artwork reachable only by a script that is not part of the library.
        docs = sorted(p.name for p in (ROOT / "docs").iterdir() if p.is_file())
        self.assertEqual(docs, ["API.md", "DETAILS.md", "demo.svg"])

    def test_readme_only_embeds_assets_that_ship(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        embedded = re.findall(r'<img[^>]+src="([^"]+)"', readme)
        embedded += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
        self.assertTrue(embedded, "the README no longer embeds any image")
        for source in embedded:
            with self.subTest(source=source):
                if source.startswith("http"):
                    continue
                self.assertTrue(
                    (ROOT / source.lstrip("/")).exists(),
                    f"the README embeds {source!r}, which is not in the repository",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
