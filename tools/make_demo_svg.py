"""Render ``docs/demo.svg`` from the real output of ``examples/basic.py``.

The demo image in the README is generated, not hand-drawn: this script runs the
example, captures its actual stdout, and turns it into an animated terminal
recording as a self-contained SVG.

That matters for a project whose whole claim is "the numbers are real". A
screenshot can be staged; a generator that re-runs the example cannot drift from
what the code actually prints.

Usage::

    python tools/make_demo_svg.py

No dependencies beyond the standard library, same as everything else here.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "basic.py"
TARGET = ROOT / "docs" / "demo.svg"

# Geometry. CHAR_WIDTH is the advance width of the monospace stack below at
# FONT_SIZE; it only has to be close, since nothing is right-aligned in SVG text.
FONT_SIZE = 13
CHAR_WIDTH = 7.81
LINE_HEIGHT = 19
BAR_HEIGHT = 34
PAD_X = 18
PAD_TOP = 14
PAD_BOTTOM = 18

# Timing.
STEP_SECONDS = 0.28
HOLD_SECONDS = 4.5

FONT_STACK = (
    "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
    "'Liberation Mono', monospace"
)

PALETTE = {
    "text": "#c9d1d9",
    "dim": "#6e7681",
    "rule": "#30363d",
    "title": "#58a6ff",
    "label": "#8b949e",
    "money": "#3fb950",
    "bad": "#f85149",
}

_MONEY = re.compile(r"(\$[\d.,]+)")
_RULE = re.compile(r"^[=\-─═\s]+$")


def classify(line: str) -> str:
    """Pick a colour for one output line, from its content alone."""
    stripped = line.strip()
    if not stripped:
        return "dim"
    if stripped.startswith("agent-guard"):
        return "title"
    if _RULE.match(stripped):
        return "rule"
    if "STOPPED" in stripped or stripped.startswith("!"):
        return "bad"
    if stripped.startswith(("limits", "by model", "tripped:")):
        return "label"
    if "$" in stripped:
        return "money"
    return "text"


def spans(line: str, kind: str) -> list[tuple[str, str]]:
    """Split a line into coloured runs, so dollar amounts stand out.

    Only the ``money`` kind is split; ``bad`` lines stay uniformly red because the
    colour *is* the signal there and a green figure inside a red row reads as a
    contradiction.
    """
    if kind == "money":
        runs: list[tuple[str, str]] = []
        for index, part in enumerate(_MONEY.split(line)):
            if not part:
                continue
            runs.append((part, PALETTE["money"] if index % 2 else PALETTE["text"]))
        return runs
    return [(line, PALETTE[kind])]


def capture() -> list[str]:
    """Run the example and return its stdout as a list of lines."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    # Force ASCII so the captured text is identical on every machine, and the
    # SVG does not depend on the box-drawing glyphs being present in the font.
    env["AGENT_GUARD_ASCII"] = "1"

    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"examples/basic.py exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    if not result.stdout.strip():
        raise SystemExit("examples/basic.py produced no output")
    return result.stdout.rstrip("\n").split("\n")


def render(lines: list[str]) -> str:
    """Build the animated SVG document."""
    longest = max(len(line) for line in lines)
    width = round(longest * CHAR_WIDTH) + 2 * PAD_X
    body_top = BAR_HEIGHT + PAD_TOP
    height = body_top + len(lines) * LINE_HEIGHT + PAD_BOTTOM

    duration = len(lines) * STEP_SECONDS + HOLD_SECONDS
    last_start = max(0.0, (len(lines) - 1) * STEP_SECONDS)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="agent-guard stopping an agent that exceeded its budget">',
        "<title>agent-guard stopping an agent that exceeded its budget</title>",
        "<style>",
        f"  text {{ font-family: {FONT_STACK}; font-size: {FONT_SIZE}px; "
        f"white-space: pre; }}",
        "  .bar { fill: #161b22; }",
        "  .body { fill: #0d1117; }",
        "  @keyframes blink { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }",
        f"  .cursor {{ animation: blink 1.1s steps(1, end) infinite; "
        f"animation-delay: {last_start + 0.4:.2f}s; opacity: 0; }}",
        "</style>",
        # Window chrome.
        f'<rect class="bar" x="0" y="0" width="{width}" height="{BAR_HEIGHT}" rx="8"/>',
        f'<rect class="bar" x="0" y="{BAR_HEIGHT - 8}" width="{width}" height="8"/>',
        f'<rect class="body" x="0" y="{BAR_HEIGHT}" width="{width}" '
        f'height="{height - BAR_HEIGHT}" rx="0"/>',
        f'<rect class="body" x="0" y="{height - 8}" width="{width}" height="8" rx="8"/>',
        '<circle class="dot" cx="20" cy="17" r="5" fill="#ff5f57"/>',
        '<circle class="dot" cx="38" cy="17" r="5" fill="#febc2e"/>',
        '<circle class="dot" cx="56" cy="17" r="5" fill="#28c840"/>',
        f'<text x="{width / 2:.0f}" y="21" text-anchor="middle" fill="#8b949e" '
        f'font-size="11">python examples/basic.py</text>',
    ]

    for index, line in enumerate(lines):
        y = body_top + index * LINE_HEIGHT
        # keyTimes must be non-decreasing and span 0..1; a floor keeps the first
        # line from producing a degenerate "0;0;1" timeline.
        fraction = min(0.999, max(0.001, (index * STEP_SECONDS) / duration))
        runs = spans(line, classify(line))

        # Leading spaces must survive, hence xml:space="preserve".
        parts.append(
            f'<text xml:space="preserve" x="{PAD_X}" y="{y:.0f}" opacity="0">'
            f'<animate attributeName="opacity" dur="{duration:.2f}s" '
            f'repeatCount="indefinite" calcMode="discrete" '
            f'values="0;1;1" keyTimes="0;{fraction:.4f};1"/>'
        )
        for text, colour in runs:
            parts.append(f'<tspan fill="{colour}">{html.escape(text)}</tspan>')
        parts.append("</text>")

    # Blinking prompt at the foot of the terminal.
    prompt_y = body_top + len(lines) * LINE_HEIGHT
    parts.append(
        f'<text xml:space="preserve" x="{PAD_X}" y="{prompt_y:.0f}">'
        f'<tspan fill="{PALETTE["money"]}">$ </tspan>'
        f'<tspan class="cursor" fill="{PALETTE["text"]}">_</tspan>'
        f"</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    lines = capture()
    svg = render(lines)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the generated file byte-identical on Windows and Linux.
    # Without it, Windows writes CRLF and every regeneration shows up as a diff
    # until .gitattributes normalises it back.
    TARGET.write_text(svg, encoding="utf-8", newline="\n")
    print(
        f"wrote {TARGET.relative_to(ROOT)} "
        f"({len(lines)} lines, {len(svg):,} bytes, "
        f"{len(lines) * STEP_SECONDS + HOLD_SECONDS:.1f}s loop)"
    )


if __name__ == "__main__":
    main()
