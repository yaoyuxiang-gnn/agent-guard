"""Produce the report agent-guard prints at the end of a realistic run.

Run it::

    python examples/report_demo.py
    python examples/report_demo.py --json

Deliberately includes an unpriced model, because the honest handling of a model
whose price agent-guard does not know is the part most cost libraries get wrong.
"""

from __future__ import annotations

import argparse

from agentguard import Guard


def build_guard() -> Guard:
    """Simulate a nightly indexing job that fans out across three models."""
    guard = Guard(
        max_usd=0.60,
        max_tokens=500_000,
        max_steps=25,
        name="nightly-indexer",
    )

    with guard:
        for page in range(6):
            with guard.step(tag=f"chunk-{page}") as step:
                step.record(
                    "gpt-4o", input_tokens=18_000, output_tokens=1_400, cached_input_tokens=6_000
                )
        for page in range(6, 12):
            with guard.step(tag=f"label-{page}") as step:
                step.record("gpt-4o-mini", input_tokens=9_000, output_tokens=700)
        # A fine-tune with no public price: counted, never guessed at.
        with guard.step(tag="rerank") as step:
            step.record("acme-rerank-v3", input_tokens=40_000, output_tokens=0)

    return guard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    guard = build_guard()
    report = guard.report()

    if args.json:
        print(guard.to_json())
    else:
        print(report.render())


if __name__ == "__main__":
    main()
