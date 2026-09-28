"""Command line interface: render a saved report, or inspect the price table.

.. code-block:: console

   $ agent-guard report run.json
   $ agent-guard report run.json --json
   $ agent-guard pricing gpt-4o
   $ agent-guard pricing

The ``report`` subcommand exists so that a long-running job can dump its guard
report to JSON and something else — CI, a cron job, a human — can read it later
without importing agent-guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ._util import format_usd
from ._version import __version__
from .pricing import PRICING_AS_OF, PriceTable
from .report import Report

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-guard",
        description="Inspect agent-guard cost reports and bundled model prices.",
    )
    parser.add_argument("--version", action="version", version=f"agent-guard {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    report = sub.add_parser(
        "report",
        help="render a JSON report written by Guard.save()",
        description="Render a JSON report written by guard.save('run.json').",
    )
    report.add_argument("path", help="path to the JSON report")
    report.add_argument("--json", action="store_true", help="re-print as normalised JSON")
    report.add_argument(
        "--ascii",
        action="store_true",
        help="use ASCII instead of box-drawing characters",
    )
    report.set_defaults(func=_cmd_report)

    pricing = sub.add_parser(
        "pricing",
        help="show the bundled model price table",
        description="Show bundled prices, or the detail and example costs for one model.",
    )
    pricing.add_argument("model", nargs="?", help="a model name to look up in detail")
    pricing.set_defaults(func=_cmd_pricing)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``agent-guard`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"agent-guard: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"agent-guard: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print(f"agent-guard: {path} must contain a JSON object", file=sys.stderr)
        return 2

    report = Report.from_dict(data)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render(ascii_only=True if args.ascii else None))
    return 0


def _cmd_pricing(args: argparse.Namespace) -> int:
    table = PriceTable()

    if args.model:
        resolved = table.resolve(args.model)
        if resolved is None:
            print(
                f"agent-guard: no bundled price for {args.model!r} "
                f"(snapshot {PRICING_AS_OF}).",
                file=sys.stderr,
            )
            print(
                "Add it with Guard(pricing={{"
                f"{args.model!r}: (input_per_1m, output_per_1m)}}).",
                file=sys.stderr,
            )
            return 1

        canonical, price = resolved
        print(f"{canonical}  (USD per 1M tokens, snapshot {PRICING_AS_OF})")
        print()
        print(f"  input    {format_usd(price.input_per_1m):>10} / 1M")
        print(f"  output   {format_usd(price.output_per_1m):>10} / 1M")
        if price.cached_input_per_1m is not None:
            print(f"  cached   {format_usd(price.cached_input_per_1m):>10} / 1M")
        print()
        print("  example costs")
        for label, input_tokens, output_tokens in (
            ("1M in + 1M out", 1_000_000, 1_000_000),
            ("100k in + 20k out", 100_000, 20_000),
            ("10k in + 2k out", 10_000, 2_000),
        ):
            cost = price.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)
            print(f"    {label:<20}{format_usd(cost):>12}")
        return 0

    print(f"{len(table)} models bundled (USD per 1M tokens, snapshot {PRICING_AS_OF})")
    print()
    print(f"  {'model':<24}{'input':>10}{'output':>10}{'cached':>10}")
    for name, price in sorted(table.items()):
        cached = (
            format_usd(price.cached_input_per_1m)
            if price.cached_input_per_1m is not None
            else "-"
        )
        print(
            f"  {name:<24}"
            f"{format_usd(price.input_per_1m):>10}"
            f"{format_usd(price.output_per_1m):>10}"
            f"{cached:>10}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
