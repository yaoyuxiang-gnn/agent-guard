"""Command line interface: render a report, inspect prices, edit the config.

.. code-block:: console

   $ agentguard report run.json
   $ agentguard report run.json --json
   $ agentguard pricing gpt-4o
   $ agentguard pricing
   $ agentguard config path
   $ agentguard config set my-finetune-v3 --input 3 --output 12
   $ agentguard config alias acme/fast claude-3-5-haiku
   $ agentguard config disable gpt-4
   $ agentguard config list

The ``report`` subcommand exists so that a long-running job can dump its guard
report to JSON and something else — CI, a cron job, a human — can read it later
without importing agent-guard.

The ``config`` subcommand is the same pricing configuration
:class:`~agentguard.Guard` discovers at import time, written from the terminal
instead of by hand: a JSON file of custom models, their prices, the aliases your
gateway reports, and the bundled prices you refuse to trust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from ._util import format_usd
from ._version import __version__
from .config import (
    CONFIG_ENV_VAR,
    CONFIG_TRUST_ENV_VAR,
    PricingConfig,
    config_paths,
    ignored_project_config,
    initialize_config,
    load_config,
    project_config_path,
    project_config_trusted,
    remove_entry,
    set_alias,
    set_disabled,
    set_model_price,
    user_config_path,
)
from .exceptions import GuardConfigError
from .pricing import PRICING_AS_OF, Price, PriceTable, normalize_model_key
from .report import Report

__all__ = ["main"]

#: The library's "I skipped an untrusted project file" warning, matched so the CLI
#: can replace it with a printed note rather than a stack-level warning.
_IGNORED_CONFIG_WARNING = "agentguard is ignoring the project config"


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentguard",
        description="Inspect agentguard cost reports and model prices.",
        epilog=(
            "Prices come from a bundled snapshot plus an optional JSON config file. "
            "Run `agentguard config path` to see which file is in use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"agentguard {__version__}")
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
        help="show the effective model price table",
        description=(
            "Show the effective prices — bundled models, plus anything your config "
            "file adds, reprices, aliases or disables — or the detail and example "
            "costs for one model."
        ),
    )
    pricing.add_argument("model", nargs="?", help="a model name to look up in detail")
    pricing.add_argument(
        "--no-config",
        action="store_true",
        help="ignore the config file and show only bundled prices",
    )
    pricing.add_argument("--json", action="store_true", help="print the table as JSON")
    pricing.set_defaults(func=_cmd_pricing)

    config = sub.add_parser(
        "config",
        help="inspect or edit the pricing config file",
        description=(
            "Inspect or edit the JSON file that lists your own models and prices. "
            "Guard() reads it automatically."
        ),
        epilog=(
            "examples:\n"
            "  agentguard config set my-finetune-v3 3 12\n"
            "  agentguard config set acme-local-7b 0.05 0.08 --cached 0.01\n"
            "  agentguard config alias acme/fast claude-3-5-haiku\n"
            "  agentguard config disable gpt-4\n"
            "  agentguard config remove my-finetune-v3\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config.set_defaults(func=_cmd_config_help, parser=config)
    actions = config.add_subparsers(dest="config_action", metavar="ACTION")

    where = argparse.ArgumentParser(add_help=False)
    where.add_argument("--user", action="store_true", help="use the per-user config file")
    where.add_argument("--project", action="store_true", help="use ./agentguard.json")
    where.add_argument("--file", type=Path, metavar="PATH", help="use an explicit file")

    action = actions.add_parser(
        "path",
        help="show where config is read from",
        description="Show every config location, which one is in effect, and why.",
    )
    action.set_defaults(func=_cmd_config_path)

    action = actions.add_parser(
        "init",
        help="create a starter config file",
        parents=[where],
        description="Write an empty but valid config file.",
    )
    action.add_argument("--force", action="store_true", help="overwrite an existing file")
    action.set_defaults(func=_cmd_config_init)

    action = actions.add_parser(
        "set",
        help="add or update a model price",
        parents=[where],
        description=(
            "Add a model price, in USD per 1M tokens. Works for models agentguard "
            "does not know and for repricing bundled ones."
        ),
    )
    action.add_argument("name", help="model name as your provider reports it")
    action.add_argument("input", type=float, help="USD per 1M input tokens")
    action.add_argument("output", type=float, help="USD per 1M output tokens")
    action.add_argument(
        "--cached",
        type=float,
        default=None,
        metavar="USD",
        help="USD per 1M cached input tokens (default: bill cache at the input rate)",
    )
    action.set_defaults(func=_cmd_config_set)

    action = actions.add_parser(
        "alias",
        help="point a reported name at a priced model",
        parents=[where],
        description="Resolve a name your gateway reports to a model that has a price.",
    )
    action.add_argument("name", help="the name as reported")
    action.add_argument("target", help="the model whose price should be used")
    action.set_defaults(func=_cmd_config_alias)

    action = actions.add_parser(
        "remove",
        help="remove a model or alias",
        parents=[where],
        description=(
            "Remove a model, an alias, or a disable entry. Aliases pointing at a "
            "removed model are removed with it."
        ),
    )
    action.add_argument("name")
    action.set_defaults(func=_cmd_config_remove)

    action = actions.add_parser(
        "disable",
        help="stop trusting a bundled price",
        parents=[where],
        description=(
            "Make a bundled model unpriced instead of billed at a price you do not "
            "trust. Unpriced calls are reported and excluded from the budget, never "
            "guessed at."
        ),
    )
    action.add_argument("name")
    action.set_defaults(func=_cmd_config_disable)

    action = actions.add_parser(
        "enable",
        help="restore a disabled bundled price",
        parents=[where],
        description="Undo `agentguard config disable`.",
    )
    action.add_argument("name")
    action.set_defaults(func=_cmd_config_enable)

    action = actions.add_parser(
        "list",
        help="show what the config files contain",
        description="Show the contents of every config file that is in effect.",
    )
    action.add_argument("--json", action="store_true", help="print as JSON")
    action.set_defaults(func=_cmd_config_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``agentguard`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except GuardConfigError as exc:
        print(f"agentguard: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"agentguard: {exc}", file=sys.stderr)
        return 2


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"agentguard: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"agentguard: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print(f"agentguard: {path} must contain a JSON object", file=sys.stderr)
        return 2

    report = Report.from_dict(data)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render(ascii_only=True if args.ascii else None))
    return 0


# --------------------------------------------------------------------------- #
# pricing
# --------------------------------------------------------------------------- #


def _effective_config(*, use_config: bool) -> PricingConfig:
    """Load config for a read-only command.

    The library warns when it skips an untrusted project file; a CLI command
    reports that in the output the user asked for instead of emitting a warning
    they did not ask for, so the warning is suppressed here and printed as a note.
    """
    if not use_config:
        return PricingConfig()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_IGNORED_CONFIG_WARNING, category=RuntimeWarning)
        return load_config()


def _print_ignored_config_note() -> None:
    ignored = ignored_project_config()
    if ignored is not None:
        print()
        print(f"  note: {ignored} exists but is not trusted, so it is not read")
        print(f"        set {CONFIG_TRUST_ENV_VAR}=1 to use it")


def _rows(table: PriceTable) -> list[tuple[str, Price, str]]:
    return [(name, price, table.origin(name) or "builtin") for name, price in sorted(table.items())]


def _cmd_pricing(args: argparse.Namespace) -> int:
    config = _effective_config(use_config=not args.no_config)
    table = PriceTable.from_config(config)

    if args.json:
        return _pricing_json(table, config, args.model)

    if args.model:
        return _pricing_detail(args.model, table, config)

    configured = sum(1 for _, _, origin in _rows(table) if origin != "builtin")
    summary = f"{len(table)} models bundled"
    if configured:
        summary += f", {configured} configured"
    print(f"{summary} (USD per 1M tokens, snapshot {PRICING_AS_OF})")
    print()
    print(f"  {'model':<24}{'input':>10}{'output':>10}{'cached':>10}  source")
    for name, price, origin in _rows(table):
        cached = (
            format_usd(price.cached_input_per_1m) if price.cached_input_per_1m is not None else "-"
        )
        print(
            f"  {name:<24}"
            f"{format_usd(price.input_per_1m):>10}"
            f"{format_usd(price.output_per_1m):>10}"
            f"{cached:>10}  {origin}"
        )
    _print_config_extras(config)
    _print_ignored_config_note()
    return 0


def _print_config_extras(config: PricingConfig) -> None:
    if config.aliases:
        print()
        print("  aliases")
        width = max(len(name) for name in config.aliases)
        for name, target in sorted(config.aliases.items()):
            print(f"    {name:<{width}} -> {target}")
    if config.disable:
        print()
        print("  disabled")
        for name in sorted(config.disable):
            print(f"    {name}")
    if config.sources:
        print()
        for source in config.sources:
            print(f"  config: {source}")


def _pricing_json(table: PriceTable, config: PricingConfig, model: str | None) -> int:
    """Emit the effective table as JSON, optionally narrowed to one model."""
    ignored = ignored_project_config()
    payload: dict[str, object] = {
        "snapshot": PRICING_AS_OF,
        "aliases": dict(config.aliases),
        "disabled": sorted(config.disable),
        "config_files": [str(source) for source in config.sources],
        "ignored_config": str(ignored) if ignored is not None else None,
        "models": [
            {"model": name, "source": origin, **price.as_dict()}
            for name, price, origin in _rows(table)
        ],
    }
    if model is None:
        print(json.dumps(payload, indent=2))
        return 0

    resolved = table.resolve(model)
    if resolved is None:
        print(f"agentguard: no price for {model!r}", file=sys.stderr)
        return 1
    canonical, price = resolved
    payload["models"] = [
        {"model": canonical, "source": table.origin(model) or "builtin", **price.as_dict()}
    ]
    payload["requested"] = model
    payload["alias_of"] = table.alias_of(model)
    print(json.dumps(payload, indent=2))
    return 0


def _pricing_detail(model: str, table: PriceTable, config: PricingConfig) -> int:
    alias = table.alias_of(model)
    resolved = table.resolve(model)
    if resolved is None:
        if normalize_model_key(model) in table.disabled:
            where = ", ".join(str(source) for source in config.sources) or "your config"
            print(
                f"agentguard: {model!r} is disabled in {where}, so it counts as unpriced.",
                file=sys.stderr,
            )
            print(
                f"Undo it with: agentguard config enable {model}",
                file=sys.stderr,
            )
            return 1
        print(
            f"agentguard: no bundled price for {model!r} (snapshot {PRICING_AS_OF}), "
            f"and no config entry.",
            file=sys.stderr,
        )
        print(
            f"Add it to your config: agentguard config set {model} <input> <output> "
            f"(USD per 1M tokens)",
            file=sys.stderr,
        )
        print(
            f"Or in code:            Guard(pricing={{{model!r}: (input_per_1m, output_per_1m)}})",
            file=sys.stderr,
        )
        return 1

    canonical, price = resolved
    heading = canonical if alias is None else f"{model} -> {canonical}"
    print(f"{heading}  (USD per 1M tokens, snapshot {PRICING_AS_OF})")
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
    print()
    source = table.origin(model) or "builtin"
    if alias is not None:
        print(f"  source   {source} (aliased from {model!r} to {canonical!r})")
    else:
        print(f"  source   {source}")
    return 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def _cmd_config_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def _target_file(args: argparse.Namespace) -> Path:
    """Where a write goes: an explicit flag, else the per-user file.

    Writing to the per-user file by default keeps ``config set`` predictable: it
    does not depend on the working directory, and it cannot surprise a repository
    with an unexpected new file.
    """
    chosen = [name for name in ("file", "user", "project") if getattr(args, name, None)]
    if len(chosen) > 1:
        raise GuardConfigError(
            "choose one target: --user, --project or --file (got "
            + ", ".join("--" + name for name in chosen)
            + ")"
        )
    if args.file is not None:
        return Path(args.file).expanduser()
    if args.project:
        return Path.cwd() / "agentguard.json"
    return user_config_path()


def _cmd_config_path(args: argparse.Namespace) -> int:
    user = user_config_path()
    project = project_config_path()
    explicit = (os.environ.get(CONFIG_ENV_VAR) or "").strip()
    trusted = project_config_trusted()
    ignored = ignored_project_config()

    print("config locations")
    print()
    print(f"  {'env':<8}{CONFIG_ENV_VAR}={explicit or '(not set)'}")
    print(f"  {'trust':<8}{CONFIG_TRUST_ENV_VAR}={'set' if trusted else '(not set)'}")
    print(f"  {'user':<8}{user}{'' if user.is_file() else '  (not found)'}")
    if project is None:
        print(f"  {'project':<8}{Path.cwd() / 'agentguard.json'}  (not found)")
    else:
        print(f"  {'project':<8}{project}{'' if trusted else '  (ignored: not trusted)'}")
    print()
    active = config_paths()
    if not active:
        print("  reading: bundled prices only")
    else:
        for path in active:
            print(f"  reading: {path}")
    if ignored is not None:
        print()
        print(f"  ! {ignored} was not read: a config file inside a source tree")
        print("    travels with that repository, so it could reprice models or")
        print("    disable them without you noticing. Ask for it explicitly:")
        print(f"      set {CONFIG_TRUST_ENV_VAR}=1   (or pass Guard(config_path=...))")
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    target = _target_file(args)
    written = initialize_config(target, force=args.force)
    print(f"wrote {written}")
    print("Add a model with: agentguard config set <name> <input> <output>")
    return 0


def _report_write(target: Path, description: str, *, name: str, price: Price) -> None:
    print(f"{description} in {target}")
    note = _shadow_note(name, price, target)
    if note:
        print(f"note: {note}", file=sys.stderr)


def _shadow_note(name: str, price: Price, target: Path) -> str | None:
    """Explain when a freshly written entry is not what Guard will actually use."""
    ignored = ignored_project_config()
    if ignored is not None and target == ignored:
        return (
            f"this is a project config and it is not trusted, so it is not read yet; "
            f"set {CONFIG_TRUST_ENV_VAR}=1 to use it"
        )
    try:
        effective = load_config()
    except GuardConfigError as exc:
        return f"the effective config does not load, so Guard() will fail: {exc}"
    if target not in effective.sources:
        others = ", ".join(str(source) for source in effective.sources) or "no config file"
        return (
            f"{target} is not in effect ({others}); set {CONFIG_ENV_VAR} or edit that file instead"
        )
    key = normalize_model_key(name)
    for other, candidate in effective.models.items():
        if normalize_model_key(other) == key and candidate != price:
            return f"{other!r} in a higher-precedence config file wins over this entry"
    return None


def _cmd_config_set(args: argparse.Namespace) -> int:
    target = _target_file(args)
    price = set_model_price(
        target,
        args.name,
        input_per_1m=args.input,
        output_per_1m=args.output,
        cached_input_per_1m=args.cached,
    )
    _report_write(
        target,
        f"{args.name} priced at {format_usd(price.input_per_1m)} in / "
        f"{format_usd(price.output_per_1m)} out per 1M tokens",
        name=args.name,
        price=price,
    )
    return 0


def _cmd_config_alias(args: argparse.Namespace) -> int:
    target = _target_file(args)
    set_alias(target, args.name, args.target)
    print(f"{args.name} -> {args.target} in {target}")
    return 0


def _cmd_config_remove(args: argparse.Namespace) -> int:
    target = _target_file(args)
    removal = remove_entry(target, args.name)
    if not removal.changed:
        print(f"agentguard: {args.name!r} is not in {target}", file=sys.stderr)
        return 1
    print(f"removed {removal.describe()} from {target}")
    return 0


def _cmd_config_disable(args: argparse.Namespace) -> int:
    return _set_disabled(args, disabled=True)


def _cmd_config_enable(args: argparse.Namespace) -> int:
    return _set_disabled(args, disabled=False)


def _set_disabled(args: argparse.Namespace, *, disabled: bool) -> int:
    target = _target_file(args)
    changed = set_disabled(target, args.name, disabled)
    verb = "disabled" if disabled else "enabled"
    if not changed:
        state = "already disabled" if disabled else "not disabled"
        print(f"agentguard: {args.name!r} is {state} in {target}", file=sys.stderr)
        return 1
    print(f"{verb} {args.name} in {target}")
    if disabled:
        print(
            f"note: {args.name} now counts as unpriced -- reported and excluded from "
            f"the budget rather than billed at a price you do not trust"
        )
    return 0


def _cmd_config_list(args: argparse.Namespace) -> int:
    paths = config_paths()
    ignored = ignored_project_config()
    loaded: list[tuple[Path, PricingConfig | None, str | None]] = []
    broken = False
    for path in paths:
        try:
            loaded.append((path, load_config(path), None))
        except GuardConfigError as exc:
            # A file that cannot be loaded is exactly what someone runs this
            # command to find out, so report it in place instead of failing.
            loaded.append((path, None, str(exc)))
            broken = True

    if args.json:
        print(
            json.dumps(
                {
                    "files": [
                        {"path": str(path), **(config.as_dict() if config else {}), "error": error}
                        for path, config, error in loaded
                    ],
                    "env": CONFIG_ENV_VAR,
                },
                indent=2,
            )
        )
        return 1 if broken else 0

    if not loaded:
        print("no config file found; run `agentguard config path` to see where to put one")
        if ignored is not None:
            print(f"note: {ignored} exists but is not trusted, so it is not read")
        return 0

    for index, (path, config, error) in enumerate(loaded):
        if index:
            print()
        print(f"{path}")
        if error is not None or config is None:
            print(f"  ! {error}")
            continue
        if config.is_empty:
            print("  (empty)")
            continue
        if config.models:
            print("  models")
            width = max(len(name) for name in config.models)
            for name, price in sorted(config.models.items()):
                cached = (
                    f"  cached {format_usd(price.cached_input_per_1m)}"
                    if price.cached_input_per_1m is not None
                    else ""
                )
                print(
                    f"    {name:<{width}}  "
                    f"{format_usd(price.input_per_1m)} in / "
                    f"{format_usd(price.output_per_1m)} out per 1M{cached}"
                )
        if config.aliases:
            print("  aliases")
            width = max(len(name) for name in config.aliases)
            for name, alias_target in sorted(config.aliases.items()):
                print(f"    {name:<{width}} -> {alias_target}")
        if config.disable:
            print("  disabled (counted as unpriced)")
            for name in sorted(config.disable):
                print(f"    {name}")
    if ignored is not None:
        print()
        print(f"note: {ignored} exists but is not trusted, so it is not read")
    return 1 if broken else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
