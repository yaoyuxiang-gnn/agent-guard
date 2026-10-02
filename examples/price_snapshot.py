"""Refresh the price table without waiting for a release.

Run it::

    python examples/price_snapshot.py

Runs fully offline: it imports a small catalogue from a local file, which is the
same code path ``agentguard pricing --from-file`` uses. The download path is the
one thing this example cannot show you without a network, and it is one flag away::

    agentguard pricing --update

The problem this solves: the bundled table is a snapshot dated ``PRICING_AS_OF``,
so a model released afterwards is *unpriced* — safe, because agent-guard refuses to
guess a rate, but not useful, because an unpriced call does not move the budget.

A snapshot is a second table, merged **underneath** everything you configured. It is
checksummed, verified on every read, and never fetched automatically: nothing in
this library touches the network unless you run the command that says to.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from agentguard import Guard
from agentguard.config import CONFIG_ENV_VAR
from agentguard.exceptions import GuardConfigError
from agentguard.pricing import PRICING_AS_OF
from agentguard.snapshot import (
    DEFAULT_SNAPSHOT_URL,
    parse_snapshot_payload,
    read_snapshot_file,
    snapshot_path,
    write_snapshot_file,
)

#: A stand-in for the downloaded catalogue. The real one is a JSON array of models
#: with per-token USD strings; `price_snapshot` scales them to per-1M for you.
CATALOGUE: dict[str, Any] = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "anthropic/claude-fable-5.1",
            "pricing": {
                "prompt": "0.00001",
                "completion": "0.00005",
                "input_cache_read": "0.00000025",
            },
        },
        {
            "id": "acme/frontier-v9",
            "pricing": {"prompt": "0.000004", "completion": "0.000016"},
        },
        # Catalogues carry entries they cannot express as a flat rate. "-1" is a
        # sentinel meaning "priced elsewhere"; it is skipped, never read as a rate.
        {"id": "acme/router", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        # Everything below lives in a scratch config directory, so running this
        # example cannot touch your real one.
        os.environ["APPDATA"] = folder
        os.environ["LOCALAPPDATA"] = folder
        os.environ["XDG_CONFIG_HOME"] = folder
        os.environ[CONFIG_ENV_VAR] = ""

        bundled = Guard(max_usd=10.0, on_unknown_model="ignore")
        print(f"Bundled table, snapshot {PRICING_AS_OF}:")
        _show(bundled, "gpt-4o")
        # The catalogue carries a gateway id ("acme/frontier-v9"); agent-guard keys
        # on the bare name the provider reports, because normalize_model_key strips
        # the namespace on both sides of the lookup.
        _show(bundled, "frontier-v9")
        print()

        print(f"A catalogue arrives ({DEFAULT_SNAPSHOT_URL} in real use)...")
        snapshot = parse_snapshot_payload(CATALOGUE, url="https://example.test/models")
        written = write_snapshot_file(snapshot_path(), snapshot)
        print(f"  {snapshot.describe()}, {snapshot.skipped} entry skipped")
        print(f"  checksum {snapshot.checksum[:32]}...")
        print(f"  written to {written.name}")
        print()

        reloaded = read_snapshot_file(written)
        print(f"Verified on read: {reloaded.describe()}")
        print()

        refreshed = Guard(max_usd=10.0, on_unknown_model="ignore")
        print("The same guards, now with the snapshot merged underneath:")
        _show(refreshed, "gpt-4o")
        _show(refreshed, "frontier-v9", note="  (catalogue id: acme/frontier-v9)")
        _show(refreshed, "claude-fable-5.1", note="  (catalogue id: anthropic/...)")
        print()

        # Your own prices still win. This is the whole reason the snapshot is the
        # lowest layer rather than the highest.
        negotiated = Guard(
            max_usd=10.0, on_unknown_model="ignore", pricing={"gpt-4o": (0.50, 1.00)}
        )
        _show(negotiated, "gpt-4o", note="  <- Guard(pricing=...) in code")
        print()

        # Tampering is caught rather than billed.
        document = json.loads(written.read_text(encoding="utf-8"))
        document["models"]["gpt-4o"]["input"] = 0.0000001
        written.write_text(json.dumps(document), encoding="utf-8")
        try:
            Guard(max_usd=10.0, on_unknown_model="ignore")
            print("  a tampered snapshot was LOADED — that would be a bug")
        except GuardConfigError as exc:
            print(f"  a tampered snapshot is refused: {str(exc)[:72]}...")

        print()
        print(refreshed.report())


def _show(guard: Guard, model: str, *, note: str = "") -> None:
    """Print what one model resolves to, or that it has no price at all."""
    price = guard.price_table.resolve_price(model)
    if price is None:
        print(f"  {model:<20} unpriced - its calls will not move the budget{note}")
        return
    cached = (
        f"  cached ${price.cached_input_per_1m:g}" if price.cached_input_per_1m is not None else ""
    )
    print(f"  {model:<20} ${price.input_per_1m:g} in / ${price.output_per_1m:g} out{cached}{note}")


if __name__ == "__main__":
    main()
