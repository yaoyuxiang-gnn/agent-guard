"""Price your own models without editing the library.

Run it::

    python examples/custom_models.py

The bundled table knows public list prices. It cannot know a fine-tune, a
gateway alias, a regional endpoint or a rate you negotiated, so agent-guard reads
a JSON config file for the rest. This example writes one into a temporary
directory, points a guard at it, and shows what each call is billed at.

The file the CLI writes is exactly what ``Guard`` reads, so the same thing can be
done from a terminal::

    agentguard config set my-finetune-v3 3 12 --cached 0.3
    agentguard config alias acme/fast claude-3-5-haiku
    agentguard config disable gpt-4
    agentguard pricing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentguard import Guard, load_config

CONFIG = {
    "version": 1,
    "models": {
        # A fine-tune of your own, with a cached-input discount.
        "my-finetune-v3": {"input": 3.0, "output": 12.0, "cached_input": 0.3},
        # A model hosted somewhere cheap; a short array means (input, output).
        "acme-local-7b": [0.05, 0.08],
        # Repricing a bundled model, because the published number is not yours.
        "gpt-4o": [2.00, 8.00, 1.00],
    },
    "aliases": {
        # What the gateway reports -> what it actually is.
        "acme/fast": "claude-3-5-haiku",
        "internal-llm": "acme-local-7b",
    },
    # Do not trust the bundled price at all: bill it as unpriced instead.
    "disable": ["gpt-4"],
}


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agentguard.json"
        path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

        config = load_config(path)
        print(f"config: {path}")
        print(f"  {len(config.models)} model(s), {len(config.aliases)} alias(es)")
        print()

        # Guard(pricing=...) in code always wins over the file; use_config=False
        # ignores the file entirely.
        guard = Guard(max_usd=0.50, config_path=path, name="custom-pricing")

        print("what each model is billed at")
        print("  (gpt-4 is disabled, so its call warns that it is unpriced)")
        for model in ("my-finetune-v3", "acme-local-7b", "gpt-4o", "acme/fast", "gpt-4"):
            record = guard.record(model, input_tokens=10_000, output_tokens=1_000)
            cost = "unpriced" if record.cost_usd is None else f"${record.cost_usd:.4f}"
            origin = guard.price_table.origin(model) or "-"
            print(f"  {model:<18}{cost:>10}   {origin:<8}-> {record.canonical_model}")

        print()
        print(f"  spent ${guard.spent_usd:.4f} of ${guard.remaining_usd + guard.spent_usd:.2f}")
        print(f"  unpriced calls: {guard.report().unpriced_calls} (excluded from the budget)")
        print()

        # The report carries the source, so a saved artefact is auditable.
        print(guard.report().render())


if __name__ == "__main__":
    main()
