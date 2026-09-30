"""Checkpoint a run's spend, then resume it in a new guard.

An agent that already checkpoints its own state — a cursor, a queue position, a
half-finished plan — should be able to checkpoint its budget in the same file.
Otherwise a job that restarts after spending $4 of a $5 cap comes back with $5 to
spend, and the cap stops meaning anything across restarts.

This runs offline and prints both halves of the story.
"""

from __future__ import annotations

import json

from agentguard import Guard, GuardTripped

BUDGET = 0.60


def do_work(guard: Guard, *, start_at: int, stop_at: int) -> None:
    """Spend on a few models, as an indexing loop would."""
    for page in range(start_at, stop_at):
        tag = "index" if page % 2 == 0 else "verify"
        with guard.step(tag=tag) as step:
            # A cheap model summarises, an expensive one writes the summary.
            step.record("gpt-4o-mini", input_tokens=20_000, output_tokens=1_000)
            step.record("gpt-4o", input_tokens=40_000, output_tokens=2_000)
        print(f"  page {page} indexed")


def main() -> None:
    print("== first process: work until the checkpoint ==")
    first = Guard(max_usd=BUDGET, max_steps=100, name="nightly-indexer", use_config=False)
    do_work(first, start_at=0, stop_at=3)

    payload = first.as_snapshot(indent=2)
    print(f"  spent ${first.spent_usd:.4f} of ${BUDGET}, {first.calls} calls")
    print(f"  checkpointing {len(payload)} bytes of JSON (not a call log)")

    # What a real agent writes alongside its own state.
    checkpoint = {"cursor": 3, "guard": json.loads(payload)}

    print("\n== second process: resume from the checkpoint ==")
    resumed = Guard.from_snapshot(
        checkpoint["guard"],
        max_usd=BUDGET,
        max_steps=100,
        name="nightly-indexer",
        use_config=False,
    )
    print(f"  resumed with ${resumed.spent_usd:.4f} already spent")
    print(f"  remaining ${resumed.remaining_usd:.4f}, not ${BUDGET:.4f}")
    print(f"  steps carried over: {resumed.steps}")

    # The point: the first call of the resumed run is measured against the money
    # the *first* process spent, so the cap holds across the restart.
    print("\n  continuing until the budget stops it ...")
    try:
        do_work(resumed, start_at=checkpoint["cursor"], stop_at=20)
    except GuardTripped as exc:
        # GuardStopped is a GuardTripped, so this catches both modes.
        print(f"  stopped: {exc}")

    print()
    print(resumed.report())

    print("\n== what survived, and what did not ==")
    print("  survived : cost, steps, tokens, by-model, by-tag, by-tool, detector windows")
    print("  lost     : per-call records, timestamps and meta - the report says so")
    print("  reset    : wall-clock time, because max_seconds caps *this* process")


if __name__ == "__main__":
    main()
