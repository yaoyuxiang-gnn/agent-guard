"""The four ways an agent gets stuck, and which detector catches each one.

Run it::

    python examples/loop_detection.py

The last scenario is the important one: a healthy agent doing real work must sail
straight through. A loop detector that also stops productive runs is worse than no
detector at all.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_guard import Guard, LoopDetected


def run_scenario(title: str, drive: Callable[[Guard], None], *, expect_trip: bool = True) -> None:
    """Drive one pattern through a fresh guard and report the outcome.

    ``expect_trip=False`` inverts the verdict, so the healthy-work scenario fails
    loudly if the detectors ever start killing productive runs.
    """
    guard = Guard(name="loop-demo")
    try:
        drive(guard)
    except LoopDetected as exc:
        if expect_trip:
            print(f"  {title:<32} -> {exc.kind}")
            print(f"  {'':<32}    {exc.detail}")
        else:
            print(f"  {title:<32} -> FALSE POSITIVE [{exc.kind}] (this would be a bug)")
            print(f"  {'':<32}    {exc.detail}")
    else:
        if expect_trip:
            print(f"  {title:<32} -> NOT CAUGHT (this would be a bug)")
        else:
            print(f"  {title:<32} -> clean, as it should be")


def exact_repeat(guard: Guard) -> None:
    """The same tool with byte-identical arguments, over and over."""
    for _ in range(5):
        with guard.tool("search_web", {"query": "weather in oslo"}):
            pass


def ping_pong(guard: Guard) -> None:
    """Two tools bouncing off each other, changing nothing."""
    for _ in range(4):
        with guard.tool("read_file", {"path": "app.py"}):
            pass
        with guard.tool("write_file", {"path": "app.py", "body": "print('hi')"}):
            pass


def paraphrasing(guard: Guard) -> None:
    """The same search, reworded by one whitespace character each time."""
    query = "how do i fix asyncio event loop is closed in python"
    for padding in range(4):
        with guard.tool("search_web", {"query": query + " " * padding}):
            pass


def no_progress(guard: Guard) -> None:
    """A progress marker that never moves, however long the agent runs."""
    for _ in range(8):
        guard.progress({"rows_written": 0})


def healthy_run(guard: Guard) -> None:
    """Genuine, varied work. Must not trip anything."""
    plan = [
        ("search_web", {"query": "postgres index bloat"}),
        ("read_file", {"path": "schema.sql"}),
        ("run_sql", {"query": "SELECT count(*) FROM orders"}),
        ("write_file", {"path": "report.md", "body": "bloat is 40%"}),
        ("search_web", {"query": "vacuum full versus reindex"}),
        ("run_sql", {"query": "SELECT pg_size_pretty(1024)"}),
        ("progress", {"rows_written": 0}),
        ("progress", {"rows_written": 250}),
        ("progress", {"rows_written": 900}),
    ]
    for tool, payload in plan:
        if tool == "progress":
            guard.progress(payload)
        else:
            with guard.tool(tool, payload):
                pass


def main() -> None:
    print("Detector defaults, one scenario each:\n")
    run_scenario("exact repeat", exact_repeat)
    run_scenario("two-step ping-pong", ping_pong)
    run_scenario("paraphrased calls", paraphrasing)
    run_scenario("no progress", no_progress)
    print()
    run_scenario("healthy varied work", healthy_run, expect_trip=False)
    print()
    print(f"  detectors in use: {', '.join(d.name for d in Guard().detectors)}")


if __name__ == "__main__":
    main()
