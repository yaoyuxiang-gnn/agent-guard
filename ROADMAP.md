# Roadmap

What is planned, roughly in order, with the reasoning behind each item. Nothing
here is a promise and dates are deliberately absent — this is a library maintained
in the open, not a product with a schedule.

Scope is bounded by the four constraints in [CONTRIBUTING.md](CONTRIBUTING.md):
zero dependencies, no I/O, never guess a number, fail at construction.

---

## Shipped in 0.3

*Everything planned for 0.3 is in 0.3.0. The next section is what comes after it.*

### ~~Per-tag and per-tool cost attribution~~ — shipped

`report().by_tag` and `report().by_tool` answer "which tool is eating my budget?"
directly. A call made inside `with guard.tool("search", ...)` is attributed to that
tool automatically, calls that carried neither a tag nor a tool land under
`UNATTRIBUTED` so the parts always add up, and the text report prints a breakdown
only when it has at least two buckets — one row would just repeat the run total,
which was the part of this item that actually needed deciding. See "Where the money
went" in the README.

### ~~`on_trip="stop"` ergonomics~~ — shipped

`"stop"` records the trip, calls `on_trip_callback`, lets the current step finish so
a caller can clean up — and then raises `GuardStopped` from every entry point, so a
loop that never checks `guard.stopped` stops instead of spending on. The trip it
carries as `cause` keeps `except GuardTripped` handlers working, and accounting still
happens before the exception, because a call that already went out is real money.

### ~~Anthropic and OpenAI streaming helpers~~ — shipped

Streaming responses are wrapped in a `GuardedStream` that records usage once,
when the stream is drained; Anthropic's `messages.stream()` manager and async
clients are covered the same way. See `examples/streaming.py`.

### ~~`Guard.snapshot()` for checkpointing~~ — shipped

`Guard.snapshot()` / `Guard.from_snapshot()` let an agent that already checkpoints
its own state checkpoint its spend too, so a resumed run keeps counting against what
it already spent. The format is counters rather than records — calls grouped by
`(model, tag, tool)` — which keeps every breakdown total exact while dropping the
per-call log, and the report says how many calls it inherited instead of passing
them off as its own. See `examples/checkpointing.py`.

Two questions this item left open, now answered:

- **Detector windows do survive a checkpoint.** A loop that spans one is still a
  loop, so the four built-in detectors serialise their windows. Wall-clock time is
  the one thing that deliberately does *not* survive: `max_seconds` caps how long
  the current process may run, so restoring an elapsed duration would make a resumed
  run trip on time it never spent. Money and steps accumulate; the clock restarts.
- **A checkpoint that cannot be read exactly is refused, not half-applied.** A
  wrong format version, a negative count, or a call count that disagrees with its
  groups raises at restore. A budget that restores approximately is a budget that
  might not stop — and that is the one outcome worth being strict about.

---

## Later (0.4+)

### A price table that can be refreshed without a release

`PRICING_AS_OF` goes stale between releases, and asking users to wait for a patch to
learn a new model's price is a poor answer. The tension is the no-I/O rule: fetching
prices at import time is exactly the kind of hidden network call this library
refuses to make.

Likely shape: an explicit, opt-in `agentguard pricing --update` that writes a local
cache file, with the bundled table always the fallback. Never automatic.

*Partly shipped.* The user-facing half — price a model agent-guard has never heard
of, reprice one it has, alias a gateway name, disable a bundled price you do not
trust — landed as the JSON config file (`agentguard config set ...`, see the
README). A downloaded snapshot would be just another entry in the same merge chain,
so the remaining work is the fetch itself: opt-in, checksummed, and written to the
user config directory rather than imported over the network.

This is also the honest fix for the bundled table's real weakness. A refresh
mechanism makes staleness a two-command problem; until it exists, a model newer than
the snapshot is reported as *unpriced* and excluded from the budget, which is safe
but not useful.

### Additional detectors, if they earn their place

Candidates, none of them committed:

- **Alternating arguments** — the same tool called with arguments that oscillate
  between two values rather than repeating exactly.
- **Escalating retries** — the same call with a growing timeout or a shrinking
  query, which is a distinct failure from an exact repeat.
- **Cost-rate anomaly** — burn rate far above the run's own baseline.

Each one has to pass the bar in [CONTRIBUTING.md](CONTRIBUTING.md): reachable with
the defaults, silent on healthy work, and explainable in one sentence. A detector
that cannot clear all three makes the library worse, not better.

### Framework adapters

**LangGraph is shipped**: `agentguard.integrations.langgraph.guard_langgraph`
returns a callback handler that prices LLM calls and fingerprints tool calls for
the loop detectors, without adding a dependency (langchain-core is subclassed
lazily, only if already installed). See `examples/langgraph_demo.py`.

CrewAI, Pydantic AI and the OpenAI Agents SDK all have a natural place to hook a
guard. The work is not the wrapping; it is doing so without adding a dependency
and without the adapter rotting when the framework changes its callback shape.
Worth doing once a clear pattern emerges from real usage.

---

## Explicitly not planned

Recorded so that the answer is available before the issue is opened.

- **A server, dashboard, or hosted service.** agent-guard runs in your process and
  sends nothing anywhere. That is the point.
- **A proxy.** Sitting between you and your provider would mean seeing traffic it
  was not told about, and would break the no-I/O rule.
- **Runtime dependencies.** Not for tokenization, not for HTTP, not for anything.
- **Exact token counting.** It needs provider tokenizers, which are dependencies.
  Pre-flight estimates are documented as heuristic and will stay that way.
- **Automatic price fetching.** See above; a hidden network call at import would be
  a worse bug than a stale price.
- **Buying, trading, or gamifying stars.** Not a roadmap item, but worth stating on
  a page like this.

---

## How to influence this

Open an issue describing the failure you want prevented. A concrete story about a
run that cost too much, or an agent that looped in a way the detectors miss, moves
an item up this list far more than a feature request does.

If you want to implement something here, comment on the issue first so we can agree
on the interface before you write the code.
