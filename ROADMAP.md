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

## Shipped in 0.4

### ~~A price table that can be refreshed without a release~~ — shipped

`agentguard pricing --update` downloads a catalogue, checksums it, and caches it
beside your config; `--from-file` imports one you already have, and `--status` and
`--remove` manage the layer. The snapshot merges **underneath** your configuration, so
a public catalogue can reprice a bundled model but can never override a rate you set
yourself.

The tension the item named — the no-I/O rule — is resolved by making the network
reachable from exactly one place: an explicit subcommand. Nothing fetches at import,
on a timer, or in a background thread, and the checksum is re-verified on every read,
so a corrupted or hand-edited snapshot is refused rather than billed.

Two things about the bundled table remain true, and the refresh does not change
either: retired models keep their last published price, and a model absent from both
the bundle and the snapshot is still *unpriced* rather than guessed.

### ~~Per-tool and per-tag cost attribution~~ — shipped

`scoped_budgets={"tool:search": 1.0, "tag:index": 0.5}` caps a part of the run rather
than the whole of it, tripping `BudgetScopeExceeded` with the tool or tag named.
Attribution is O(1) per call, survives a checkpoint, and reaches worker threads
through `guard.bind(...)` / `guard.context()` — no thread inherits a `contextvars`
context, which is why those exist.

### Smaller items in 0.4

- **`preflight(strict=True)`** refuses a model it cannot price instead of returning
  `0.0`. It defaults to on when `on_unknown_model="error"`.
- **Bedrock's Converse usage shape** (`inputTokens` / `outputTokens` /
  `cacheReadInputTokens`) is now understood, so a Bedrock run is priced rather than
  reported as unpriced.
- **Worker-thread attribution**: `guard.bind()` and `guard.context()`.

---

## Later (0.5+)

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

### The price snapshot, extended

- **A `--check` mode** that reports how far the snapshot has drifted from the
  bundled table, so a refresh can be reviewed rather than trusted.
- **Per-provider sources.** The default catalogue is a gateway's, not each
  provider's own page; a user who wants OpenAI's rates only has no way to say so.
- **`--url` presets** for catalogues known to work, rather than one default.

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
  a worse bug than a stale price. It is a subcommand you run, not something that
  happens to you.
- **Buying, trading, or gamifying stars.** Not a roadmap item, but worth stating on
  a page like this.

---

## How to influence this

Open an issue describing the failure you want prevented. A concrete story about a
run that cost too much, or an agent that looped in a way the detectors miss, moves
an item up this list far more than a feature request does.

If you want to implement something here, comment on the issue first so we can agree
on the interface before you write the code.
