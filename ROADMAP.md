# Roadmap

What is planned, roughly in order, with the reasoning behind each item. Nothing
here is a promise and dates are deliberately absent — this is a library maintained
in the open, not a product with a schedule.

Scope is bounded by the four constraints in [CONTRIBUTING.md](CONTRIBUTING.md):
zero dependencies, no I/O, never guess a number, fail at construction.

---

## Next (0.2)

### Per-tag and per-tool cost attribution

The tracker already records `tag` and `step` on every call but does not aggregate
by them. The ask is a report that answers "which tool is eating my budget?"
directly, instead of forcing you to sum `guard.tracker.records` yourself.

```python
guard.report().by_tag()     # {"search": 0.41, "summarise": 0.88, ...}
```

Interface not settled. Aggregation is trivial; deciding what belongs in the text
report without making it noisy is the actual work.

### `on_trip="stop"` ergonomics

`"stop"` currently requires the caller to write `while not guard.stopped:`. That is
easy to get wrong, and getting it wrong means an agent keeps running after the
budget is gone. Worth exploring whether `guard.step()` should raise a dedicated
`GuardStopped` once stopped, so the loop unwinds without any caller discipline —
while still allowing cleanup.

### ~~Anthropic and OpenAI streaming helpers~~ — shipped

Streaming responses are wrapped in a `GuardedStream` that records usage once,
when the stream is drained; Anthropic's `messages.stream()` manager and async
clients are covered the same way. See `examples/streaming.py`.

### `Guard.snapshot()` for checkpointing

An agent that checkpoints its own state should be able to checkpoint its spend too,
so a resumed run does not start with a fresh budget. Requires a serialisable
counter form; the record list is too heavy for the purpose.

---

## Later (0.3+)

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
