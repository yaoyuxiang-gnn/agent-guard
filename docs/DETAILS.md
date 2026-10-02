# agent-guard — all the details

The [README](../README.md) is the short version: what this is, what it catches, and
how to wire it in. This file is everything else — every detector and how to tune it,
the full pricing configuration and why it is shaped that way, the checkpoint format,
the complete CLI, and the constraints the library is built to.

Nothing here is needed to use agent-guard. It is what you read when a default does
not fit, or when you want to know why one was chosen.

**Contents**

- [The four detectors](#the-four-detectors) — including writing your own
- [What happens when it trips](#what-happens-when-it-trips) — `raise`, `warn`, `stop`
- [Per-tool and per-tag budgets](#per-tool-and-per-tag-budgets)
- [Pre-flight](#pre-flight-refuse-a-call-before-paying-for-it)
- [Prices: the whole configuration](#prices-the-whole-configuration) — including refreshing the table
- [Checkpointing](#checkpointing)
- [The complete CLI](#the-complete-cli)
- [Design principles](#design-principles)
- [Limitations, in full](#limitations-in-full)
- [Development](#development)

---

## The four detectors

A budget cap eventually catches a runaway loop — after the money is gone. Loop
detection catches it *while* it is happening, and says why.

| Detector | Default | Fires on | Tune with |
|---|---|---|---|
| `RepeatDetector` | on | 3rd identical call in a 12-call window | `max_repeats`, `window` |
| `CycleDetector` | on | 2nd full pass of a 2–4 step pattern | `min_cycle`, `max_cycle`, `repeats` |
| `SimilarityDetector` | on | 4th call ≥ 95% similar in an 8-call window | `threshold`, `window`, `max_similar`, `compare_chars` |
| `NoProgressDetector` | on (`progress` channel) | 6th unchanged progress marker | `max_stagnant` |

`RepeatDetector`, `CycleDetector` and `SimilarityDetector` watch **tool calls**.
`NoProgressDetector` watches the **`progress()` channel**, because only your agent
knows what progress means — a row count, an HTTP cursor, a test-pass tally:

```python
with guard.step() as step:
    rows = load_more()
    step.progress({"rows_written": rows})     # nothing moves for 6 of these -> trip
```

The defaults are ordered cheapest-and-most-precise first, so the verdict you get is
the most specific explanation available. `CycleDetector` deliberately defaults to
`repeats=2` rather than 3: with 3 it would need six observations while
`RepeatDetector` fires on the third, so it would never be reachable — the cycle would
always be reported as a repeat. Two identical passes is already a strong signal that
nothing changed between them.

Replace the detector set per guard, or switch detection off entirely:

```python
from agentguard import Guard, RepeatDetector

Guard(detectors=[RepeatDetector(max_repeats=2, window=8)])   # stricter
Guard(loop_detection=False)                                  # limits only
```

### Writing your own detector

A detector is a small state machine fed one signature per observation. Return a
verdict to stop the run, `None` to keep going.

```python
from agentguard import Detector, LoopVerdict

class SchemaThrashDetector(Detector):
    """Trip when the agent migrates the same table back and forth."""

    name = "schema-thrash"

    def observe(self, signature, step):
        if signature.count("alter_table") >= 4:
            return LoopVerdict(self.name, "table altered 4 times", signature, 4, step)
        return None

    def reset(self):                      # called when a guard is reused
        ...

guard = Guard(detectors=[SchemaThrashDetector()])
```

Three optional pieces, all with sensible defaults:

- **`get_state()` / `set_state()`** — implement both to make your detector survive a
  [checkpoint](#checkpointing). The base class raises `NotImplementedError` rather
  than returning `{}`, because a detector that silently reported "no state" would
  look checkpointed while losing the history it exists to keep. A detector that does
  not implement them does not block the restore; the guard warns once that a loop
  beginning before the checkpoint may need more observations.
- **`reset()`** — clear accumulated state when a guard is reused via `guard.reset()`.
- **`name`** — the stable identifier used as `LoopVerdict.kind`, and the `[kind]`
  shown in a trip message.

A detector only sees the fingerprint, never the raw arguments:

```python
from agentguard import Guard

signature = Guard.call_signature("search", {"q": "a", "n": 1})
# 'search({"n":1,"q":"a"})' — keys sorted, so insertion order cannot matter
```

## What happens when it trips

| Mode | Behaviour |
|---|---|
| `"raise"` (default) | Raise the trip exception at the moment it fires |
| `"warn"` | Emit a `RuntimeWarning`, set `guard.stopped`, keep going — useful to *measure* before you enforce |
| `"stop"` | Record the trip, call `on_trip_callback`, finish the current step, then raise `GuardStopped` from every entry point afterwards |

`"stop"` exists for the loop that forgets to check `guard.stopped`: a long-running job
can flush, alert and clean up, and still be stopped rather than spending on.

```python
from agentguard import Guard, GuardStopped

guard = Guard(max_usd=5.00, on_trip="stop", on_trip_callback=alert_page)

try:
    while True:
        with guard.step():
            ...
except GuardStopped as exc:
    log.warning("stopped by %s: %s", exc.cause.reason, exc.cause)
finally:
    print(guard.report())
```

`GuardStopped` is a `GuardTripped` whose `cause` is the original trip, so `except
GuardTripped` still catches both a budget overrun and a loop.

**Accounting is never skipped to raise.** A call that already went out is recorded
*before* the exception surfaces, in every mode. An unrecorded call is spent money the
report denies, which is the one thing a cost report must not do. A guard stopped by
`"stop"` therefore also refuses a wrapped client's next call, which means it costs
nothing.

## Per-tool and per-tag budgets

`max_usd` caps the run. It cannot cap a *part* of the run, so a single tool that has
gone into a retry storm can spend the entire allowance before the run-level ceiling
notices — and by the time you read the report, the answer to "what ate the budget?"
is historical.

`scoped_budgets` puts a smaller ceiling on one tool or one tag:

```python
guard = Guard(max_usd=5.00, scoped_budgets={"tool:search": 1.00, "tag:index": 0.50})
```

A scope whose spend exceeds its cap trips `BudgetScopeExceeded`, which carries
`scope`, `name`, `spent_usd` and `limit_usd` — so the message names the culprit
rather than the run:

```text
Budget exceeded for tool 'fetch': spent $0.0125 of its $0.012 allowance.
```

Three decisions in that shape:

**Scope keys are validated at construction.** `"tool:search"` or `"tag:index"`; a key
that is misspelled, empty, or names some other kind of scope raises `GuardConfigError`
while the guard is being built. A key no call could ever match is a cap that silently
never fires, and a limit that reports it is working while doing nothing is the exact
failure this library exists to prevent. A `NaN` limit is refused for the same reason —
every comparison against `NaN` is false.

**The run-level limit is checked first**, so when both are over, the trip you get is
`BudgetExceeded`. "The run is over budget" is the more urgent fact, and it is the one
whose exception carries `max_usd`.

**Calls with no tool or no tag belong to no scope.** They are covered by `max_usd` and
nothing else. There is no `"(unattributed)"` scope to cap, because a caller who wanted
to cap that bucket would have to name it, and naming it is what `guard.tool(...)` and
`guard.step(tag=...)` are for.

Cost accumulates as calls are recorded, so the check is O(1) per call rather than a
scan of the run so far — and a scoped cap counts spend restored from a
[checkpoint](#checkpointing), because a resumed run that had already spent its
allowance has still spent it.

```bash
python examples/scoped_budgets.py
```

## Threads and attribution

Live attribution travels in `contextvars`, which are copied into an `asyncio` task
but **not** into any thread — not `threading.Thread`, not a `ThreadPoolExecutor`
worker. Each starts with the defaults.

The money is never lost: a call recorded on a worker thread is counted, priced, and
in the report. What is lost is the *attribution* — `step` and `tool` are `None`, so
`by_tag` and `by_tool` under-report, and a `scoped_budgets` cap on that tool never
fires. Two ways to carry it over, both captured at the moment you call them:

```python
with guard.step(tag="fan-out"), guard.tool("fetch"):
    results = list(pool.map(guard.bind(fetch), urls))   # wraps one callable
```

```python
def worker(ctx):
    with ctx:                 # ctx = guard.context(), captured in the main thread
        ...
```

`guard.bind(...)` is the one-argument form of `guard.context()`. Both install nothing
when there is no ambient step or tool, so calling them outside a block is a no-op
rather than an error.

## Pre-flight: refuse a call *before* paying for it

A post-hoc budget check can only report overspend. `preflight()` refuses a call whose
worst case will not fit in what is left:

```python
guard.preflight("gpt-4o", input_tokens=180_000, max_output_tokens=16_000)
# raises BudgetExceeded before the request is issued
```

Or for a whole client with one flag:

```python
client = guard_openai(OpenAI(), max_usd=0.05, preflight=True)
```

Input tokens are estimated from the serialised prompt, because counting them properly
needs a provider tokenizer that agent-guard deliberately does not depend on. Treat
pre-flight as a net for catastrophic calls, not as an accounting figure — the number
in the report is what the provider said, not what pre-flight guessed.

**The unpriced hole, and how to close it.** For a model with no price at all,
`preflight()` returns `0.0` and does *not* refuse. That is deliberate: an unbounded
cost cannot be shown to exceed the budget, and refusing every call to an unpriced
model would be a different failure than overspending. It is also a real hole in a
gate you were treating as a hard stop — an expensive model that is merely absent from
the price table sails straight through. So `strict=True` refuses it instead:

```python
guard.preflight("mystery-model", input_tokens=200_000, strict=True)
# BudgetExceeded, with .unpriced_model set
```

`Guard(preflight_strict=True)` makes that the default, and it is already the default
when `on_unknown_model="error"` — someone who has said "an unpriced model is an error"
does not want the pre-flight gate waving one through.


## Prices: the whole configuration

### Model names resolve conservatively

`normalize_model_key()` folds the four shapes seen in the wild — gateway namespacing,
version pinning, variant tags and dotted provider prefixes:

```python
normalize_model_key("OpenAI/GPT-4o")                             # 'gpt-4o'
normalize_model_key("anthropic.claude-sonnet-4@20250514")        # 'claude-sonnet-4'
normalize_model_key("deepseek/deepseek-chat-v3:free")            # 'deepseek-chat-v3'
normalize_model_key("gpt-3.5-turbo")                             # 'gpt-3.5-turbo'
```

After an exact lookup fails, a **trailing version segment** may be dropped, but only
when it cannot change the model family:

```python
table.resolve_price("gpt-4o-2024-08-06")    # -> gpt-4o
table.resolve_price("claude-opus-5-20260601")  # -> claude-opus-5
table.resolve_price("gpt-5.6-sol")          # -> gpt-5.6-sol, never gpt-5
```

That last case is the point: `6-sol` is part of the model, not a build number, so a
prefix match is refused. An unknown model stays unknown rather than being billed at
the nearest thing that looked similar.

### The config file

The bundled table cannot know your fine-tune, your gateway's aliases, a regional
endpoint, or a rate you negotiated. Those go in JSON that `Guard` picks up
automatically:

```json
{
  "version": 1,
  "models": {
    "my-finetune-v3": {"input": 3.0, "output": 12.0, "cached_input": 0.3},
    "acme-local-7b": [0.05, 0.08],
    "gpt-4o": [2.0, 8.0]
  },
  "aliases": {"acme/fast": "claude-3-5-haiku"},
  "disable": ["gpt-4"]
}
```

| key | answers |
|---|---|
| `models` | *What does this model cost?* USD per 1M tokens, as an object or a short `[input, output]` array. A name matching a bundled model **reprices** it. |
| `aliases` | *What is this name really?* Matched against the reported model string exactly, before any other interpretation — so `acme/fast` can point at a model that has a price. |
| `disable` | *Which bundled prices do I not trust?* A disabled model becomes **unpriced**: counted, reported, and excluded from the budget rather than billed at a number you rejected. |

### Where it is read from, and why

1. `$AGENTGUARD_CONFIG` — an explicit path (`none` / `off` / `0` disables config entirely)
2. `%APPDATA%\agentguard\pricing.json` on Windows,
   `$XDG_CONFIG_HOME/agentguard/pricing.json` (default `~/.config/...`) elsewhere —
   **your own file, always read**
3. `agentguard.json` or `.agentguard.json` in the working directory or the nearest
   parent — a **project** file, read only when you trust it with
   `AGENTGUARD_TRUST_PROJECT_CONFIG=1`

Files merge, with the more specific one winning on a conflict.

The third one is deliberate. A project file travels with the repository it sits in,
so it is written by whoever wrote that repository. Reading it by default would make
"clone this repo and run your agent in it" a way to reprice every model to nearly
nothing, or `disable` the expensive ones so their calls stop counting against the
budget — a guard bypass performed with a data file. So it is skipped unless you ask.
Skipping is reported once per process, and `agentguard config path` shows what is in
effect and what was skipped.

Code always wins over any file:

```python
from agentguard import Guard, Price

guard = Guard(
    max_usd=5.0,
    pricing={"my-finetune-v3": Price(3.00, 12.00)},   # or (3.00, 12.00), or {...}
    aliases={"internal-llm": "acme-local-7b"},
    disable=["gpt-4"],
    use_config=False,                                 # ignore the file entirely
)
```

### Inspecting what is in effect

```bash
$ agentguard pricing
119 models bundled, 2 configured (USD per 1M tokens, snapshot 2026-09)

  model                        input    output    cached  source
  claude-3-5-haiku              $0.8        $4     $0.08  builtin
  claude-opus-5.5                 $4       $20      $0.2  builtin
  ...
  gpt-4o                          $2        $8         -  config
  my-finetune-v3                  $3       $12      $0.3  config
  ...

  aliases
    acme/fast -> claude-3-5-haiku

  disabled
    gpt-4

  config: ~/.config/agentguard/pricing.json
```

The `source` column is the point: `builtin` means the bundled snapshot, `config`
means your file decided it. `--no-config` shows the bundled table alone and `--json`
emits the lot for tooling.

### A misconfigured price fails at construction

An unknown key, a negative or `NaN` rate, a boolean, an alias pointing at a model with
no price, a `disable` entry matching nothing — each raises `GuardConfigError` while
the guard is being built, naming the file. A misconfigured price must never quietly
change what a budget means.

`NaN` is the dangerous one and the reason `Price` validates its three rates: `NaN >
budget` is false, so a single bad number would silently make every budget check a
no-op. That is a guard that reports it is working while doing nothing.

### Why the table is dated

`PRICING_AS_OF` names the month the table was reviewed. It is a snapshot of public
list prices, and a provider can change a rate or retire a model the day after. Two
consequences worth knowing:

- **Retired models keep their last published price** rather than being dropped.
  Removing a name is not neutral — it silently turns every call to that model
  *unpriced*, which excludes the spend from the budget, the opposite of what someone
  still running it needs. A stale number is a smaller error than a disabled cap.
  `agentguard config disable <model>` is there when you would rather have the other
  behaviour.
- **A model newer than the snapshot is unpriced**, which is safe but not useful: your
  cap will not fire for it. That is what the refresh below is for.

### Refreshing the table

The bundled table is a snapshot of public list prices taken on `PRICING_AS_OF`. A
provider can change a rate, or retire a model, the day after. Rather than making you
wait for a patch, the table can be refreshed on purpose:

```bash
agentguard pricing --update              # download, verify, cache
agentguard pricing --from-file cat.json  # ...or import one you already have
agentguard pricing --status              # is a snapshot in effect, and from where
agentguard pricing --no-snapshot         # show what the bundled table alone says
agentguard pricing --remove              # forget it
```

Four decisions shape this, and each exists because the alternative is worse:

**Never automatic.** `pricing --update` is the only command in agent-guard that
touches the network. Nothing fetches prices at import, on a timer, or in a background
thread. A hidden request during `import agentguard` would be a far worse bug than a
stale table — this library's whole promise is that nothing leaves your process.

**Checksummed, and verified on every read.** The snapshot records the SHA-256 of its
own model map and re-checks it whenever the file is loaded. An edited, truncated or
corrupted snapshot raises rather than repricing models. Falling back silently would
leave you believing prices had been refreshed when the bundled table is what is
actually in effect.

**Underneath everything you configured.** Precedence, lowest first: the **bundled
table**, then the **snapshot**, then the **per-user config**, then the **project
config** (if trusted), then `Guard(pricing=...)` in code. A downloaded public
catalogue can reprice a bundled model, which is the point — but it can never override
a rate you set deliberately, and it cannot introduce an alias or a `disable`.

**Honest about what it cannot read.** A catalogue entry that carries no flat
per-token rate is *skipped*, not guessed at: some entries publish `-1` to mean "priced
elsewhere", and reading that as a number would bill those calls at a negative cost.
So are **variants** — `model:batch` at half price, `model:free` at nothing — because
they normalize to the same model name as the standard SKU and only one of the two can
be stored. Keeping the cheaper one was the first thing this code did, and it priced
most of a real catalogue at the batch rate; an agent billed at list would then have a
cap firing at twice the spend it thought it was tracking. A `:free` model therefore
reports as *unpriced* rather than as `$0`, which is the same conservative choice this
library makes everywhere else. The count of skipped entries is reported by `--update`
and by `--status`.

Two things worth knowing about the file itself. It lives beside your config
(`pricing-snapshot.json` in the same directory) and merges by the same rules, so
`$AGENTGUARD_CONFIG=none` disables both and `Guard(use_snapshot=False)` disables just
this layer. And catalogue ids are namespaced (`anthropic/claude-sonnet-4.5`) while
providers report the bare name — `normalize_model_key` strips the namespace on both
sides of the lookup, so the two meet.

Until a snapshot exists, `agentguard config set` remains the two-command answer for
one model, and a refresh is a release.

## Checkpointing

An agent that checkpoints its own state should be able to checkpoint its budget, so a
job that restarts after spending $4 of a $5 cap does not come back with $5 to spend.

```python
write_checkpoint({"cursor": 41, "guard": guard.snapshot()})

# later, in a new process
guard = Guard.from_snapshot(read_checkpoint()["guard"], max_usd=5.0)
guard.remaining_usd      # what is actually left, not the full budget
```

`examples/checkpointing.py` runs both halves and prints the carried-over report.

### The format is counters, not records

```
{
  "version": 1,
  "guard": {"steps": 3},
  "tracker": {
    "version": 1, "calls": 6, "unpriced_calls": 0,
    "groups": [
      {"model": "gpt-4o", "tag": "index", "tool": "(unattributed)",
       "calls": 3, "unpriced_calls": 0,
       "input_tokens": 120000, "output_tokens": 6000, "cost_usd": 0.36}
    ]
  },
  "detectors": {"actions": [...], "progress": [...]}
}
```

Call groups keyed by `(model, tag, tool)`. A six-call run over two models is a
**554-byte** snapshot, and the payload grows with the number of distinct
`(model, tag, tool)` combinations — a handful, in practice — not with the number of
calls, the way a record log would.

Grouping rather than three separate marginal breakdowns is what makes the format
**invertible**: `set_state()` rebuilds records from these groups and recovers
`by_model`, `by_tag` and `by_tool` exactly. Independent marginals would each be right
only if the others were ignored, and a report that is right one section at a time is
a report you cannot use.

What is genuinely lost is per-call detail: individual records, their timestamps, their
`meta`, and the order calls happened in. Per-call costs in a restored run are their
group's average, which is all a checkpoint knows — and the report says so:

```text
----------------------------------------------------------------
  prices as of 2026-09 (indicative only)
  includes 6 call(s) restored from a checkpoint
```

`Report.checkpointed_calls` carries the number for tooling.

### Three decisions in the format

**Detector windows survive.** A loop that spans a checkpoint is still a loop, so all
four built-in detectors serialise their sliding windows. Forgetting them would let a
run start its loop over — the exact failure the detectors exist to catch.

**Wall-clock time does not survive.** `max_seconds` caps how long *this process* may
run, so restoring an elapsed duration would make a resumed run trip on time it never
spent. Money and steps accumulate; the clock restarts. If you need a wall-clock bound
across restarts, carry your own start time and pass the remaining budget as a fresh
`max_seconds`.

**An unreadable snapshot is refused, never half-applied.** A wrong format version, a
negative count, a cost on a fully unpriced group, or a call count that disagrees with
its groups all raise `GuardConfigError` at restore — and the guard is left exactly as
it was, because the tracker is rebuilt before anything is assigned. A budget that
restores *approximately* is a budget that might not stop.

Unpriced models stay unpriced through a round trip (`cost_usd=None`), so restoring can
never turn forgotten money into budget headroom.

### Custom detectors

Implement `get_state()` / `set_state()` to join in. One that does not is recorded as
`null`; the budget still restores, and the guard warns once:

```text
RuntimeWarning: 1 detector(s) in this guard do not implement get_state/set_state,
so their observation history was not restored from the checkpoint; a loop that
began before it may need more observations before it trips. The budget, step count
and accounting are unaffected.
```

Refusing a budget over a detector would be the wrong trade: the money is the part that
must not be lost.

## The complete CLI

```bash
agentguard report run.json          # render a report saved by guard.save(...)
agentguard report run.json --json
agentguard pricing                  # effective table, with a source per model
agentguard pricing --no-config      # bundled prices only
agentguard pricing --no-snapshot    # ...and no downloaded snapshot either
agentguard pricing --json           # everything, for tooling
agentguard pricing gpt-4o           # one model, with example costs
agentguard pricing --update         # refresh from a public catalogue
agentguard pricing --from-file CAT  # import a catalogue you already have
agentguard pricing --status         # what the snapshot layer is doing
agentguard pricing --remove         # back to bundled prices
agentguard config path              # where config is read from, and what is ignored
agentguard config init              # write a starter file
agentguard config set NAME IN OUT [--cached C]
agentguard config alias NAME TARGET
agentguard config remove NAME
agentguard config disable NAME      # / agentguard config enable NAME
agentguard config list
```

`config set` and friends take `--user` / `--project` / `--file` to choose which file
they write. `pricing --update` is the only one that touches the network; `--url`
points it somewhere other than the default source.

```bash
$ agentguard pricing gpt-4o
gpt-4o  (USD per 1M tokens, snapshot 2026-09)

  input        $2.5 / 1M
  output        $10 / 1M
  cached      $1.25 / 1M

  example costs
    1M in + 1M out            $12.5
    100k in + 20k out         $0.45
    10k in + 2k out          $0.045

  source   builtin
```

The CI pattern this enables: `guard.save("run.json")` in the worker that holds the
guard, `agentguard report run.json` in the job that renders it. The reading process
does not need agent-guard installed as a dependency of anything except its own CLI.

## Design principles

These are the constraints the library is built to, and the reason it is safe to drop
into an existing agent stack. They are enforced in CI, not just documented — the
[CONTRIBUTING](../CONTRIBUTING.md) guide states them as the bar every change has to
clear.

**Zero dependencies.** No `pydantic`, no `httpx`, no provider SDK. Standard library
only. The build asserts the wheel declares no runtime dependencies, so it cannot drift
— if that step ever needs an install, the promise has been broken.

**Never guess a price.** An unknown model is counted as unpriced and surfaced in the
report, never billed at a plausible-looking rate. A rate that is negative, `NaN` or
infinite is rejected where it is written. A response with no usable token counts warns
rather than counting as `$0`.

**Fail at construction, not mid-run.** A bad `max_usd`, an unknown `on_trip` mode, a
misconfigured detector or a malformed config file raises while the `Guard` is being
built. A safety tool must never be the thing that crashes at 3am.

**Thread safe.** An agent fanning out concurrent tool calls will not lose a record:
every mutation happens under one re-entrant lock, and live counters use `contextvars`,
which is `asyncio`-correct.

**Explainable, not clever.** Loop detection is stdlib `difflib` and sliding windows,
not an embedding model. Every verdict is a sentence a human can act on, and a detector
that fires on healthy work is worse than no detector.

## Limitations, in full

- **A response that reports no usage cannot be priced.** It warns once and counts as
  unpriced rather than inventing a number. This is usually a provider that needs a flag
  — see the next point.
- **A stream is priced when it is drained.** Abandoning one records whatever usage it
  saw. For OpenAI-compatible streams pass `stream_options={"include_usage": True}` or
  the final chunk carries no usage and the call lands as unpriced.
- **Pre-flight input counts are estimates**, derived from the serialised prompt
  length. They are a net for catastrophic calls, not an accounting figure.
- **`max_seconds` measures this process.** It restarts on a restored checkpoint by
  design; see [Checkpointing](#checkpointing).
- **Retired models carry their last published price.** Correct at the time it was
  published, possibly not now; `disable` or reprice anything you depend on.
- **Cycle detection is bounded** to patterns of 2–4 steps by default, and similarity
  compares a bounded prefix of each fingerprint, so two very long tool payloads that
  differ only near the end may read as similar. Both bounds are adjustable.
- **A fingerprint is truncated at 256 characters**, but never ambiguously: the head,
  the tail and the digest of the whole payload are all kept, so two different calls
  can never share one. This is stricter than it sounds necessary, and it is not —
  truncating to a head alone made an agent indexing documents that share a boilerplate
  body look like it was repeating itself.

## Development

```bash
git clone https://github.com/yaoyuxiang-gnn/agent-guard
cd agent-guard

python -m unittest discover -s tests -t .   # no install required
pytest --cov=agentguard                     # if you prefer pytest
python examples/basic.py
```

No network, no fixtures to download, and the suite runs against a checkout with no
install step. `make all` runs what CI runs: lint, typecheck, tests, and every example.
