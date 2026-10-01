# agent-guard — API reference

Every public name, what it does, and the behaviour worth knowing before you rely on
it. The [README](../README.md) is the tour; this is the map.

Signatures in this file are copied from the source, and every example below is run
by `tests/test_api_reference.py` — a claim here that stops being true fails the
suite. You can check any of them yourself:

```python
>>> import inspect
>>> from agentguard import Guard
>>> "cached_input_tokens" in inspect.signature(Guard.record).parameters
True
```

**Contents**

- [Guard](#guard) — construction, limits, lifecycle
- [Step](#step) — the per-iteration context
- [Recording](#recording) — `record`, `observe`, `progress`, `tool`
- [Checkpointing](#checkpointing) — `snapshot`, `restore`
- [Reporting](#reporting) — `Report`, `LimitStatus`, `save`, `to_json`
- [Loop detection](#loop-detection) — detectors and `LoopVerdict`
- [Cost accounting](#cost-accounting) — `CostTracker`, `Usage`, `CallRecord`, summaries
- [Pricing](#pricing) — `Price`, `PriceTable`, and the table itself
- [Pricing config](#pricing-config) — the JSON file and its helpers
- [Exceptions](#exceptions)
- [Adapters](#adapters) — wrapped clients
- [Integrations](#integrations) — framework callbacks
- [Decorators](#decorators) — `guarded`, `current_guard`
- [Command line](#command-line)
- [Export surface](#export-surface) — everything importable, and from where

---

## Guard

```python
from agentguard import Guard

guard = Guard(
    max_usd=None,               # float  — stop once known spend exceeds this
    max_tokens=None,            # int    — input + output tokens
    max_steps=None,             # int    — guard.step() calls
    max_seconds=None,           # float  — wall clock since construction
    on_trip="raise",            # "raise" | "warn" | "stop"
    on_trip_callback=None,      # Callable[[GuardTripped], None]
    name=None,                  # str    — shown in the report
    pricing=None,               # {model: Price | (in, out) | {..}}
    aliases=None,               # {reported_name: priced_name}
    disable=(),                 # bundled models to treat as unpriced
    price_table=None,           # a prebuilt PriceTable; skips config discovery
    use_config=True,            # read the user's pricing JSON
    config_path=None,           # read exactly this file instead
    config=None,                # a prebuilt PricingConfig
    default_price=None,         # Price to assume for unknown models
    on_unknown_model="warn",    # "warn" | "error" | "ignore"
    loop_detection=True,        # False disables detectors entirely
    detectors=None,             # replace the action detectors
    progress_detectors=None,    # replace the progress detectors
    clock=time.monotonic,       # injectable for tests
)
```

Every limit is optional and independent. **A `Guard()` with no arguments still
detects loops and still produces a report** — it just cannot stop on cost.

Beyond the limits, three arguments decide how the guard behaves when something
fires:

| `on_trip` | Behaviour |
|---|---|
| `"raise"` (default) | Raise the trip exception at the moment it fires. |
| `"warn"` | Emit a `RuntimeWarning`, set `guard.stopped`, keep going. For measuring before enforcing. |
| `"stop"` | Record the trip, call `on_trip_callback`, let the current step finish, then raise `GuardStopped` from **every** entry point afterwards. |

Accounting happens before any exception, in all three modes: a call that already
went out is recorded, so the report never denies money that was spent.

A `Guard` is cheap; make one per request, task or job. A single shared guard cannot
tell two concurrent agents apart.

### Read-only state

| Attribute | Type | Meaning |
|---|---|---|
| `guard.name` | `str \| None` | The label passed at construction. |
| `guard.steps` | `int` | Steps opened so far. |
| `guard.calls` | `int` | Calls accounted so far. |
| `guard.spent_usd` | `float` | Known spend. Unpriced calls contribute `0`. |
| `guard.remaining_usd` | `float \| None` | `max_usd - spent`, floored at 0; `None` if uncapped. |
| `guard.usage` | `Usage` | Aggregate tokens. |
| `guard.elapsed_s` | `float` | Seconds since construction or the last `reset()`. |
| `guard.stopped` | `bool` | True once any limit has tripped. |
| `guard.tripped` | `GuardTripped \| None` | The exception that stopped it. |
| `guard.detectors` | `tuple[Detector, ...]` | The action detectors in use. |
| `guard.tracker` | `CostTracker` | The accounting object underneath. |
| `guard.price_table` | `PriceTable` | The lookup actually in use. |
| `guard.pricing_config` | `PricingConfig` | The config that produced it. |

### Lifecycle

```python
guard.reset() -> None
```

Clears all counters, spend and detector state, and restarts the clock. Inherited
checkpoint state is cleared too. Use it when one guard object serves many
independent runs.

The guard is also a context manager, which sets a `contextvar` so `current_guard()`
works inside the block:

```python
with Guard(max_usd=1.0) as guard:
    ...            # current_guard() is this guard here
```

`__enter__` is re-entrant and context-local, so the same guard can be used from two
threads or two asyncio tasks without them interfering.

### Methods at a glance

| Method | Does |
|---|---|
| `step(tag=None)` | Open a step. Returns a `Step`. |
| `record(...)` | Account one call. Returns a `CallRecord`. |
| `preflight(model, *, input_tokens, max_output_tokens=1024, price=None)` | Refuse a call whose worst case does not fit. Returns that worst case. |
| `observe(signature, *, step=None)` | Feed the loop detectors directly. |
| `progress(value, *, step=None)` | Feed the progress detectors. |
| `tool(name, args=None, *, step=None)` | Context manager; attributes calls made inside it. |
| `check()` | Re-evaluate every limit without recording anything. |
| `raise_if_tripped()` | Raise if the guard has already tripped; no-op otherwise. |
| `reset()` | Clear everything. |
| `snapshot()` / `as_snapshot()` / `restore()` / `from_snapshot()` | [Checkpointing](#checkpointing). |
| `report()` / `as_dict()` / `to_json()` / `save()` | [Reporting](#reporting). |
| `call_signature(name, args=None)` | Static. Build a stable fingerprint for a tool call. |

---

## Step

`guard.step(tag=None)` returns a `Step` and increments `guard.steps`. Used as a
context manager so one agent iteration is one step:

```python
with guard.step(tag="retrieval") as step:
    step.record(response)
    with step.tool("search", {"q": query}):
        results = search(query)
```

The `tag` is what appears in `report().by_tag`. A call recorded inside a
`with step.tool(...)` block is attributed to that tool, innermost block winning.

| Method | Does |
|---|---|
| `step.record(response=None, **kwargs)` | Same keyword arguments as `Guard.record`, but `tag` and `step` are filled in from the step. |
| `step.tool(name, args=None)` | Context manager; attributes calls inside it. |
| `step.observe(signature)` | Feed the action detectors at this step's index. |
| `step.progress(value)` | Feed the progress detectors at this step's index. |

A step increments the counter on **entry**, so `max_steps=25` trips when the 26th
step is opened, before its body runs.

---

## Recording

```python
guard.record(
    response=None,              # a provider response, OR a model name string
    *,
    model=None,
    input_tokens=None, output_tokens=None,
    cached_input_tokens=None, reasoning_tokens=None,
    tag=None, step=None, tool=None,
    meta=None,
    price=None,                 # bypass the price table for this call
) -> CallRecord
```

The positional argument is **either** a raw provider response **or** a model name.
A bare string is always read as the model, because no SDK returns a string as a
response. Token counts and the model name are extracted from the response when
present; explicit keyword arguments always win.

`extract_usage()` understands OpenAI (`prompt_tokens` / `completion_tokens`),
Anthropic (`input_tokens` / `output_tokens`) and Gemini
(`prompt_token_count` / `candidates_token_count`) shapes, as objects or plain dicts.
A response carrying no usable counts raises a `RuntimeWarning` rather than counting
as `$0`.

```python
guard.observe("search({\"q\":\"x\"})")   # straight to the detectors
guard.progress({"rows_written": 120})    # the progress channel

with guard.tool("search", {"q": "x"}):
    guard.record(response)               # attributed to `search`
```

`Guard.call_signature(name, args)` canonicalises arguments with sorted JSON keys, so
two calls differing only in dict insertion order produce the same fingerprint — and
therefore the same loop-detection verdict:

```python
>>> Guard.call_signature("search", {"q": "a", "n": 1})
'search({"n":1,"q":"a"})'
>>> Guard.call_signature("search", None)
'search()'
```

Signatures are truncated to 512 characters, and an argument that cannot be
serialised falls back to its `repr()` rather than raising.

### preflight

```python
guard.preflight(model, *, input_tokens, max_output_tokens=1024, price=None) -> float
```

Returns the worst-case cost of the call, or raises `BudgetExceeded` if that cost
would not fit in `remaining_usd`. Input tokens are counted from the number you pass,
not estimated — the adapter is what estimates them, from the serialised prompt.

```python
>>> guard = Guard(max_usd=100.0, use_config=False)
>>> guard.preflight("gpt-4o", input_tokens=1_000_000, max_output_tokens=1_000_000)
12.5
```

**Worth knowing:** for a model with no price and no `default_price`, preflight
returns `0.0` and does not refuse. An unbounded cost cannot be shown to exceed the
budget, and refusing every call to an unpriced model would be a different failure.
If you rely on preflight, price your models.

---

## Checkpointing

```python
guard.snapshot() -> dict            # counters, JSON-serialisable
guard.as_snapshot(*, indent=None) -> str
guard.restore(snapshot) -> None     # accepts the dict or the JSON string
Guard.from_snapshot(snapshot, *, max_usd=None, max_tokens=None,
                    max_steps=None, max_seconds=None, **kwargs) -> Guard
```

A snapshot carries the counters — calls grouped by `(model, tag, tool)` — so a
resumed run keeps counting against money it already spent. It does **not** carry
per-call records, timestamps or `meta`; `Report.checkpointed_calls` says how many
calls were inherited so the report does not present them as its own observations.

`from_snapshot` takes the same limit arguments as `Guard`, so limits are stated by
the run you are starting rather than re-imposed by a stale checkpoint.

Detector windows survive a checkpoint. Wall-clock time does not, because
`max_seconds` caps the current process. A snapshot that cannot be read exactly
raises `GuardConfigError` and leaves the guard untouched.

The full format, and the reasoning behind those choices, is in
[DETAILS.md](DETAILS.md#checkpointing).

---

## Reporting

```python
guard.report() -> Report
guard.as_dict() -> dict[str, Any]
guard.to_json(*, indent=2) -> str
guard.save(path, *, indent=2) -> Path
```

`report()` is an immutable snapshot of the run and stays valid after the guard is
gone. `save()` writes UTF-8 with `\n` endings on every platform, so a report
committed from Windows is byte-identical to one from Linux.

`Report` fields:

| Field | Type |
|---|---|
| `name` | `str \| None` |
| `elapsed_s` | `float` |
| `steps`, `calls` | `int` |
| `usage` | `Usage` |
| `cost_usd` | `float` |
| `limits` | `tuple[LimitStatus, ...]` — one per limit you actually set |
| `by_model` | `tuple[ModelSummary, ...]` |
| `by_tag`, `by_tool` | `tuple[AttributionSummary, ...]` |
| `unpriced_models` | `tuple[str, ...]` |
| `unpriced_calls` | `int` |
| `trip` | `LoopVerdict \| None` — set when a detector tripped |
| `tripped_reason` | `str \| None` — `"budget"`, `"tokens"`, `"steps"`, `"time"` |
| `pricing_as_of` | `str` — the snapshot date, e.g. `"2026-09"` |
| `pricing_sources` | `tuple[str, ...]` — config files that priced it |
| `checkpointed_calls` | `int` — calls inherited from a restored snapshot |

| Method | Does |
|---|---|
| `report.render(*, width=64, ascii_only=None)` | Text report. Deterministic: never reads the clock or the environment beyond a one-time Unicode probe, so it is safe to assert on. |
| `report.as_dict()` | JSON-serialisable form. |
| `Report.from_dict(data)` | Rebuild from `as_dict()` output — this is how the CLI renders a report written by another process. |

```python
LimitStatus:  name, used, limit, unit     # unit is "usd", "tokens" or "s"
```

`LimitStatus.fraction` is `used / limit` as a float in `[0, 1]`, and
`LimitStatus.exceeded` is `True` once `used` is past `limit`. `report.render()` uses
both — that is where the progress bar and the leading `!` come from. There is one
`LimitStatus` per limit you actually set, so a guard with only `max_usd` reports one
row, not four.

`by_model`, `by_tag` and `by_tool` are ordered by descending cost, then by name.

---

## Loop detection

```python
Detector                       # base class — subclass and implement observe()
RepeatDetector(max_repeats=3, window=12)
CycleDetector(min_cycle=2, max_cycle=4, repeats=2)
SimilarityDetector(threshold=0.95, window=8, max_similar=3, compare_chars=512)
NoProgressDetector(max_stagnant=6)

LoopMonitor(detectors=None)    # runs detectors in order, returns the first verdict
LoopVerdict(kind, detail, signature=None, count=0, step=None)
call_signature(name, args=None, *, max_len=512) -> str
```

`RepeatDetector`, `CycleDetector` and `SimilarityDetector` watch **tool calls** and
are the defaults for `Guard(detectors=...)`. `NoProgressDetector` watches the
**`progress()` channel** and is the default for `Guard(progress_detectors=...)`.

A detector is a state machine fed one signature per observation. Return a verdict to
stop the run, `None` to continue:

```python
from agentguard import Detector, LoopVerdict

class SchemaThrashDetector(Detector):
    name = "schema-thrash"

    def observe(self, signature, step):
        if signature.count("alter_table") >= 4:
            return LoopVerdict(self.name, "table altered 4 times", signature, 4, step)
        return None
```

Three optional members: `reset()` to clear state when a guard is reused,
`get_state()` / `set_state()` to survive a checkpoint. The base class raises
`NotImplementedError` for the last two rather than returning `{}`, so a detector
that cannot be checkpointed never *looks* checkpointed. A detector without them
does not block a restore; the guard warns once.

`LoopMonitor` feeds **every** detector every observation, even after one trips, so a
guard that logs and continues keeps its detectors in sync. It exposes
`detectors`, `observe(signature, step)`, `reset()`, `get_states()` and
`set_states(states)`.

```python
LoopVerdict:  kind, detail, signature, count, step
```

`kind` is the detector's `name` (`"repeat"`, `"cycle"`, `"similarity"`,
`"no-progress"`). `detail` is the sentence shown in the exception and the report.

---

## Cost accounting

```python
CostTracker(table=None, *, default_price=None, on_unknown_model="warn")
```

The object behind `guard.tracker`. Thread-safe: every mutation happens under one
re-entrant lock and every read returns a consistent snapshot.

| Member | Does |
|---|---|
| `record(*, model, usage, elapsed_s=0.0, tag=None, step=None, tool=None, meta=None, price=None)` | Account one call; returns a `CallRecord`. |
| `records` | `tuple[CallRecord, ...]` |
| `calls`, `total_usd`, `unpriced_calls`, `unpriced_models` | Totals |
| `usage`, `input_tokens`, `output_tokens`, `total_tokens` | Token totals |
| `by_model()`, `by_tag()`, `by_tool()` | Breakdowns, as dicts of summaries |
| `burn_rate_usd_per_step()` | Average dollars per accounted call, or `None` |
| `as_dict()` | JSON form, including every record |
| `get_state()`, `set_state()`, `restore()` | [Checkpointing](#checkpointing) |
| `price_table` | The table in use |

```python
Usage(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0)
```

`cached_input_tokens` is a **subset** of `input_tokens` (how both OpenAI and
Anthropic report cache hits), and `reasoning_tokens` is informational — providers
already include it in `output_tokens`. `total_tokens` is input + output.
`Usage.is_empty` is `True` when neither input nor output was reported, which is
usually a sign that extraction failed rather than that a call was free.
`Usage.__add__` and `as_dict()` are available.

```python
CallRecord: index, model, canonical_model, usage, cost_usd, at, elapsed_s,
            tag, step, tool, meta
```

`cost_usd` is `None` — not `0.0` — when the model had no price. `record.priced` is
`False` in that case, which is how the report knows to say *unpriced*.

```python
ModelSummary:        model, calls, input_tokens, output_tokens, cost_usd, unpriced_calls
AttributionSummary:  name, calls, input_tokens, output_tokens, cost_usd, unpriced_calls
UNATTRIBUTED = "(unattributed)"
```

`AttributionSummary.name` is a tag, a tool name, or `UNATTRIBUTED` for calls that
carried neither — so a breakdown's parts always add up to the whole.

---

## Pricing

```python
Price(input_per_1m, output_per_1m, cached_input_per_1m=None)
```

USD per 1,000,000 tokens. All three are validated at construction: a negative,
`NaN` or infinite rate, or a `bool`, raises `GuardConfigError`. `NaN` is the reason
this matters — `NaN > budget` is false, so one bad number would silently disarm
every budget check.

| Member | Does |
|---|---|
| `price.cost_usd(*, input_tokens=0, output_tokens=0, cached_input_tokens=0)` | The arithmetic |
| `price.worst_case_usd(*, input_tokens, max_output_tokens)` | What preflight uses |
| `price.as_dict()` | `{"input_per_1m": .., "output_per_1m": .., "cached_input_per_1m": ..}` |

```python
PriceTable(base=None, overrides=None, *, aliases=None, disable=(), sources=())
```

`base` replaces the bundled table entirely; `overrides` layer on top of it.
`PriceTable()` with no arguments is the bundled table.

| Member | Does |
|---|---|
| `resolve(model)` | `(canonical_name, price)` or `None` |
| `resolve_price(model)` | Just the `Price`, or `None` |
| `get(model)` | Exact normalized lookup — no alias, no version fallback |
| `origin(model)` | `"builtin"`, `"config"`, `"override"`, or `None` |
| `alias_of(model)`, `aliases`, `disabled`, `sources` | Provenance |
| `items()` | `(name, Price)` pairs for everything priced |
| `len(table)`, `iter(table)` | Number of entries / the model names |

Resolution is deliberately conservative, and this is worth internalising before you
wonder why a model came back unpriced:

```python
>>> table = PriceTable()
>>> table.resolve("GPT-4o")[0]
'gpt-4o'
>>> table.resolve("openai/gpt-4o")[0]        # gateway namespacing
'gpt-4o'
>>> table.resolve("gpt-4o-2024-08-06")[0]    # a trailing version segment
'gpt-4o'
>>> table.resolve("gpt-5.6-sol")[0]          # never gpt-5 — 6-sol is the model
'gpt-5.6-sol'
>>> table.resolve("some-new-model") is None
True
```

Names are folded by `normalize_model_key()`, which handles gateway namespacing
(`/`), version pinning (`@`), variant tags (`:free`) and dotted Bedrock/Vertex
prefixes. After an exact lookup fails, a **trailing version segment** may be
dropped — but only when dropping it cannot change the model family. An unknown model
stays unknown.

```python
DEFAULT_PRICING: dict[str, Price]   # the bundled table
PRICING_AS_OF = "2026-09"           # the month it was reviewed
```

The table is a snapshot of public list prices, reviewed against each provider's own
pricing page. Retired models keep their last published price rather than being
removed, because deleting a name silently makes every call to it unpriced. Two
consequences to plan around: a model **newer** than the snapshot is unpriced (safe,
but your cap will not fire for it), and a rate can change the day after a release.
`agentguard config set` is the two-command answer. See
[DETAILS.md](DETAILS.md#why-the-table-is-dated).

---

## Pricing config

```python
PricingConfig(models={}, aliases={}, disable=frozenset(), sources=())
```

The parsed form of the user's JSON file. `models` maps a name to a `Price`,
`aliases` maps a reported name to a priced one, `disable` is the set of bundled
models to treat as unpriced, and `sources` records which files it came from.
`PricingConfig.is_empty` is `True` when a config would change nothing — no models,
no aliases, nothing disabled — which is what `agentguard config path` uses to tell
you that no file is in effect.

```python
load_config(path=None, *, env=None) -> PricingConfig
parse_config(text, *, source=None) -> PricingConfig
config_paths(*, env=None, cwd=None, windows=None) -> tuple[Path, ...]
project_config_trusted(*, env=None) -> bool
initialize_config(path=None, *, user=False, project=False) -> Path
```

| Function | Does |
|---|---|
| `load_config()` | Discover and merge the config files; returns an empty config when there are none. With an explicit `path`, a missing file is an error rather than an empty config. |
| `parse_config(text)` | Parse config text directly — no filesystem. Useful for tests and tooling. |
| `config_paths()` | The paths that would be considered, in load order. |
| `project_config_trusted()` | Whether `$AGENTGUARD_TRUST_PROJECT_CONFIG` opts this process in. |
| `initialize_config()` | Write a starter file. Refuses to overwrite an existing one. |

Editing helpers, all of which write the file and return its path:

```python
set_model_price(path, model, input_per_1m, output_per_1m, *, cached_input_per_1m=None)
set_alias(path, name, target)
set_disabled(path, model, disabled=True)   # disabled=False enables it again
remove_entry(path, name)
```

```python
CONFIG_ENV_VAR = "AGENTGUARD_CONFIG"        # explicit path, or none/off/0
CONFIG_TRUST_ENV_VAR = "AGENTGUARD_TRUST_PROJECT_CONFIG"
```

Discovery order, most specific last: `$AGENTGUARD_CONFIG`, then the per-user file
(`%APPDATA%\agentguard\pricing.json`, or `$XDG_CONFIG_HOME/agentguard/pricing.json`),
then a project file (`agentguard.json` / `.agentguard.json`) — which is read **only**
when trusted, because it travels with the repository rather than with you. A project
file that repriced models to nothing would be a guard bypass performed with a data
file.

Every helper raises `GuardConfigError` on a malformed file, naming the file and the
key. See [DETAILS.md](DETAILS.md#the-config-file) for the file format and the trust
reasoning.

---

## Exceptions

```
GuardError
├── GuardConfigError          bad construction, bad config, unreadable snapshot
└── GuardTripped              base class for "a limit fired"
    ├── BudgetExceeded        max_usd
    ├── TokenLimitExceeded    max_tokens
    ├── StepLimitExceeded     max_steps
    ├── TimeLimitExceeded     max_seconds
    ├── LoopDetected          a detector returned a verdict
    └── GuardStopped          wrapper raised by on_trip="stop"
```

Everything the guard raises derives from `GuardError`, so one `except GuardError`
catches all of it. Catch `GuardTripped` when you want both a budget overrun and a
loop, and `GuardStopped` when you specifically want to know that a stopped guard
refused a call.

`GuardStopped.cause` is the original trip, and its `reason` mirrors it, so an
`except GuardTripped` handler written before `on_trip="stop"` existed still works.

| Exception | Extra attributes |
|---|---|
| `GuardTripped` | `reason` (`"budget"`, `"tokens"`, `"steps"`, `"time"`, `"loop"`) |
| `LoopDetected` | `kind`, `detail`, `signature`, `count`, `step` — copied from the verdict |
| `GuardStopped` | `cause` — the trip it wraps |
| `GuardConfigError` | none; the message names the offending value or file |

---

## Adapters

```python
from agentguard.adapters import GuardedClient, GuardedStream, guard_client
from agentguard.adapters.openai import guard_openai
from agentguard.adapters.anthropic import guard_anthropic

guard_client(target, guard, *, record_on=("create",), stream_on=(),
             depth=3, preflight=False, chars_per_token=3.0) -> GuardedClient
guard_openai(client, guard=None, *, preflight=False,
             chars_per_token=3.0, **guard_kwargs) -> GuardedClient
guard_anthropic(client, guard=None, *, preflight=False,
                chars_per_token=3.0, **guard_kwargs) -> GuardedClient
```

`guard_client` is the general form: it wraps `target` so that calls to any of the
`record_on` methods are recorded. The two provider helpers are preset names for it —
`guard_openai` and `guard_anthropic` add the streaming method names their SDKs use.
Pass `guard=` an existing guard, or pass `Guard` keyword arguments and one is built
for you.

Adapters are **pure duck-typing**: they never import a provider SDK, relying only on
attribute access down to a `create()` method. The same wrapper therefore covers
OpenAI, Azure OpenAI, Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq and
anything else following that convention.

`GuardedClient` is a transparent proxy — it forwards everything it does not
intercept — and is itself a context manager. `GuardedStream` wraps a streaming
response: chunks pass through untouched, and usage is recorded once when the stream
is drained. It is iterable, a context manager, and has `close()`; abandoning a
stream records whatever usage it saw.

`preflight=True` makes the wrapper refuse a call before issuing it, estimating input
tokens from the serialised prompt at `chars_per_token` characters per token. That is
a heuristic — a net for catastrophic calls, not an accounting figure. See
[DETAILS.md](DETAILS.md#pre-flight-refuse-a-call-before-paying-for-it).

---

## Integrations

```python
from agentguard.integrations.langgraph import guard_langgraph

guard_langgraph(guard=None, **guard_kwargs) -> Any
```

Returns a LangChain callback handler that records LLM calls (from `llm_output`
token usage, or per-generation `usage_metadata`) and fingerprints tool calls for the
loop detectors. Pass it as a callback:

```python
handler = guard_langgraph(Guard(max_usd=1.0, max_steps=25))
graph.invoke(inputs, config={"callbacks": [handler]})
```

The handler subclasses `BaseCallbackHandler` when langchain-core is installed and
falls back to a duck-typed class otherwise, so **no new dependency either way** —
and importing the module never fails because a framework is missing. `raise_error`
is set, so a trip stops the run instead of being logged away.

`agentguard.integrations` has no `__all__`; import the module you need by name.

---

## Decorators

```python
from agentguard import guarded, current_guard

@guarded(max_usd=0.50, max_steps=20)          # or guarded(guard=shared, ...)
def summarise(url: str) -> str:
    guard = current_guard()                   # the guard for this call
    ...

current_guard() -> Guard | None
```

`@guarded(...)` accepts the same keyword arguments as `Guard`, or `guard=` an
existing one. With no `guard=`, it creates a **fresh guard per call** — the right
default for a request handler, where one caller exhausting a budget must not stop
the next. Pass `guard=` when spend should accumulate across calls.

`current_guard()` returns the guard active in this context, or `None` outside any.
It is a `contextvar`, so it is correct under threads and asyncio.

---

## Command line

Installed as `agentguard`:

```bash
agentguard report run.json [--json]
agentguard pricing [MODEL] [--no-config] [--json]
agentguard config path|init|list
agentguard config set NAME INPUT OUTPUT [--cached C]
agentguard config alias NAME TARGET
agentguard config remove NAME
agentguard config disable NAME | enable NAME
```

`config` commands take `--user` / `--project` / `--file` to choose which file they
write. `agentguard report` renders a report saved by `guard.save()`, so the reading
process needs nothing installed but the CLI. Exit codes: `0` success, `1` error
(including an unknown model for `pricing MODEL`), `2` usage.

See [DETAILS.md](DETAILS.md#the-complete-cli) for full output examples.

---

## Export surface

Everything importable from the top-level package, and nothing else:

```python
import agentguard
agentguard.__all__      # 46 names
```

| Group | Names |
|---|---|
| Core | `Guard`, `Step`, `guarded`, `current_guard` |
| Errors | `GuardError`, `GuardTripped`, `GuardStopped`, `GuardConfigError`, `BudgetExceeded`, `TokenLimitExceeded`, `StepLimitExceeded`, `TimeLimitExceeded`, `LoopDetected` |
| Loop detection | `Detector`, `RepeatDetector`, `CycleDetector`, `SimilarityDetector`, `NoProgressDetector`, `LoopMonitor`, `LoopVerdict`, `call_signature` |
| Accounting | `Usage`, `CallRecord`, `CostTracker`, `ModelSummary`, `AttributionSummary`, `UNATTRIBUTED` |
| Pricing | `Price`, `PriceTable`, `DEFAULT_PRICING`, `PRICING_AS_OF` |
| Config | `PricingConfig`, `CONFIG_ENV_VAR`, `CONFIG_TRUST_ENV_VAR`, `load_config`, `parse_config`, `config_paths`, `initialize_config`, `project_config_trusted`, `set_model_price`, `set_alias`, `set_disabled`, `remove_entry` |
| Reporting | `Report`, `LimitStatus` |
| Metadata | `__version__` |

Importable from submodules rather than the top level:

| Name | From |
|---|---|
| `GuardedClient`, `GuardedStream`, `guard_client` | `agentguard.adapters` |
| `guard_openai` | `agentguard.adapters.openai` |
| `guard_anthropic` | `agentguard.adapters.anthropic` |
| `guard_langgraph` | `agentguard.integrations.langgraph` |
| `normalize_model_key`, `ORIGIN_BUILTIN`, `ORIGIN_CONFIG`, `ORIGIN_OVERRIDE` | `agentguard.pricing` |
| `SNAPSHOT_VERSION`, `extract_usage`, `extract_model` | `agentguard.tracker` |
| `default_detectors`, `default_progress_detectors` | `agentguard.loop` |

Anything not listed here is internal and may change in a patch release.

The package ships `py.typed`, so these annotations are what your type checker sees.
