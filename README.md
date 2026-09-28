<div align="center">

# agent-guard

**Stop your agent before it burns $400 overnight.**

Budget caps, runaway-loop detection and circuit breakers for AI agents.
Zero dependencies. No provider SDK. No server. No telemetry.

[![CI](https://github.com/yaoyuxiang-gnn/agent-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/yaoyuxiang-gnn/agent-guard/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-budget-guard-py.svg)](https://pypi.org/project/agent-budget-guard-py/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-budget-guard-py.svg)](https://pypi.org/project/agent-budget-guard-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#design-principles)

[English](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/README.md) · [简体中文](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/README.zh-CN.md)

<br>

<img src="https://raw.githubusercontent.com/yaoyuxiang-gnn/agent-guard/main/docs/demo.svg" width="591" alt="A terminal running examples/basic.py: an agent with a $0.05 budget is stopped on its third call, and the report shows the budget bar at 114%.">

</div>

---

## The problem

An agent that stops making progress does not stop running. It calls the same tool
with the same arguments, or bounces between two tools forever, paying for a fresh
context window on every pass.

A budget check that runs *after* the call can only tell you what you already spent.
`agent-guard` gives your agent loop the three things it is missing:

| | |
|---|---|
| **A ceiling** | `max_usd`, `max_tokens`, `max_steps`, `max_seconds` — re-checked on every recorded call |
| **A loop detector** | Four explainable detectors that fire *while* the agent is stuck, not after |
| **A receipt** | A cost report you can paste into an issue |

```bash
pip install agent-budget-guard-py
```

Python 3.10+. **No runtime dependencies** — not even a provider SDK.

> **About the name.** The PyPI distribution is `agent-budget-guard-py`; the import and the
> console script are both `agentguard`. `agent-guard` was already taken, and PyPI rejects a
> new name that differs from an existing one only by punctuation — so `agentguard` was not
> registrable either. Nothing you import or type is affected.

---

## Quickstart

```python
from agentguard import BudgetExceeded, Guard

guard = Guard(max_usd=1.00, max_steps=25, name="research-agent")

try:
    with guard:
        while True:
            with guard.step() as step:
                response = client.chat.completions.create(...)
                step.record(response)                    # tokens and cost, extracted
                with step.tool("search", {"q": query}):  # fingerprinted for loop detection
                    results = search(query)
except BudgetExceeded as exc:
    print(f"stopped: {exc}")

print(guard.report())
```

That is the whole API. Three things happen automatically:

1. **`step.record(response)`** pulls token counts and the model name out of *any*
   SDK response — OpenAI, Anthropic, Gemini, or a plain `dict`.
2. **`step.tool(name, args)`** builds a stable fingerprint of the call, so a repeat
   is caught **before** the side effect runs a fourth time.
3. **`guard.report()`** prints this:

```
agent-guard  nightly-indexer
================================================================
  wall time   12.4s            steps     13
  llm calls   13               tokens    214,600  (in 202,000 / out 12,600)
                               cached    36,000 input tokens

  limits
    budget   $0.3196 / $0.6               53.3%  [#########.......]
    tokens   214,600 / 500,000            42.9%  [#######.........]
    steps    13 / 25                      52.0%  [########........]

  by model
    gpt-4o           6 calls      $0.309     108,000 in / 8,400 out
    gpt-4o-mini      6 calls     $0.0106      54,000 in / 4,200 out
    acme-rerank-v3    1 call    unpriced      40,000 in / 0 out

  ! 1 call(s) had no known price and are excluded from the budget:
      acme-rerank-v3
    Price them with `agentguard config set <model> <input> <output>`,
    or pass Guard(pricing={...}) in code.

----------------------------------------------------------------
  prices as of 2026-01 (indicative only)
```

Note the last block. **agent-guard never guesses a price.** A model it does not
know is counted as *unpriced* and reported loudly, because a safety tool that
silently assumes `$0.00` is worse than no safety tool at all — and pricing that
model is one command away:

```bash
agentguard config set acme-rerank-v3 0.50 1.50
```

---

## What it catches

| Limit | Trips when |
|---|---|
| `max_usd=1.00` | Known spend exceeds one dollar |
| `max_tokens=500_000` | Input + output tokens exceed the allowance |
| `max_steps=25` | A 26th `guard.step()` is opened |
| `max_seconds=300` | Wall-clock time since the guard was created |
| loops | Any detector returns a verdict (see below) |

All limits are optional and independent. A guard with no limits still detects loops
and still produces a report.

---

## The four loop detectors

A budget cap eventually catches a runaway loop — after the money is gone. Loop
detection catches it *while* it is happening, and tells you why.

| Detector | Catches | Fires on |
|---|---|---|
| `RepeatDetector` | The same call, identical arguments | 3rd identical call |
| `CycleDetector` | `A, B, A, B` — two tools bouncing forever | 2nd full pass |
| `SimilarityDetector` | Paraphrasing: `search("python asyncio")` → `search("python asyncio ")` | 4th near-identical call |
| `NoProgressDetector` | A progress marker that never moves | 6th unchanged marker |

Every verdict is **explainable**, because "why did you kill my agent?" is the first
question anyone asks:

```
exact repeat                     -> repeat
                                    the same call appeared 3 times in the last 3 steps: search_web({"query":"weather in oslo"})
two-step ping-pong               -> cycle
                                    a 2-step pattern repeated 2 times: read_file({"path":"app.py"}) -> write_file({"body":...
paraphrased calls                -> similarity
                                    4 near-identical calls (>= 95% similar) in the last 4 steps: search_web({"query":"how do i...
no progress                      -> no-progress
                                    the progress marker did not change for 6 consecutive observations: {"rows_written":0}

healthy varied work              -> clean, as it should be
```

That last line matters as much as the others. Run it yourself — every example in
this repository works offline:

```bash
python examples/loop_detection.py
```

### Writing your own detector

A detector is a small state machine fed one signature per observation:

```python
from agentguard import Detector, LoopVerdict

class SchemaThrashDetector(Detector):
    """Trip when the agent migrates the same table back and forth."""

    name = "schema-thrash"

    def observe(self, signature, step):
        if signature.count("alter_table") >= 4:
            return LoopVerdict(self.name, "table altered 4 times", signature, 4, step)
        return None

guard = Guard(detectors=[SchemaThrashDetector()])
```

### Handling a trip gracefully

`on_trip="raise"` (the default) raises the moment a limit fires. `"stop"` records
the trip, calls `on_trip_callback` and lets the current step finish, so a
long-running job can flush, alert and clean up — and then raises `GuardStopped`
from every entry point afterwards, so a loop that forgets to check still stops
instead of spending on:

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

| Mode | Behaviour |
|---|---|
| `"raise"` | Raise the trip exception at the moment it fires (default) |
| `"warn"` | Emit a `RuntimeWarning`, set `guard.stopped`, keep going — useful to *measure* before you enforce |
| `"stop"` | Record the trip, finish the current step, then raise `GuardStopped` from the next `step()` / `record()` / `tool()` / `check()` / `preflight()` |

`GuardStopped` is a `GuardTripped` whose `cause` is the original trip, so
`except GuardTripped` still catches both a budget overrun and a loop. Accounting is
never skipped to raise: a call that already went out is recorded, then the
exception surfaces, because an unrecorded call is spent money the report denies.

---

## Pre-flight: refuse a call *before* paying for it

A post-hoc budget check can only report overspend. `preflight()` refuses a call
whose worst case will not fit in what is left:

```python
guard.preflight("gpt-4o", input_tokens=180_000, max_output_tokens=16_000)
# raises BudgetExceeded before the request is issued
```

Turn it on for a whole client with one flag:

```python
from agentguard.adapters.openai import guard_openai

client = guard_openai(OpenAI(), max_usd=0.05, preflight=True)

client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "summarise this 200 page report"}],
    max_tokens=60_000,
)
# BudgetExceeded: refused before spending: a gpt-4o call could reach $0.6,
# over the $0.05 limit (already spent $0)
```

Input tokens are estimated from the serialised prompt, because counting them
properly needs a provider tokenizer that agent-guard deliberately does not depend
on. Treat pre-flight as a net for catastrophic calls, not as an accounting figure.

---

## Three integration depths

Use as much or as little as you want. Nothing is required beyond the first level.

**1. Manual** — framework agnostic, works with anything, including a `while` loop
you wrote by hand:

```python
guard = Guard(max_usd=1.0)
guard.record("gpt-4o", input_tokens=1200, output_tokens=300)
guard.check()
```

**2. Structured** — one step per iteration, tools fingerprinted for you:

```python
with Guard(max_usd=1.0, max_steps=25) as guard:
    while True:
        with guard.step() as step:
            step.record(call_model(...))
            with step.tool("search", {"q": query}):
                ...
```

**3. Wrapped client** — every call accounted with no call-site changes:

```python
from openai import OpenAI
from agentguard.adapters.openai import guard_openai

client = guard_openai(OpenAI(), max_usd=1.0, max_steps=25)
response = client.chat.completions.create(...)   # recorded automatically
```

Adapters are pure duck-typing — they never import a provider SDK — so the same
wrapper covers **Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq and Azure
OpenAI**:

```python
from agentguard.adapters.anthropic import guard_anthropic

client = guard_anthropic(Anthropic(), max_usd=2.0)
```

**Streaming just works.** Pass `stream=True` and the response is wrapped in a
`GuardedStream`: chunks pass through untouched, and usage is recorded once, when
the stream is drained (or whatever was seen, if you abandon it early). Anthropic's
`messages.stream()` context manager and async clients (`async for`) are covered
the same way:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},   # so the final chunk carries usage
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
# recorded here, exactly once
```

**LangGraph** agents are guarded with one callback handler — LLM calls are
priced, and tool calls feed the loop detectors:

```python
from agentguard import Guard
from agentguard.integrations.langgraph import guard_langgraph

handler = guard_langgraph(Guard(max_usd=1.0, max_steps=25))
graph.invoke(inputs, config={"callbacks": [handler]})
```

For a decorator instead of a context manager:

```python
from agentguard import current_guard, guarded

@guarded(max_usd=0.50, max_steps=20)
def summarise(url: str) -> str:
    guard = current_guard()
    ...
```

`@guarded(max_usd=...)` creates a **fresh guard per call** — the right default for
a request handler, where one caller exhausting a budget must not stop the next.
Pass `guard=` an existing guard when spend should accumulate across calls.

---

## Your own models and prices

The bundled table knows public list prices. It cannot know your fine-tune, your
gateway's aliases, a regional endpoint, or a rate you negotiated. Those go in a
JSON file that `Guard` picks up automatically — no code change, no fork:

```bash
$ agentguard config set my-finetune-v3 3 12 --cached 0.3
$ agentguard config alias acme/fast claude-3-5-haiku
$ agentguard config disable gpt-4          # do not trust this bundled price
```

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
| `models` | *What does this model cost?* USD per 1M tokens, as an object or a short `[input, output]` array. A name that matches a bundled model **reprices** it. |
| `aliases` | *What is this name really?* Matched against the reported model string exactly, before any other interpretation — so `acme/fast` can point at a model that has a price. |
| `disable` | *Which bundled prices do I not trust?* A disabled model becomes **unpriced**: counted, reported, and excluded from the budget rather than billed at a number you rejected. |

The file is found in this order:

1. `$AGENTGUARD_CONFIG` — an explicit path (`none`/`off`/`0` disables config entirely)
2. `%APPDATA%\agentguard\pricing.json` on Windows, `$XDG_CONFIG_HOME/agentguard/pricing.json` (default `~/.config/...`) elsewhere — your own file, always read
3. `agentguard.json` or `.agentguard.json` in the working directory or the nearest parent — a **project** file, read only when you trust it with `AGENTGUARD_TRUST_PROJECT_CONFIG=1`

That third one is deliberate. A project file travels with the repository it sits in, so it is written by whoever wrote that repository — and reading it by default would make "clone this repo and run your agent in it" a way to reprice every model to nearly nothing, or `disable` the expensive ones so their calls stop counting against the budget. So it is skipped unless you ask (`agentguard config path` shows what is in effect and what was skipped, and skipping is reported once per process). Everything is overridable in code too, and code always wins over any file:

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

Inspect what is actually in effect before trusting a number:

```bash
$ agentguard pricing
41 models bundled, 2 configured (USD per 1M tokens, snapshot 2026-01)

  model                        input    output    cached  source
  claude-3-5-haiku              $0.8        $4     $0.08  builtin
  claude-3-5-sonnet               $3       $15      $0.3  builtin
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
means your file decided it. `agentguard config list` shows what each file
contains, `agentguard config path` shows which file is being read and why, and a
config typo — an unknown key, a negative or `NaN` rate, an alias pointing at a
model with no price, a `disable` entry that matches nothing — raises
`GuardConfigError` **at construction**, naming the file. A misconfigured price
must never quietly change what a budget means.

---

## Where the money went

Cost by model answers "what is expensive". Cost by step tag and by tool answers
"who is spending it", which is the question that changes what you do next:

```python
with guard.step(tag="retrieval") as step:
    with step.tool("search", {"q": query}):
        step.record(response)          # attributed to `search` *and* `retrieval`
```

`guard.report()` then carries all three breakdowns, and the text report prints the
ones that say something:

```
  by tag
    summarise        2 calls     $0.1468
    retrieval        1 call        $0.06
    (unattributed)   1 call     $0.00021

  by tool
    search           1 call       $0.145
    (unattributed)   3 calls      $0.062
```

`(unattributed)` is always included, so the parts add up to the run total rather
than telling a partial story that looks complete. A breakdown with a single bucket
is left out of the text report — one row would just repeat the total — and a long
one collapses its tail (`... 3 more`). `report().by_tag` / `.by_tool` return
`AttributionSummary` objects, `guard.to_json()` carries both lists, and
`guard.tracker.by_tool()` gives you the same numbers mid-run.

---

## Command line

```bash
$ agentguard report run.json          # render a report saved by guard.save(...)
$ agentguard report run.json --json
$ agentguard pricing gpt-4o
$ agentguard pricing                 # effective table, with a source per model
$ agentguard pricing --no-config     # bundled prices only
$ agentguard config path             # where config is read from, and what is ignored
$ agentguard config init             # write a starter file
$ agentguard config set NAME IN OUT [--cached C]
$ agentguard config alias NAME TARGET
$ agentguard config remove NAME
$ agentguard config disable NAME     # / agentguard config enable NAME
$ agentguard config list
```

```
$ agentguard pricing gpt-4o
gpt-4o  (USD per 1M tokens, snapshot 2026-01)

  input        $2.5 / 1M
  output        $10 / 1M
  cached      $1.25 / 1M

  example costs
    1M in + 1M out            $12.5
    100k in + 20k out         $0.45
    10k in + 2k out          $0.045

  source   builtin
```

`guard.save("run.json")` in the worker, `agentguard report run.json` in CI. The
reading process does not need agent-guard installed as a dependency of anything
except its own CLI.

---

## Design principles

These are the constraints the library is built to, and the reason it is safe to
drop into an existing agent stack.

**Zero dependencies.** No `pydantic`, no `httpx`, no provider SDK. Standard library
only. agent-guard can be added to a stack that vendors its dependencies, pinned to
an old Python, or shipped inside a Lambda without changing a lockfile.

**Never guess a price.** An unknown model is counted as unpriced and surfaced in
the report — never billed at a plausible-looking rate. Likewise, a response with no
usable token counts raises a warning rather than silently counting as `$0`.

**Fail at construction, not mid-run.** A bad `max_usd`, an unknown `on_trip` mode,
or a misconfigured detector raises `GuardConfigError` while the `Guard` is being
built. A safety tool must never be the thing that crashes at 3am.

**Thread safe.** An agent fanning out concurrent tool calls will not lose a record:
every mutation happens under one re-entrant lock, and live counters use
`contextvars`, which is `asyncio`-correct.

**Explainable, not clever.** Loop detection is stdlib `difflib` and sliding windows,
not an embedding model. Every verdict is a sentence a human can act on.

---

## What agent-guard is not

Being explicit about scope is cheaper than a GitHub issue.

- **Not an observability platform.** Nothing is sent anywhere. There is no server,
  no UI, no account, no background thread.
- **Not a proxy.** It does not sit between you and your provider, and it cannot see
  traffic it was not told about.
- **Not a tokenizer.** Bundled prices are an indicative snapshot
  ([`PRICING_AS_OF`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/src/agentguard/pricing.py)). Verify anything you bill on, and
  configure what matters — in a file, or in code:
  ```bash
  agentguard config set my-finetune-v3 3 12
  ```
  ```python
  Guard(pricing={"my-finetune-v3": Price(3.00, 12.00)})
  ```
- **Not a substitute for provider-side spend limits.** Use both. agent-guard stops
  *your* loop; the provider's limit is what saves you when your process dies with a
  request already in flight.

### Honest limitations

- A response that reports no usage cannot be priced. agent-guard warns once and
  counts it as unpriced rather than inventing a number.
- A stream is priced when it is drained; one that reports no usage warns instead
  of inventing a number. For OpenAI-compatible streams, remember
  `stream_options={"include_usage": True}`.
- Pre-flight input estimates are heuristic. See above.

---

## Documentation

| | |
|---|---|
| [`examples/basic.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/basic.py) | Budget cap, start to finish |
| [`examples/loop_detection.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/loop_detection.py) | All four detectors, plus a healthy run that must not trip |
| [`examples/wrapped_client.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/wrapped_client.py) | Zero-touch recording, and pre-flight refusal |
| [`examples/streaming.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/streaming.py) | Recording a streamed response, once, on drain |
| [`examples/langgraph_demo.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/langgraph_demo.py) | LangGraph callbacks: cost accounting plus tool-loop detection |
| [`examples/report_demo.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/report_demo.py) | A realistic multi-model run report |
| [`examples/custom_models.py`](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/examples/custom_models.py) | Custom models, prices, aliases and disabled entries |
| [ROADMAP.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/ROADMAP.md) | What is planned next |
| [CHANGELOG.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CHANGELOG.md) | Release history |

Every docstring example in the package runs as a test, so the documentation cannot
drift from the behaviour.

---

## Development

```bash
git clone https://github.com/yaoyuxiang-gnn/agent-guard
cd agent-guard

python -m unittest discover -s tests -t .   # no install required
pytest --cov=agentguard                     # if you prefer pytest
python examples/basic.py
```

473 tests, no network, no fixtures to download. See [CONTRIBUTING.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/LICENSE).
