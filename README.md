<div align="center">

# agent-guard

**Stop your agent before it burns $400 overnight.**

Budget caps, runaway-loop detection and circuit breakers for AI agents.
Zero dependencies. No provider SDK. No server. No telemetry.

[![CI](https://github.com/yaoyuxiang-gnn/agent-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/yaoyuxiang-gnn/agent-guard/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-guard.svg)](https://pypi.org/project/agent-guard/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-guard.svg)](https://pypi.org/project/agent-guard/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#design-principles)

[English](README.md) · [简体中文](README.zh-CN.md)

<br>

<img src="docs/demo.svg" width="591" alt="A terminal running examples/basic.py: an agent with a $0.05 budget is stopped on its third call, and the report shows the budget bar at 114%.">

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
pip install agent-guard
```

Python 3.10+. **No runtime dependencies** — not even a provider SDK.

---

## Quickstart

```python
from agent_guard import BudgetExceeded, Guard

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
    Pass Guard(pricing={...}) to include them.

----------------------------------------------------------------
  prices as of 2026-01 (indicative only)
```

Note the last block. **agent-guard never guesses a price.** A model it does not
know is counted as *unpriced* and reported loudly, because a safety tool that
silently assumes `$0.00` is worse than no safety tool at all.

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
from agent_guard import Detector, LoopVerdict

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

`on_trip="raise"` (the default) raises. The other two modes let a long-running job
stop cleanly instead of unwinding a stack:

```python
guard = Guard(max_usd=5.00, on_trip="stop", on_trip_callback=alert_page)

while not guard.stopped:
    with guard.step():
        ...
```

| Mode | Behaviour |
|---|---|
| `"raise"` | Raise the trip exception (default) |
| `"warn"` | Emit a `RuntimeWarning`, set `guard.stopped`, keep going — useful to *measure* before you enforce |
| `"stop"` | Set `guard.stopped` and call `on_trip_callback`, without raising |

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
from agent_guard.adapters.openai import guard_openai

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
from agent_guard.adapters.openai import guard_openai

client = guard_openai(OpenAI(), max_usd=1.0, max_steps=25)
response = client.chat.completions.create(...)   # recorded automatically
```

Adapters are pure duck-typing — they never import a provider SDK — so the same
wrapper covers **Anthropic, LiteLLM, OpenRouter, vLLM, Together, Groq and Azure
OpenAI**:

```python
from agent_guard.adapters.anthropic import guard_anthropic

client = guard_anthropic(Anthropic(), max_usd=2.0)
```

For a decorator instead of a context manager:

```python
from agent_guard import current_guard, guarded

@guarded(max_usd=0.50, max_steps=20)
def summarise(url: str) -> str:
    guard = current_guard()
    ...
```

`@guarded(max_usd=...)` creates a **fresh guard per call** — the right default for
a request handler, where one caller exhausting a budget must not stop the next.
Pass `guard=` an existing guard when spend should accumulate across calls.

---

## Command line

```bash
$ agent-guard report run.json          # render a report saved by guard.save(...)
$ agent-guard report run.json --json
$ agent-guard pricing gpt-4o
$ agent-guard pricing | head
```

```
$ agent-guard pricing gpt-4o
gpt-4o  (USD per 1M tokens, snapshot 2026-01)

  input        $2.5 / 1M
  output        $10 / 1M
  cached      $1.25 / 1M

  example costs
    1M in + 1M out            $12.5
    100k in + 20k out         $0.45
    10k in + 2k out          $0.045
```

`guard.save("run.json")` in the worker, `agent-guard report run.json` in CI. The
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
  ([`PRICING_AS_OF`](src/agent_guard/pricing.py)). Verify anything you bill on, and
  override what matters:
  ```python
  Guard(pricing={"my-finetune-v3": Price(3.00, 12.00)})
  ```
- **Not a substitute for provider-side spend limits.** Use both. agent-guard stops
  *your* loop; the provider's limit is what saves you when your process dies with a
  request already in flight.

### Honest limitations

- A response that reports no usage cannot be priced. agent-guard warns once and
  counts it as unpriced rather than inventing a number.
- Streaming responses carry no usage until drained. For streams, collect the final
  usage chunk yourself and call `guard.record(...)` once.
- Pre-flight input estimates are heuristic. See above.

---

## Documentation

| | |
|---|---|
| [`examples/basic.py`](examples/basic.py) | Budget cap, start to finish |
| [`examples/loop_detection.py`](examples/loop_detection.py) | All four detectors, plus a healthy run that must not trip |
| [`examples/wrapped_client.py`](examples/wrapped_client.py) | Zero-touch recording, and pre-flight refusal |
| [`examples/report_demo.py`](examples/report_demo.py) | A realistic multi-model run report |
| [ROADMAP.md](ROADMAP.md) | What is planned next |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

Every docstring example in the package runs as a test, so the documentation cannot
drift from the behaviour.

---

## Development

```bash
git clone https://github.com/yaoyuxiang-gnn/agent-guard
cd agent-guard

python -m unittest discover -s tests -t .   # no install required
pytest --cov=agent_guard                     # if you prefer pytest
python examples/basic.py
```

264 tests, no network, no fixtures to download. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
