<h1 align="center">agent-guard</h1>

<p align="center"><b>Stop your agent before it burns $400 overnight</b><br>
budget caps, loop detection and circuit breakers for AI agents<br>
zero dependencies · no proxy · nothing leaves your process</p>

<p align="center"><img src="https://raw.githubusercontent.com/yaoyuxiang-gnn/agent-guard/main/docs/demo.svg" width="600" alt="A terminal running examples/basic.py: an agent with a $0.05 budget is stopped on its third call, and the report shows the budget bar at 114%"></p>

An agent that stops making progress does not stop running. It calls the same tool
with the same arguments, or bounces between two tools forever, buying a fresh
context window on every pass. `agent-guard` is the part of your loop that notices —
and the receipt that tells you what it cost.

```bash
pip install agent-budget-guard-py
```

```python
from agentguard import Guard

guard = Guard(max_usd=1.00, max_steps=25)

with guard:
    while True:
        with guard.step() as step:
            response = client.chat.completions.create(...)
            step.record(response)                     # tokens and cost, extracted
            with step.tool("search", {"q": query}):   # fingerprinted for loop detection
                results = search(query)

print(guard.report())
```

Three things happen on their own: `step.record(response)` pulls tokens and the model
out of *any* SDK response, `step.tool(...)` fingerprints the call so a repeat is
caught **before** it runs a fourth time, and `guard.report()` prints the receipt.
Python 3.10+, no runtime dependencies — not even a provider SDK.

> **Jump to:** [See it work](#see-it-work) · [What it catches](#what-it-catches) ·
> [Wired into your stack](#wired-into-your-stack) · [Models it doesn't know](#models-it-doesnt-know) ·
> [Running it in CI](#running-it-in-ci) · [Honest answers](#honest-answers) ·
> [All the details](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md)

---

## The problem

A budget check that runs *after* the call can only tell you what you already spent.
By the time a cap notices a stuck agent, the money is gone — and the usual failure is
not a crash, it is a loop that looks busy.

Two things stop it, and they stop different things:

| | Catches | When it fires |
|---|---|---|
| **A ceiling** — `max_usd`, `max_tokens`, `max_steps`, `max_seconds` | spend that keeps climbing | after the call that crossed it |
| **A loop detector** — four of them, on by default | an agent repeating itself | **before** the repeat runs |

`agent-guard` does both, in your process, using the standard library. No proxy, no
server, no account, and nothing is sent anywhere.

## See it work

A research agent asked the same question three times. The third identical
`search_web` call is where it stopped — not because the budget ran out, but because
the call was identical to the two before it. This is the whole library in one run:

```python
guard = Guard(max_usd=5.00, name="research-agent")
with guard.step(tag="search") as step:
    step.record("gpt-4o", input_tokens=8_000, output_tokens=400)
    with step.tool("search_web", {"query": "weather in oslo"}):
        results = search("weather in oslo")        # 3rd time: stopped here
```

```text
turn 1: searched, spent $0.0240
turn 2: searched, spent $0.0480

LoopDetected: Loop detected [repeat]: the same call appeared 3 times in the
last 3 steps: search_web({"query":"weather in oslo"}).
```

```text
agentguard  research-agent
================================================================
  wall time   0ms             steps     3
  llm calls   3               tokens    25,200  (in 24,000 / out 1,200)

  limits
    budget   $0.072 / $5                   1.4%  [................]

  by model
    gpt-4o   3 calls      $0.072      24,000 in / 1,200 out

  tripped: loop [repeat] the same call appeared 3 times in the last 3 steps: ...
```

The difference between that and an overnight bill is $0.07. Both output blocks are
real — the exception line is wrapped to fit here and the report is truncated the same
way. Two examples run offline and show the same machinery:

```bash
python examples/loop_detection.py   # all four detectors, plus a healthy run that must not trip
python examples/basic.py            # a budget cap instead: stopped at $0.057 of a $0.05 limit
```

## What it catches

Every limit is optional and independent. A guard with no limits still detects loops
and still produces a report.

| Limit | Trips when |
|---|---|
| `max_usd=1.00` | Known spend exceeds one dollar |
| `max_tokens=500_000` | Input + output tokens exceed the allowance |
| `max_steps=25` | A 26th `guard.step()` is opened |
| `max_seconds=300` | Wall-clock time since the guard was created |

The four detectors, and what each one is actually for:

| Detector | Catches | Fires on |
|---|---|---|
| `RepeatDetector` | The same call, identical arguments | 3rd identical call |
| `CycleDetector` | `A, B, A, B` — two tools bouncing forever | 2nd full pass |
| `SimilarityDetector` | Paraphrasing: `search("python asyncio")` → `search("python asyncio ")` | 4th near-identical call |
| `NoProgressDetector` | A progress marker that never moves | 6th unchanged marker |

Every verdict is a sentence, because "why did you kill my agent?" is the first
question anyone asks. These are real, from `examples/loop_detection.py`:

```text
exact repeat                     -> repeat
                                    the same call appeared 3 times in the last 3 steps: search_web({"query":"weather in oslo"})
two-step ping-pong               -> cycle
                                    a 2-step pattern repeated 2 times: read_file({"path":"app.py"}) -> write_file(...)
paraphrased calls                -> similarity
                                    4 near-identical calls (>= 95% similar) in the last 4 steps: search_web(...)
no progress                      -> no-progress
                                    the progress marker did not change for 6 consecutive observations: {"rows_written":0}

healthy varied work              -> clean, as it should be
```

That last line matters as much as the others: a detector that fires on healthy work
is a detector you turn off. Each threshold is tuned so its scenario is reachable and
nothing else is, and every one is adjustable per guard.

## Wired into your stack

Three depths. Nothing is required beyond the first.

**1. Manual** — works with anything, including a `while` loop you wrote by hand:

```python
guard = Guard(max_usd=1.0)
guard.record("gpt-4o", input_tokens=1200, output_tokens=300)
guard.check()
```

**2. Structured** — one step per iteration, tools fingerprinted for you. This is the
Quickstart above.

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

**Streaming works.** With `stream=True` the response is wrapped, chunks pass through
untouched, and usage is recorded once when the stream drains. Anthropic's
`messages.stream()` and async clients (`async for`) are covered the same way.

**LangGraph** takes one callback handler:

```python
from agentguard.integrations.langgraph import guard_langgraph

handler = guard_langgraph(Guard(max_usd=1.0, max_steps=25))
graph.invoke(inputs, config={"callbacks": [handler]})
```

**Or a decorator**, for a request handler rather than a loop:

```python
from agentguard import current_guard, guarded

@guarded(max_usd=0.50, max_steps=20)
def summarise(url: str) -> str:
    guard = current_guard()
    ...
```

`@guarded(max_usd=...)` creates a **fresh guard per call** — the right default when
one caller exhausting a budget must not stop the next. Pass `guard=` a shared guard
when spend should accumulate across calls.

**Refuse a call before paying for it.** A post-hoc check can only report overspend;
`preflight()` refuses a call whose worst case will not fit in what is left:

```python
client = guard_openai(OpenAI(), max_usd=0.05, preflight=True)
# BudgetExceeded: refused before spending: a gpt-4o call could reach $0.6,
# over the $0.05 limit (already spent $0)
```

**Resume without refilling the budget.** An agent that checkpoints its own state can
checkpoint its spend too, so a restart does not hand it a fresh cap:

```python
write_checkpoint({"cursor": 41, "guard": guard.snapshot()})

# later, in a new process
guard = Guard.from_snapshot(read_checkpoint()["guard"], max_usd=5.0)
guard.remaining_usd      # what is actually left, not the full budget
```

Three decisions worth knowing, all in
[the details](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md#checkpointing):
detector windows survive a checkpoint (a loop that spans one is still a loop);
wall-clock time does not (`max_seconds` caps *this process*, and restoring elapsed
time would trip on time you never spent); and a snapshot it cannot read exactly is
**refused** rather than half-applied.

## Models it doesn't know

The bundled table carries **119 models** — current families from OpenAI, Anthropic,
Google, xAI, DeepSeek, Mistral and Qwen, dated `2026-09` in `PRICING_AS_OF`. It
cannot know your fine-tune, your gateway's aliases, or your negotiated rate:

```bash
agentguard config set my-finetune-v3 3 12 --cached 0.3
agentguard config alias acme/fast claude-3-5-haiku
agentguard config disable gpt-4          # do not trust this bundled price
```

**A model it has never heard of is never billed at a guessed rate.** It is counted
as *unpriced*, kept out of the budget arithmetic, and reported loudly — because a
safety tool that silently assumes `$0.00` is worse than no safety tool at all:

```text
  by model
    gpt-4o            1 call       $2.52   1,000,000 in / 2,000 out
    acme-rerank-v3    1 call    unpriced      40,000 in / 0 out

  ! 1 call(s) had no known price and are excluded from the budget:
      acme-rerank-v3
    Price them with `agentguard config set <model> <input> <output>`,
    or pass Guard(pricing={...}) in code.
```

A model *disabled* by `disable` becomes unpriced the same way, so the excluded spend
is visible instead of quietly billed at a number you rejected. Everything is
overridable in code, and code always beats a file.

## Running it in CI

`guard.save()` in the worker, `agentguard report` in CI — the reading process needs
nothing installed:

```bash
agentguard report run.json           # render a report saved by guard.save(...)
agentguard report run.json --json
agentguard pricing                   # effective table, with a source per model
agentguard pricing gpt-4o
agentguard config path               # where config is read from, what is ignored
```

## Honest answers

**Zero dependencies, and it stays that way.** No `pydantic`, no `httpx`, no provider
SDK — standard library only. CI fails the build if a runtime dependency ever appears
in the wheel, so it can be dropped into a stack that vendors its dependencies, pinned
to an old Python, or shipped inside a Lambda without touching a lockfile.

**A price table goes stale.** This one is dated, and a provider can change a rate or
retire a model the day after. Retired models keep their last published price rather
than being dropped, because removing a name silently turns every call to it unpriced.
Anything you bill on should be verified, and anything you depend on should be
configured. The honest fix — an opt-in, checksummed `pricing --update` — is not built
yet; it is first on the
[roadmap](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/ROADMAP.md).

**What it is not.** Not an observability platform (nothing is sent anywhere, there is
no server and no background thread), not a proxy (it cannot see traffic it was not
told about), not a tokenizer (pre-flight estimates are heuristic), and not a
substitute for provider-side spend limits — use both. agent-guard stops *your* loop;
the provider's limit is what saves you when your process dies with a request already
in flight.

**Limitations, stated plainly.** A response that reports no usage cannot be priced: it
warns once and counts as unpriced rather than inventing a number. A stream is priced
when drained, so for OpenAI-compatible streams pass
`stream_options={"include_usage": True}`. Pre-flight input counts are estimated from
the serialised prompt.

## Documentation

| | |
|---|---|
| [docs/DETAILS.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md) | Everything: every detector and its tuning, the full pricing config and its trust model, the checkpoint format, the complete CLI reference, design principles |
| [examples/](https://github.com/yaoyuxiang-gnn/agent-guard/tree/main/examples) | Eight runnable programs — budget cap, all four detectors, wrapped client, streaming, LangGraph, custom models, checkpointing, and a realistic report |
| [CHANGELOG.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CHANGELOG.md) | Release history |
| [ROADMAP.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/ROADMAP.md) | What is planned next |
| [CONTRIBUTING.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CONTRIBUTING.md) | The four constraints the library is built to |

Every docstring example in the package runs as a test, so the documentation cannot
drift from the behaviour. `python -m unittest discover -s tests -t .` runs the whole
suite — 563 tests, no network, no fixtures.

## License

MIT — see [LICENSE](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/LICENSE).
