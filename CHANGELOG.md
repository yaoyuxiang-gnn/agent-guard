# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The README claimed a bundled-model count the CLI did not print.** 0.3.0's
  `agentguard pricing` transcript said 113 models while the table held 119 — the
  refresh added six entries (the `claude-mythos` pair and `grok-4.20`) after the
  transcript was written, and nothing compared the two. Both READMEs are corrected,
  and a test now runs the CLI and compares its count against the quoted one, so the
  next refresh cannot reintroduce the drift. The published 0.3.0 page keeps the
  wrong number: PyPI files are immutable, so this lands in the next release.

## [0.3.0] - 2026-09-30

### Added

- **`Guard.snapshot()`: checkpoint a run's spend, and resume from it.** An agent
  that already checkpoints its own state can now checkpoint its budget too, so a
  resumed run keeps counting against what it already spent instead of starting
  from zero:

  ```python
  write_checkpoint({"cursor": 41, "guard": guard.snapshot()})

  # later, in a new process
  guard = Guard.from_snapshot(read_checkpoint()["guard"], max_usd=5.0)
  guard.remaining_usd      # what is actually left, not the full budget
  ```

  New: `Guard.snapshot()` / `as_snapshot()` / `restore()` / `Guard.from_snapshot()`,
  `CostTracker.get_state()` / `.restore()` / `.set_state()`, `Detector.get_state()` /
  `.set_state()` on all four built-in detectors, `LoopMonitor.get_states()` /
  `.set_states()`, `Report.checkpointed_calls`, and `agentguard.tracker.SNAPSHOT_VERSION`.

  The format is **counters, not records**: calls are grouped by
  `(model, tag, tool)`, so `by_model`, `by_tag` and `by_tool` all survive exactly
  and the per-call log does not. That is a size decision a checkpoint written every
  loop iteration forces — and it is why a restored report says
  `includes N call(s) restored from a checkpoint` rather than presenting inherited
  numbers as its own observations. Per-call costs in a restored run are their
  group's average, which is all a checkpoint knows.

  Three decisions worth naming:

  - **Detector windows survive; wall-clock time does not.** A loop that spans a
    checkpoint is still a loop, so the four built-in detectors serialise their
    sliding windows. `max_seconds` is not restored, because it caps how long *this
    process* may run and restoring an elapsed duration would make a resumed run
    trip on time it never spent. Money and steps accumulate; the clock restarts.
  - **A custom detector without `get_state` does not block the restore.** Its
    history is recorded as `null`, the budget still restores, and the guard warns
    once that a loop beginning before the checkpoint may need more observations.
    Refusing a budget over a detector would be the wrong trade.
  - **An unreadable checkpoint is refused, never half-applied.** A wrong format
    version, a negative count, a cost on a fully unpriced group or a call count
    that disagrees with its groups all raise `GuardConfigError` at restore, and the
    guard is left untouched — a budget that restores *approximately* is a budget
    that might not stop.

  Unpriced models stay unpriced through a round trip (`cost_usd=None`), so
  restoring can never turn forgotten money into budget headroom.
- **`examples/checkpointing.py`**, which resumes a run mid-flight and shows the
  carried-over report.

### Changed

- **The bundled price table was reviewed against current list prices.** The
  snapshot was nine months old (`2026-01`), and a model the table does not know is
  reported as *unpriced* and **excluded from the budget** — which is the safe
  behaviour, and also means a cap on a current model never fires. The table now
  carries 119 entries (from 41) and is dated `2026-09`, read from each provider's
  own pricing page and cross-checked against
  `https://openrouter.ai/api/v1/models`, with the provider's page winning where the
  two disagree.

  Added, one current family per provider at least: OpenAI `gpt-6-astra` /
  `gpt-6-sol` / `gpt-6.1-sol` / `gpt-6-luna`, the `gpt-5.1`–`gpt-5.6` line
  (including `gpt-5.6-cyber`) and `gpt-oss-120b` / `gpt-oss-20b`; Anthropic
  `claude-fable-5` / `-5.1`, `claude-mythos-5` / `-5.1`, `claude-opus-5` through
  `claude-opus-5.5`, `claude-sonnet-5` / `-5.5` and `claude-haiku-4.5`; Google
  `gemini-3` and `gemini-3.1`–`gemini-3.8`; DeepSeek `deepseek-v3.1` through
  `deepseek-v4-pro`; Mistral `mistral-medium-3.5`, `devstral` and the `ministral`
  line; xAI `grok-4.3` through `grok-4.7` and `grok-4.20`; the Qwen `qwen3.7` /
  `qwen3.8` families.

  Corrected: `gemini-2.5-pro` cached (0.31 → 0.125), `gemini-2.5-flash` cached
  (0.075 → 0.03), `deepseek-chat` (0.27/1.10 → 0.2574/1.0287), `mistral-small`
  (0.20/0.60 → 0.15/0.60), `qwen-plus` (0.40/1.20 → 0.26/0.78),
  `gpt-5.6-sol` (2.00/10.00 → 4.00/20.00, its published cyber-model rate), and the
  `mistral-large` / `codestral` cached rates, which were previously absent.

  Both spellings of a dotted version are listed where providers disagree —
  Anthropic's own model ids are dash-dated (`claude-haiku-4-5-20251001`) while the
  rate catalogues write `claude-haiku-4.5` — because normalization keeps the two
  apart and one spelling would otherwise resolve to nothing.

  **Retired models keep their last published price rather than being dropped.**
  Several entries are withdrawn, closed to new callers, or no longer listed at all
  (`grok-3`, `grok-4`, `deepseek-chat`, `deepseek-reasoner`, `gemini-2.0-flash`, the
  `claude-3-*` line, `gpt-4o`/`gpt-4.1`/`o3`/`o4-mini`, which OpenAI has scheduled
  for shutdown in late 2026). Removing a name is not neutral: it silently turns
  every call to that model *unpriced*, which excludes the spend from the budget —
  the opposite of what someone still running it needs. A stale number is a smaller
  error than a disabled cap. Two entries could not be read from a provider page and
  say so in the source: `claude-sonnet-5` (Anthropic prints its cache rate but
  leaves input/output blank; the base rate follows from Anthropic's documented 0.1x
  cache multiplier and two independent catalogues agreeing) and the `grok-4`/
  `grok-3` rates (their slugs now bill at `grok-4.3` rates after the 2026-05-15
  retirement).

  Nothing about *how* the table is used changed: an unknown model is still reported
  unpriced rather than guessed at, and `agentguard config set` still overrides any
  entry.
- **The price table now has tests that a refresh has to keep passing.** Every entry
  must resolve to itself, no two names may collapse onto one key after
  normalization, a cached rate may never exceed the fresh-input rate, and a set of
  long-standing families plus one current model per provider must stay priced. Two
  of these caught real mistakes while writing this table — a dash/dot mismatch and a
  cached rate above its input rate. The CLI test that asserted a hardcoded
  `"41 models bundled"` now derives the count, so a refresh no longer fails a test
  that says nothing about whether the CLI is correct.

### Fixed

- **The Chinese README's hero image was a relative path**, so it did not render
  from the repository's rendered view the way the English one does. It is an
  absolute URL now, like every link in `README.md`.

## [0.2.1] - 2026-09-30

### Fixed

- **The macOS test failure that kept CI red from 0.2.0 onward.** macOS reaches its
  temporary directory through `/var`, a symlink to `/private/var`, so `os.getcwd()`
  spells that path differently from `tempfile` — and two assertions compared the two
  spellings instead of the two files, which fails on the one platform whose temporary
  path contains a symlink. Paths are compared by identity now. The README's CI badge
  is green again, and macOS runs the examples for the first time.
- **`agentguard config set --file <relative path>` no longer denies its own write.**
  It wrote the file correctly and then reported "not in effect (no config file); set
  `AGENTGUARD_CONFIG` or edit that file instead" — the same spelling comparison, in
  the CLI — when the file it had just written was the project config, whose honest
  note is the one about trust.
- **The same command no longer prints a library warning.** `config set` leaked
  `RuntimeWarning: agentguard is ignoring the project config at ...` with a
  `cli.py:NNN:` source line above its own note. That warning is meant for library
  callers; the CLI says the same thing in its own voice.

### Changed

- **The README renders on PyPI.** PyPI shows `README.md` as the package description
  and does not resolve relative paths, so the published page had a broken hero image
  and fifteen dead links. Every link and image in `README.md` is an absolute URL now,
  and `docs/demo.svg` was regenerated against the current report — it is a screenshot
  of real program output, and 0.2.0's attribution feature had added a `by tag`
  breakdown to it.
- The roadmap records what 0.2 shipped and moves the checkpointing work to 0.3.

## [0.2.0] - 2026-09-28

### Added

- **Cost attribution by step tag and by tool.** The tracker recorded `tag` and
  `step` on every call but never aggregated them, so "which tool is eating my
  budget?" meant summing `guard.tracker.records` by hand. `report().by_tag` and
  `report().by_tool` now answer it, a wrapped call made inside
  `with guard.tool("search", ...)` is attributed to that tool automatically
  (innermost block wins, and an explicit `record(tool=...)` overrides it), and the
  text report prints both breakdowns — but only when there are at least two
  buckets, since one row would just repeat the run total. Calls that carried no tag
  or no tool land under `UNATTRIBUTED`, so the parts always add up to the whole
  instead of a breakdown that looks complete and is not. Also in `to_json()`, in
  `CostTracker.as_dict()`, and as `CostTracker.by_tag()` / `.by_tool()` mid-run.
- **User pricing config: your own models, your own prices.** The bundled table is
  a snapshot of public list prices, which is never enough — a fine-tune, a gateway
  alias, a negotiated rate. `Guard` now reads a JSON config file automatically, so
  none of that needs a code change:

  ```bash
  agentguard config set my-finetune-v3 3 12 --cached 0.3
  agentguard config alias acme/fast claude-3-5-haiku
  agentguard config disable gpt-4
  ```

  The file has three keys: `models` (a price as `{"input": 3.0, "output": 12.0,
  "cached_input": 0.3}`, a `[input, output]` array, or a `Price`), `aliases`
  (matched against the reported model string exactly, before any normalization) and
  `disable` (drop a bundled price you do not trust — the model becomes *unpriced*,
  so it is reported and excluded from the budget instead of billed at a number you
  rejected). Discovery order: `$AGENTGUARD_CONFIG` (an explicit path, or
  `none`/`off`/`0` to switch config off entirely), then the per-user file
  (`%APPDATA%\agentguard\pricing.json`, or `$XDG_CONFIG_HOME/...` elsewhere), then
  `agentguard.json` / `.agentguard.json` in the working directory or nearest
  parent. Files merge, with the more specific one winning on a conflict.
  A config in the **project tree** is only read when explicitly trusted with
  `$AGENTGUARD_TRUST_PROJECT_CONFIG=1`: it travels with the repository, so reading
  it by default would let whoever wrote that repository reprice models or disable
  the expensive ones — a guard bypass performed with a data file. A skipped project
  file is reported once per process, `agentguard config path` shows what was
  skipped and why, and writing one through the CLI says the same thing.
  New: `agentguard.config` (`PricingConfig`, `load_config`, `parse_config`,
  `config_paths`, `project_config_trusted`, and the editing helpers the CLI uses),
  `PriceTable.from_config`, `PriceTable.origin` / `aliases` / `disabled` /
  `sources`, `Guard(use_config=, config_path=, config=, aliases=, disable=)`,
  `Guard.price_table`, `Guard.pricing_config`, and `Report.pricing_sources` so a
  saved report names the config that priced it.
- **`agentguard config` CLI.** `path`, `init`, `set`, `alias`, `remove`, `disable`,
  `enable` and `list`, with `--user` / `--project` / `--file` to choose the file.
  `agentguard pricing` now shows the *effective* table — bundled plus configured —
  with a `source` column (`builtin` / `config`), the aliases, the disabled models
  and the config files in play; `--no-config` shows the bundled table alone and
  `--json` emits the lot for tooling.
- **`pricing=` accepts mappings.** `Guard(pricing={"m": {"input": 3, "output": 12,
  "cached": 0.3}})` — the long field names from `Price`'s `repr()` work too — and
  `Price.as_dict()` returns them.
- **`examples/custom_models.py`**, plus a "Your own models and prices" section in
  both READMEs.
- **Streaming usage capture.** A wrapped client's `create(..., stream=True)` now
  returns a `GuardedStream` that passes chunks through untouched and records the
  call once, when the stream is drained; abandoning a stream records whatever
  usage it saw, and a stream that reports nothing warns instead of counting `$0`.
  OpenAI-style final-chunk usage and Anthropic-style `message_start` /
  `message_delta` events are both understood, Anthropic's `messages.stream()`
  context manager is handled (including `get_final_message()`), and the same
  wrapping covers async clients (`await` a call, `async for` a stream).
  `agentguard.adapters.GuardedStream` is exported for streams you drive by hand.
- **LangGraph integration.** `agentguard.integrations.langgraph.guard_langgraph`
  returns a callback handler that records LLM calls (`llm_output` token usage or
  per-generation `usage_metadata`) and fingerprints tool calls for the loop
  detectors. It subclasses `BaseCallbackHandler` when langchain-core is present
  and falls back to a duck-typed class otherwise — no new dependency either way.
  `raise_error` is set so a trip stops the run instead of being logged away.
- **`agent-budget-guard-py[langgraph]` extra.**

### Changed

- **`on_trip="stop"` now actually stops.** It used to record the trip, set
  `guard.stopped` and rely on the caller's loop checking that flag — a loop that
  forgot kept spending, which is the failure this library exists to prevent. The
  trip is still recorded, `on_trip_callback` still fires once and the current step
  is still allowed to finish, but every entry point afterwards (`step`, `record`,
  `tool`, `observe`, `progress`, `check`, `preflight`, a wrapped client call)
  raises the new `GuardStopped`, whose `cause` is the original trip and whose
  `reason` mirrors it, so `except GuardTripped` handlers keep working. A stopped
  guard also refuses a wrapped client's next call, which means it costs nothing.
- **A call made after a trip is now recorded before the trip is raised.** In
  `"raise"` mode, `record()` used to raise on entry when the guard had already
  tripped, so a call that had really gone out was missing from the report — an
  under-reported overspend. Accounting happens first now, in every mode.

### Fixed

- **Impossible prices are rejected at construction.** A negative, `NaN` or infinite
  rate (or a `bool`, which `float()` would happily turn into `0.0`/`1.0`) used to be
  accepted. A `NaN` rate was the dangerous one: `NaN > budget` is false, so every
  budget check silently became a no-op. `Price` now validates its three rates, and
  pricing overrides accept mappings as well as tuples.
- **Shared guard entered concurrently no longer crashes.** `Guard.__enter__` kept
  its `ContextVar` tokens in one list on the instance, so two threads (or two
  asyncio tasks) inside `with guard:` at the same time popped each other's token
  and `__exit__` raised `ValueError: Token was created in a different Context`.
  The entry stack is now context-local, which also makes `@guarded(guard=shared)`
  safe on a thread pool.
- **`agentguard pricing <unknown>` suggestion had unbalanced braces.** The
  copy-paste hint printed `Guard(pricing={{'model': ...})`; it now prints valid
  Python, and a regression test pins the exact string.

## [0.1.0] - 2026-09-28

First release.

Installs as **`agent-budget-guard-py`**. `agent-guard` was already taken on PyPI, and
PyPI rejects a new project whose name differs from an existing one only by
punctuation — so the shorter `agentguard` was never registrable either. Only the
distribution name carries the long form: the import and the console script are both
`agentguard`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the spelling rule.

### Added

- **`Guard`** — a thread-safe run scope enforcing any combination of `max_usd`,
  `max_tokens`, `max_steps` and `max_seconds`, re-checked on every recorded call.
- **Three integration depths** — manual `guard.record(...)`, structured
  `with guard.step() as step:` blocks, and a transparent wrapped client.
- **`Guard.preflight(...)`** — refuse a call whose worst-case cost cannot fit in
  the remaining budget, *before* it is issued. A post-hoc check cannot do this.
- **Usage extraction from any provider response** — OpenAI (`prompt_tokens`),
  Anthropic (`input_tokens`, `cache_read_input_tokens`), Gemini
  (`prompt_token_count`, via `usage_metadata`), and plain `dict` payloads.
- **Four loop detectors**, all standard library only and all pluggable:
  - `RepeatDetector` — the same call, identical arguments, three times.
  - `CycleDetector` — a short repeating pattern such as `A, B, A, B`.
  - `SimilarityDetector` — near-duplicate calls that differ only trivially.
  - `NoProgressDetector` — an explicitly reported progress marker that stops moving.
- **`Detector`** — a documented extension point; writing your own is a dozen lines.
- **Explainable verdicts** — every trip carries a `kind` and a human-readable
  `detail`, surfaced verbatim in `LoopDetected` and in the report.
- **Three trip modes** — `"raise"` (default), `"warn"` for measuring before you
  enforce, and `"stop"` for long-running jobs that prefer to break out cleanly.
- **`on_trip_callback`** — fired exactly once, for alerting, and never allowed to
  mask the trip if it raises.
- **`Report`** — a text report with limit bars and a per-model breakdown, plus
  `as_dict()` / `from_dict()` / `to_json()` / `save()` for CI artefacts.
- **`agentguard` CLI** — `report` renders a saved JSON report, `pricing` shows the
  bundled table and worked example costs.
- **`agentguard.adapters`** — duck-typed OpenAI, Anthropic and generic client
  wrappers. No provider SDK is imported, so LiteLLM, OpenRouter, vLLM, Together,
  Groq and Azure OpenAI work through the same code path.
- **`@guarded`** — decorator form, with a fresh guard per call by default so one
  caller exhausting a budget cannot stop the next.
- **Bundled price snapshot** for ~40 models, with `PRICING_AS_OF` and an explicit
  warning that it is indicative rather than a billing source of truth.
- **Honest handling of the unknown** — an unpriced model is counted as unpriced and
  reported loudly; a response with no usable usage warns instead of counting as
  `$0`. Neither is ever guessed at.
- Zero runtime dependencies, `py.typed`, Python 3.10–3.13, tested on Linux, macOS
  and Windows.
- 265 tests, including every docstring example.

[Unreleased]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yaoyuxiang-gnn/agent-guard/releases/tag/v0.2.0
[0.1.0]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.1.0...v0.2.0
