# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-10-02

### Added


- **`agentguard pricing --update`: refresh the price table without waiting for a
  release.** The bundled table is dated, and a model released afterwards was billed
  as *unpriced* — safe, because agent-guard refuses to guess a rate, but not useful,
  because an unpriced call does not move the budget. This is the first item on the
  roadmap, and the last piece of it that was missing.

  ```bash
  agentguard pricing --update              # download, verify, cache
  agentguard pricing --from-file cat.json  # ...or import one you already have
  agentguard pricing --status              # is a snapshot in effect, and from where
  agentguard pricing --remove              # forget it
  ```

  Four decisions shape it:

  * **Never automatic.** This is the only command in agent-guard that makes a network
    request. Nothing fetches at import, on a timer, or in a background thread —
    `urllib` is imported by `agentguard.snapshot` and reached from nowhere else. A
    hidden request during `import agentguard` would be a worse bug than a stale table.
  * **Checksummed, and verified on every read.** The snapshot records the SHA-256 of
    its own model map and re-checks it on load. An edited, truncated or corrupted file
    raises rather than repricing models; silently falling back would leave you
    believing prices had been refreshed when the bundled table is what is in effect.
  * **Underneath everything you configured.** Precedence is bundled table → snapshot →
    per-user config → project config → `Guard(pricing=...)`. A downloaded catalogue can
    reprice a bundled model, which is the point, but it can never override a rate you
    set deliberately, and it cannot add an alias or a `disable`.
  * **Honest about what it cannot read.** An entry with no flat per-token rate is
    skipped rather than guessed at. Catalogues publish `-1` to mean "priced
    elsewhere", and reading that as a rate would bill those calls at a *negative*
    cost. Variants are skipped too — `model:batch` at half price, `model:free` at
    nothing — because they normalize to the same model name as the standard SKU and
    only one of the two can be stored. Keeping the cheaper one was this code's first
    behaviour, and it priced most of a real catalogue at OpenRouter's *batch* rate;
    a run billed at list would then have had a cap firing at twice the spend it
    thought it was tracking. A `:free` model is therefore *unpriced* rather than
    `$0`, the same conservative direction this library takes everywhere else.
    Skipped entries are counted in the output.

  Per-token strings are converted to per-1M rates in `Decimal` and rounded, so a
  catalogue that says `0.0000002` yields exactly `0.2` rather than
  `0.1999999999999998`. Parsed snapshots are memoised against the file's size and
  modification time, because a guard is constructed per request and re-parsing a
  400-model catalogue on every one cost 6.7ms; editing the file still takes effect on
  the next guard.

  `Guard(use_snapshot=False)` skips the layer. `$AGENTGUARD_CONFIG=none` disables it
  along with every other config file, and `$AGENTGUARD_CONFIG=<path>` replaces
  discovery entirely — that variable means "use exactly this", which cannot also mean
  "and also read that".

- **`scoped_budgets`: a dollar cap per tool or per tag.** A run-level `max_usd` can
  only tell you the run got expensive; it cannot tell you *which part* of it did, so
  one tool in a retry storm can spend the entire allowance before the ceiling notices.
  `Guard(max_usd=5.0, scoped_budgets={"tool:search": 1.0, "tag:index": 0.5})` trips
  `BudgetScopeExceeded`, which carries `scope`, `name`, `spent_usd` and `limit_usd`,
  and the report prints the scope as its own limit row.

  Scope keys are validated at construction, including a `NaN` limit — every comparison
  against `NaN` is false, so a `NaN` cap would be a limit that reports it is working
  while doing nothing. The run-level limit is checked first, so when both are over the
  trip is `BudgetExceeded`. Totals accumulate as calls arrive rather than by scanning
  the run, so the check is O(1) per call, and they are rebuilt on restore so a cap
  counts spend inherited from a checkpoint. `guard.scope_spend(scope, name)` reads a
  total whether or not it is capped.

- **`guard.bind(func)` and `guard.context()`: attribution on worker threads.** No
  thread inherits a `contextvars` context — not `threading.Thread`, not a
  `ThreadPoolExecutor` worker — so a call recorded on one was counted but belonged to
  no step and no tool. The money was never lost; the attribution was, and the
  consequence was that `by_tag`/`by_tool` under-reported and a `scoped_budgets` cap on
  that tool silently never fired. Both capture at the moment they are called, which is
  what makes them work from the main thread, and both are no-ops with no ambient
  context rather than errors.

- **`preflight(strict=True)`: refuse a model that cannot be priced.** The lenient
  default is defensible — an unbounded cost cannot be *shown* to exceed the budget —
  but it left a hole in a gate callers treat as a hard stop, since an expensive model
  merely absent from the price table sailed through. `strict=True` refuses it with
  `BudgetExceeded.unpriced_model` set. `Guard(preflight_strict=True)` makes that the
  default, and it is already the default when `on_unknown_model="error"`.

- **Bedrock's Converse usage shape.** `inputTokens` / `outputTokens` /
  `cacheReadInputTokens` / `reasoningTokens` are understood, so a Bedrock agent is
  priced instead of landing in the unpriced report. `CONTRIBUTING.md` asks for exactly
  this — extend `extract_usage()` rather than special-casing an adapter — and these
  were the fields that were missing.

- **`examples/scoped_budgets.py`** and **`examples/price_snapshot.py`**, both offline.
  `tests/test_readme_claims.py` now asserts that every file in `examples/` is named in
  the README's count, run by CI, and listed in the Makefile, because an example nothing
  runs is documentation that has already drifted.



- **Async coverage for the decorator.** `tests/test_decorators.py` had no `async`
  test at all, which is why the bug above shipped: the guard being live inside a
  coroutine, across an `await`, across an async generator, and cleared on both normal
  completion and an exception are now all asserted.
- **`tests/test_readme_claims.py` gains a `DocumentedNumbersTests` case.** The suite
  checked detector names and defaults and `test_api_reference.py` checked names and
  signatures, but the *numbers* in the prose were checked by nothing, which is how
  `docs/DETAILS.md` came to advertise 551 tests against 588 actually running. The
  documented truncation bound, the documented `compare_chars`, the README's limit
  table and its `requires-python` claim are now asserted against the library.


### Fixed


- **`BudgetExceeded` gained `unpriced_model`**, set only by a strict pre-flight
  refusal, so a handler can tell "this would overspend" from "this cannot be priced".
- **`docs/API.md` documents the new parameters and members**, and a test now fails if
  `Guard` gains an argument the reference never names. It caught all three of this
  release's new ones.
- **`ROADMAP.md`** moves price refreshing, scoped budgets and the smaller items to a
  "Shipped in 0.4" section, and replaces the fulfilled item with what is genuinely
  next: a `--check` drift report, per-provider sources, and `--url` presets.


- **Truncating a fingerprint could make two different tool calls identical, so the
  default detectors stopped healthy agents.** `call_signature()` kept the first 512
  characters of a call and replaced the rest with a note — and the characters it
  dropped were frequently the only ones that differed. An agent indexing documents
  that share a boilerplate body (`{"body": "…500 bytes…", "id": 0}`, `id: 1`,
  `id: 2`) produced *one* fingerprint for four different documents, so the third was
  reported as `LoopDetected [repeat]` and the run was stopped. `id` and `path` sort
  after the long fields in canonical JSON, so a head-only truncation discarded
  precisely the argument the call was keyed on.

  A truncated fingerprint now keeps a head, a tail, **and the digest of the whole
  payload**, which makes "different call, different fingerprint" hold at any length.
  The bound dropped from 512 to 256 characters. This is the failure mode
  [CONTRIBUTING](CONTRIBUTING.md) ranks above all others — a detector that fires on
  healthy work gets switched off, and then it catches nothing.

- **`SimilarityDetector` no longer fires on work that is merely similar in shape.**
  Two problems, both from applying a percentage threshold to text whose length the
  caller controls:

  *Length dilution.* On a two-thousand-character call a real difference of twenty
  characters still scores 0.99, so *every* pair of similar calls cleared a 95% bar —
  the same indexing scenario above, on a `SimilarityDetector` that had survived the
  fingerprint fix. `compare_chars` now defaults to 128 and the comparison key folds
  in the digest of the whole signature, so the ratio measures similarity of calls
  rather than of whatever survived truncation.

  *`difflib` autojunk.* `SequenceMatcher` discards characters making up more than 1%
  of a sequence longer than 200, so a payload of repeated characters — base64, a CSV
  column, whitespace-padded text — had most of itself declared "junk" and scored
  **0.76** against a near-identical string where an honest comparison scores 0.96.
  The same pair scored 0.76 or 0.96 depending only on how repetitive the payload was.
  `autojunk` is now off.

  Whitespace-only differences still score 1.0, so the paraphrase case the detector
  exists for — `search("python asyncio")` then `search("python asyncio ")` — trips on
  the fourth call exactly as documented.

- **`@guarded` did nothing on an `async def` function.** The wrapper entered the guard,
  called the function and exited — but calling an `async def` only builds a coroutine,
  so the body ran *after* `__exit__`, where `current_guard()` returns `None`. Every
  call inside an async agent therefore went unrecorded:

  ```python
  @guarded(max_usd=1.0)
  async def agent():
      current_guard().record(...)   # AttributeError: 'NoneType' has no attribute 'record'
  ```

  Async functions now get an async wrapper, so the guard stays entered across every
  `await`; `async` generators get one too, since a generator body does not run when
  the function is called either. `asyncio.iscoroutinefunction()` still reports `True`
  for the decorated function. The rest of the package already handled async correctly
  (`GuardedClient`, `GuardedStream`) — only the decorator was missed, and no test
  covered it.

- **`on_unknown_model="error"` lost the call it was refusing.** The error was raised
  *instead of* creating the record, so a call that had already been made and already
  been paid for disappeared from `guard.calls`, from `report().by_model` and from the
  report entirely — `spent_usd` and `remaining_usd` both behaving as if it had never
  happened. That is the definition
  [SECURITY.md](SECURITY.md) gives for the most serious bug class in this project: a
  documented usage pattern where an LLM call is made but not accounted for.

  Accounting now happens first and the error is raised afterwards, matching the rule
  `record()` already followed for a budget trip: **a call that has gone out is
  recorded before any exception surfaces.** The call is counted as *unpriced*, never
  as `$0` — `spent_usd` does not move for it — and `report().unpriced_models` names
  it. `CostTracker` gains `record_with_policy()`, which returns
  `(CallRecord, GuardConfigError | None)` for callers that want to handle the failure
  themselves; `CostTracker.record()` is unchanged in signature and still raises.

### Changed


- **`BudgetExceeded` gained `unpriced_model`**, set only by a strict pre-flight
  refusal, so a handler can tell "this would overspend" from "this cannot be priced".
- **`docs/API.md` documents the new parameters and members**, and a test now fails if
  `Guard` gains an argument the reference never names. It caught all three of this
  release's new ones.
- **`ROADMAP.md`** moves price refreshing, scoped budgets and the smaller items to a
  "Shipped in 0.4" section, and replaces the fulfilled item with what is genuinely
  next: a `--check` drift report, per-provider sources, and `--url` presets.


- **Truncating a fingerprint could make two different tool calls identical, so the
  default detectors stopped healthy agents.** `call_signature()` kept the first 512
  characters of a call and replaced the rest with a note — and the characters it
  dropped were frequently the only ones that differed. An agent indexing documents
  that share a boilerplate body (`{"body": "…500 bytes…", "id": 0}`, `id: 1`,
  `id: 2`) produced *one* fingerprint for four different documents, so the third was
  reported as `LoopDetected [repeat]` and the run was stopped. `id` and `path` sort
  after the long fields in canonical JSON, so a head-only truncation discarded
  precisely the argument the call was keyed on.

  A truncated fingerprint now keeps a head, a tail, **and the digest of the whole
  payload**, which makes "different call, different fingerprint" hold at any length.
  The bound dropped from 512 to 256 characters. This is the failure mode
  [CONTRIBUTING](CONTRIBUTING.md) ranks above all others — a detector that fires on
  healthy work gets switched off, and then it catches nothing.

- **`SimilarityDetector` no longer fires on work that is merely similar in shape.**
  Two problems, both from applying a percentage threshold to text whose length the
  caller controls:

  *Length dilution.* On a two-thousand-character call a real difference of twenty
  characters still scores 0.99, so *every* pair of similar calls cleared a 95% bar —
  the same indexing scenario above, on a `SimilarityDetector` that had survived the
  fingerprint fix. `compare_chars` now defaults to 128 and the comparison key folds
  in the digest of the whole signature, so the ratio measures similarity of calls
  rather than of whatever survived truncation.

  *`difflib` autojunk.* `SequenceMatcher` discards characters making up more than 1%
  of a sequence longer than 200, so a payload of repeated characters — base64, a CSV
  column, whitespace-padded text — had most of itself declared "junk" and scored
  **0.76** against a near-identical string where an honest comparison scores 0.96.
  The same pair scored 0.76 or 0.96 depending only on how repetitive the payload was.
  `autojunk` is now off.

  Whitespace-only differences still score 1.0, so the paraphrase case the detector
  exists for — `search("python asyncio")` then `search("python asyncio ")` — trips on
  the fourth call exactly as documented.

- **`@guarded` did nothing on an `async def` function.** The wrapper entered the guard,
  called the function and exited — but calling an `async def` only builds a coroutine,
  so the body ran *after* `__exit__`, where `current_guard()` returns `None`. Every
  call inside an async agent therefore went unrecorded:

  ```python
  @guarded(max_usd=1.0)
  async def agent():
      current_guard().record(...)   # AttributeError: 'NoneType' has no attribute 'record'
  ```

  Async functions now get an async wrapper, so the guard stays entered across every
  `await`; `async` generators get one too, since a generator body does not run when
  the function is called either. `asyncio.iscoroutinefunction()` still reports `True`
  for the decorated function. The rest of the package already handled async correctly
  (`GuardedClient`, `GuardedStream`) — only the decorator was missed, and no test
  covered it.

- **`on_unknown_model="error"` lost the call it was refusing.** The error was raised
  *instead of* creating the record, so a call that had already been made and already
  been paid for disappeared from `guard.calls`, from `report().by_model` and from the
  report entirely — `spent_usd` and `remaining_usd` both behaving as if it had never
  happened. That is the definition
  [SECURITY.md](SECURITY.md) gives for the most serious bug class in this project: a
  documented usage pattern where an LLM call is made but not accounted for.

  Accounting now happens first and the error is raised afterwards, matching the rule
  `record()` already followed for a budget trip: **a call that has gone out is
  recorded before any exception surfaces.** The call is counted as *unpriced*, never
  as `$0` — `spent_usd` does not move for it — and `report().unpriced_models` names
  it. `CostTracker` gains `record_with_policy()`, which returns
  `(CallRecord, GuardConfigError | None)` for callers that want to handle the failure
  themselves; `CostTracker.record()` is unchanged in signature and still raises.


- **Documentation corrected against the library.** `docs/DETAILS.md` said 551 tests;
  `decorators.py` pointed at a `Guard.current` that has never existed; `SECURITY.md`
  listed 0.2.x as the supported line while 0.3.2 was current; `CONTRIBUTING.md`
  gave a `pricing <model>` command without saying it was the lookup rather than the
  writer. `docs/API.md` documents the new truncation guarantee, the
  `on_unknown_model` order of events, and `record_with_policy()`.

- **`docs/API.md`: a reference for every public name.** The README is a tour and
  `DETAILS.md` explains the reasoning, but nothing listed the interface —
  `Guard`'s 19 arguments, the parameters of `record()`, which exception carries
  which attribute, what is importable from where. A reader who wanted to *look
  something up* had to read the source.

  Organised by area rather than alphabetically, because that is how it gets used:
  Guard, Step, recording, checkpointing, reporting, loop detection, accounting,
  pricing, pricing config, exceptions, adapters, integrations, decorators, CLI.
  Each section gives the signature with every parameter named, then the behaviour
  worth knowing before relying on it — the modes of `on_trip`, that
  `CallRecord.cost_usd` is `None` rather than `0.0` for an unpriced call, that
  version stripping will never bill `gpt-5.6-sol` at the `gpt-5` rate.

  One entry states a limitation plainly rather than a feature: `preflight()` returns
  `0.0` and does **not** refuse for a model with no price, because an unbounded cost
  cannot be shown to exceed the budget. Anyone trusting preflight as a hard gate
  should know that before they rely on it.

- **`tests/test_api_reference.py`, which checks the reference against the library.**
  A hand-written API document goes stale silently — nothing imports it, so nothing
  breaks when a parameter is renamed. Fifteen tests read the document and compare:
  every name in the export table is really exported and no exported name is missing
  from it, every submodule name lives in the module claimed, every documented
  signature's parameters exist on the real callable, the exception tree matches, and
  every public member of the documented classes is mentioned somewhere.

  The examples are executed, not merely printed: each carries its expected output,
  and `tests/test_doctests.py` runs them the same way it runs docstring examples, so
  the `12.5` the reference says `preflight()` returns is asserted against the real
  return value. The blocks are fragments — only the first has its imports — so the
  runner assembles a namespace from the document's `import` lines plus the whole
  public API, which is what a reader would have in scope.

  Writing these found five genuine omissions immediately — `PriceTable.items()`,
  `Usage.is_empty`, `LimitStatus.exceeded`, `LimitStatus.fraction` and
  `PricingConfig.is_empty` were all public and all undocumented — plus an unescaped
  `|` that was breaking one of the reference's own tables. The signature parser is
  deliberately conservative: it treats a call as a declaration only when the shape
  says so, so `guard.progress({"rows_written": 120})` in an example is not read as a
  signature, and it fails loudly rather than silently checking nothing.

## [0.3.2] - 2026-09-30

Packaging and documentation. No library behaviour changed; the wheel is the same
code as 0.3.1.

### Removed

- **The asset-generation scripts no longer ship, and one maintainer-only asset is
  gone from the repository.** `tools/make_demo_svg.py` and
  `tools/make_social_preview.py` rendered the README's screenshot and the
  repository's social-preview card — repository furniture, not library code — and
  `docs/social-preview.png` existed only for the second one to produce.

  Scope of the change: the **wheel was never affected**, because it already packages
  `src/agentguard` alone, so `pip install` was always unaffected. What changes is the
  **sdist**, which shipped `/tools` and a `docs/` holding the preview card alongside
  the real documentation. `tools/` is out of the include list, and `docs/` is down to
  the two files the project actually documents with: `DETAILS.md` and the README's
  screenshot.

  The include list is now exhaustive on purpose — anything not named there is not
  published — so a new top-level directory cannot reach a release by default.
  `tests/test_packaging.py` pins that decision, including what a wholesale `docs/`
  include would otherwise let through; verified by restoring the deleted files and
  watching `test_docs_carries_only_real_documentation` fail on exactly the preview
  card.

  `docs/demo.svg` stays, and is now a checked-in asset with no generator in the
  repository. CONTRIBUTING says so, rather than pointing at a `make demo` target that
  no longer exists.

  **The two scripts are gone from the repository's history as well**, not only from
  `HEAD`: every commit and every tag was rewritten so that no reachable revision
  contains them. The version tags therefore point at new commits, which is why the
  release workflow now tolerates being re-run for a version PyPI already has.

### Changed

- **The release workflow no longer fails when re-run for a published version.**
  `pypa/gh-action-pypi-publish` is given `skip-existing: true`, and the GitHub
  release job runs even when the publish step had nothing to upload. Re-tagging an
  old commit is exactly that case, and without this a rewritten tag would have
  produced a red publish job *and* left the release page un-rebuilt, since the
  release is created downstream of the publish.

  Deliberately not a pre-flight "is this version on PyPI?" query: the two available
  queries disagree. For this project the per-version JSON endpoint reported a single
  release while every artifact of all four versions still downloaded, and the
  `/simple/` index served both answers at different times. Guessing from a cached
  answer risks attempting an upload PyPI will reject; the publisher knows what it is
  uploading.
- **The pre-commit ruff pin moved from `v0.6.9` to `v0.16.9`,** and the `dev` extra
  floor from `ruff>=0.6` to `ruff>=0.16`, so both agree with the release CI installs.
  A stale pin here is not merely a missing lint: ruff's *formatter* output changes
  between versions, so the hook could rewrite a file into a shape that
  `ruff format --check` then rejects — a failure with no visible cause.

## [0.3.1] - 2026-09-30

Documentation only: no library behaviour changed, and the wheel is the same code as
0.3.0.

### Fixed

- **The README claimed a bundled-model count the CLI did not print.** 0.3.0's
  `agentguard pricing` transcript said 113 models while the table held 119 — the
  refresh added six entries (the `claude-mythos` pair and `grok-4.20`) after the
  transcript was written, and nothing compared the two. Both READMEs are corrected,
  and a test now runs the CLI and compares its count against every claim the README
  makes about it, so the next refresh cannot reintroduce the drift. The published
  0.3.0 page keeps the wrong number: PyPI files are immutable, so this lands in the
  next release.

### Changed

- **Both READMEs were rewritten around what a reader actually needs, and the
  reference material moved to `docs/DETAILS.md`.** The old README was 600 lines and
  asked the reader to absorb the whole API before showing them a reason to care: it
  opened with a problem statement, then a quickstart, and only reached the four loop
  detectors — the feature that distinguishes this from a spend counter — two hundred
  lines in.

  The new one is 327 lines and leads with evidence. Real output from
  `examples/loop_detection.py` shows a stuck agent stopped at the third identical
  call, with the report and the cost it did *not* incur; the four detectors are a
  table with the exact call each one fires on; and the integration recipes are
  ordered by how much you have to change. Everything else — detector tuning, the
  full pricing config and its trust model, the checkpoint format, the complete CLI,
  the design principles, the limitations in full — is in `docs/DETAILS.md`, which
  ships in the sdist and is one link away from six places in the README.

  Two things the rewrite added rather than moved:

  - **A "Honest answers" section**, because the questions a cautious reader has are
    answered better up front than in an issue: what happens when a price goes stale,
    what the library deliberately is not, and what its limitations actually are.
    Most of it was already documented — it just was not where anyone would look.
  - **Claims that are checked.** The detector table says each detector fires on its
    own scenario with default settings, so that is now verified rather than asserted:
    every detector fires on its scenario *in isolation*, none of them fires on
    genuinely varied work, the default set stays clean on it, and `repeat` fires at
    exactly the third identical call.

  The Chinese README is a full rewrite rather than a patch, not a translation of the
  old one, and keeps the same structure so the two can be diffed against each other.
- **`examples/checkpointing.py` now prints both snapshot sizes.** It reported only
  the pretty-printed length (1358 bytes), which reads as the cost of a checkpoint
  when it is really the cost of reading one — the compact form is 882 bytes for the
  same run. It prints both and says which one to store.

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

[Unreleased]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yaoyuxiang-gnn/agent-guard/releases/tag/v0.2.0
[0.1.0]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.1.0...v0.2.0
