# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-28

First release.

Installs as **`agentguard`**, with no hyphen. An unrelated project already owns
`agent-guard` on PyPI, and its import name is also `agent_guard`, so both had to
move. See [CONTRIBUTING.md](CONTRIBUTING.md) for the spelling rule used across the
codebase: `agent-guard` is the repository and the name in prose, `agentguard` is
everything a program prints and everything a user types.

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
- 264 tests, including every docstring example.

[Unreleased]: https://github.com/yaoyuxiang-gnn/agent-guard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yaoyuxiang-gnn/agent-guard/releases/tag/v0.1.0
