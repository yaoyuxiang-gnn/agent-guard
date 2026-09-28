# Contributing to agent-guard

Thanks for considering it. This document is short on ceremony and long on the two
things people actually contribute: **detectors** and **prices**.

## Getting set up

No install is required to run the test suite or the examples.

```bash
git clone https://github.com/yaoyuxiang-gnn/agent-guard
cd agent-guard

python -m unittest discover -s tests -t .   # 265 tests, no network, no fixtures
python examples/basic.py
```

If you want the optional tooling:

```bash
python -m pip install -e ".[dev]"   # pytest, pytest-cov, ruff, mypy
pytest --cov=agentguard
ruff check .
mypy
```

## Non-negotiable constraints

Four rules define this library. A pull request that breaks one will be asked to
change, however good the feature is.

1. **Zero runtime dependencies.** Standard library only. `dependencies = []` in
   `pyproject.toml` stays empty. This is why agent-guard can be dropped into a
   stack that vendors its dependencies or pins an old Python.
2. **No I/O.** No network calls, no filesystem reads at import time, no background
   threads, no telemetry. The one exception is `Guard.save()`, which writes where
   the caller tells it to.
3. **Never guess a number.** If a price or a token count is unknown, say so. Do not
   substitute a plausible default. A guard that quietly assumes `$0.00` is more
   dangerous than no guard.
4. **Fail at construction.** Misconfiguration raises `GuardConfigError` while the
   `Guard` is being built, never in the middle of an agent run.

## One name rule, and why it matters

An unrelated project already publishes `agent-guard` on PyPI, and its import name is
also `agent_guard`. That is why this project installs as **`agentguard`** — no hyphen —
for the distribution name, the import name and the console script.

So there are exactly two spellings, and the split is not cosmetic:

| Where | Spelling |
|---|---|
| The repository, and prose in README / CHANGELOG / issue templates | `agent-guard` |
| Anything a program prints, and anything a user types | `agentguard` |

That means error prefixes, the report header, `--version`, warning messages and every
code sample are `agentguard`. If a user runs `pip install agentguard` and the tool then
introduces itself as `agent-guard`, they will reasonably wonder whether they installed
the wrong package — which, on this particular name, they might have.

## Adding a loop detector

Subclass `Detector` in `src/agentguard/loop.py`:

```python
class MyDetector(Detector):
    name = "my-detector"          # becomes LoopVerdict.kind

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        ...
```

Three things are required of a new detector:

- **It must be reachable with the default configuration.** A detector whose
  threshold makes it fire later than an existing detector is dead code that still
  looks maintained. This has already happened once in this repository:
  `CycleDetector` originally defaulted to `repeats=3`, which needs six
  observations, while `RepeatDetector` fires on the fifth — so every cycle was
  reported as a repeat. `tests/test_loop.py::DefaultRegimeTests` now guards
  against that class of bug.
- **It must not fire on healthy work.** Add a case to `DefaultRegimeTests`
  showing genuine, varied agent activity sailing through. False positives are
  worse than false negatives here: a detector that kills productive runs gets
  switched off, and then it catches nothing.
- **Its verdict must be explainable.** `LoopVerdict.detail` is a sentence a human
  reads at 2am. "loop detected" is not acceptable; "the same call appeared 3 times
  in the last 3 steps: search(q=...)" is.

Then add the detector to `default_detectors()` if it should be on by default, and
document it in the README table.

## Updating the price table

Prices move and this table is a best-effort snapshot. A pull request that only
touches prices is very welcome and easy to review.

In `src/agentguard/pricing.py`:

1. Edit the entry in `DEFAULT_PRICING`.
2. Bump `PRICING_AS_OF` if you have re-checked the table broadly.
3. Add the canonical name to `_DOTTED_PREFIXES` only if the provider uses dotted
   namespacing (Bedrock / Vertex style).
4. Run `python -m unittest tests.test_pricing` and `python -m agentguard.cli pricing <model>`.

Two rules for this table:

- **List prices only**, in USD per 1,000,000 tokens.
- **Do not add a guess.** If you are unsure of a number, leave the model out.
  Unknown models are reported to the user, which is strictly better than a wrong
  number that silently moves a budget.

## Changing a detector threshold

Thresholds are user-facing behaviour, so please include in the pull request:

- the near-miss case you checked it against (the loop that must still be caught),
- the healthy case you checked it against (the run that must still pass),
- and a sentence on why the new value is better than the old one.

## Adding a provider adapter

Adapters are pure duck-typing and must not import a provider SDK. See
`src/agentguard/adapters/__init__.py`. If a client's usage object has a shape the
extractor does not recognise, extend `extract_usage()` in `tracker.py` rather than
special-casing the adapter — every provider benefits.

## Tests

- Every public behaviour change needs a test.
- **Every docstring example runs as a test.** If you add a `>>>` block, it must
  pass; `tests/test_doctests.py` enforces this and fails if the total number of
  collected examples drops below a floor.
- Tests must not touch the network. Use the fake clients in `tests/` as a model.
- Time is injected, never slept on: pass `clock=` a fake to `Guard`.
- The suite runs under both `unittest` and `pytest`, on Python 3.10–3.13 and on
  Linux, macOS and Windows. Avoid platform-specific assumptions in tests.

## Commit and pull request conventions

Conventional Commits are appreciated but not enforced:

```
feat(loop): add a detector for alternating tool thrash
fix(pricing): correct the cached input rate for gemini-2.5-flash
docs(readme): clarify that pre-flight estimates input tokens
```

In the pull request body, say **what failure this prevents**. That is the question
reviewers actually have.

## Reporting bugs and security issues

- Bugs: use the issue template. Include `print(guard.report())` if a guard was
  involved — it answers most of the questions we would otherwise have to ask.
- Security: see [SECURITY.md](SECURITY.md). A guard bypass counts.
- Model prices: a pull request is better than an issue.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Assume good
faith, and review the code rather than the person.

## License

By contributing you agree that your work is released under the [MIT License](LICENSE).
