# Contributing to agent-guard

Thanks for considering it. This document is short on ceremony and long on the two
things people actually contribute: **detectors** and **prices**.

## Getting set up

No install is required to run the test suite or the examples.

```bash
git clone https://github.com/yaoyuxiang-gnn/agent-guard
cd agent-guard

python -m unittest discover -s tests -t .   # no network, no fixtures
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

The distribution name, the import name and the console script are deliberately not all
the same string, and the split is not cosmetic:

| Where | Spelling |
|---|---|
| The repository, and prose in README / CHANGELOG / issue templates | `agent-guard` |
| The PyPI distribution name | `agent-budget-guard-py` |
| What you import, and what you type on the command line | `agentguard` |

`agent-guard` was already taken on PyPI. The obvious alternative was not available
either: PyPI rejects a new project whose name differs from an existing one only by
punctuation, and `agentguard` collapses to the same string as `agent-guard` under that
rule. Hence the long distribution name, which is the only place the long form appears.

Inside the code the rule is therefore simple: **error prefixes, the report header,
`--version`, warning messages and every code sample say `agentguard`.** Nothing a user
imports or types should ever mention `agent-budget-guard-py` — that string belongs in
`pyproject.toml`, the install line, and `pip install <name>[extra]` in the docs, and
nowhere else.

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
   (`pricing` with a model name is the lookup; `config set` is what writes an entry.)

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
- **Every file in `examples/` runs in CI**, so a new example means adding it to
  `make examples` and to the "Run every example" step in `.github/workflows/ci.yml`
  in the same change. An example nothing runs is documentation that has already
  drifted.
- Tests must not touch the network. Use the fake clients in `tests/` as a model.
- Time is injected, never slept on: pass `clock=` a fake to `Guard`.
- The suite runs under both `unittest` and `pytest`, on Python 3.10–3.13 and on
  Linux, macOS and Windows. Avoid platform-specific assumptions in tests. A test
  that passes here and fails on macOS is usually a path, not a behaviour: the
  temporary directory is reached through `/var`, a symlink to `/private/var`, so
  `os.getcwd()` spells it differently than `tempfile` does. Compare paths with
  `Path.resolve()` (or `assert_same_file` in `tests/test_cli.py`), never as strings.

## The READMEs, and where they are rendered

`README.md` is the package description on PyPI as well as the repository front page,
and PyPI does not resolve relative paths: a link to `CONTRIBUTING.md` or an image at
`docs/demo.svg` is a 404 there. **Every link and image in `README.md` is therefore an
absolute URL**, and the hero image points at `raw.githubusercontent.com`, which serves
SVG with the right content type. `README.zh-CN.md` is only ever rendered by GitHub and
keeps relative links, which survive a repository rename.

`docs/demo.svg` is a screenshot of real program output — the report printed by
`examples/basic.py` — so it goes stale when the report format changes. It is a
checked-in asset with no generator in this repository: the script that produced it
is maintainer tooling and is deliberately not shipped, because a library repository
should carry the library. If a change alters what a report prints, say so in the
pull request and the maintainer will refresh the image.

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
