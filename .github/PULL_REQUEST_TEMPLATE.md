<!--
Thanks for the pull request. A few things that make review fast:

- Run `python -m unittest discover -s tests -t .` before pushing. It needs no
  install; if it does, the zero-dependency promise has been broken.
- Add a test for behaviour changes. For a new loop detector, add a case to
  `DefaultRegimeTests` in `tests/test_loop.py` proving it is reachable with the
  defaults and does not preempt the detectors that already exist.
- Update `CHANGELOG.md` under `## [Unreleased]`.
- Keep dependencies at zero.
-->

## What this changes

<!-- One or two sentences. -->

## Why

<!-- The failure being prevented or the limitation being removed. -->

## How it was verified

<!--
Which tests cover this, and what you ran locally. If you changed a detector
threshold, include the near-miss case you checked it against.
-->

## Checklist

- [ ] `python -m unittest discover -s tests -t .` passes
- [ ] `python examples/loop_detection.py` still reports `clean, as it should be` for healthy work
- [ ] Added or updated tests
- [ ] Updated `CHANGELOG.md`
- [ ] No new runtime dependencies
- [ ] No new I/O, network calls, or background threads
