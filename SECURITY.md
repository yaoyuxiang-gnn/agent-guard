# Security Policy

## What counts as a vulnerability here

`agent-guard` is a safety library, so "security" means something slightly unusual:
**its threat model is the agent itself.** The failure that matters is an agent
spending money, or looping forever, without the guard noticing.

These count as vulnerabilities:

- **A guard bypass.** Any documented usage pattern where an LLM call is made but
  is not accounted for, so the budget never moves. This is the most serious class
  of bug in this project.
- **A silent zero.** A path where an unknown model, a missing usage object, or a
  malformed response is billed as `$0.00` without a warning reaching the caller.
- **A limit that does not fire.** `max_usd`, `max_tokens`, `max_steps` or
  `max_seconds` failing to trip in a normal, documented integration.
- **A thread-safety failure.** A call recorded concurrently and then lost, or a
  limit evaluated against a stale total.
- **Arbitrary code execution or injection** through a signature, a detector name,
  a report field, or the CLI parsing an untrusted report file.

These do **not** count:

- An incorrect price in the bundled table. Prices move; the table is documented as
  indicative, unknown models are reported rather than guessed, and corrections are
  welcome as ordinary pull requests.
- A loop detector false positive, or a paraphrase loop that slips past
  `SimilarityDetector`. Detection is heuristic by design and every threshold is
  configurable.
- Pre-flight input-token estimates being imprecise. This is documented behaviour.

## Configuration trust

Prices can be configured by a JSON file (see the README). That makes the file part
of the threat model, because a file that can lower a price or `disable` a model can
weaken the budget:

- A config in **your own** config directory, or one named by `$AGENTGUARD_CONFIG`,
  is yours and is read as-is.
- A config found in the **project tree** (`agentguard.json` in the working
  directory or a parent) travels with the repository, so it is written by whoever
  wrote that repository. It is **not** read unless you set
  `$AGENTGUARD_TRUST_PROJECT_CONFIG=1`, and skipping it is reported once per
  process. Otherwise "clone this repository and run your agent in it" would be a
  documented way to reprice the guard's models or disable its caps without
  touching your code. That would be a guard bypass, and guard bypasses are the
  most serious class of bug in this project.

A malformed config file fails at `Guard` construction and names the file; it can
stop an agent from starting, but it cannot silently make a limit weaker.

## Supported versions

The latest released minor version receives security fixes. This project is 0.x, so
please upgrade to the newest release before reporting.

| Version | Supported |
|---|---|
| 0.3.x | Yes |
| 0.2.x | No — please upgrade |
| < 0.2 | No |

## Reporting

Use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/yaoyuxiang-gnn/agent-guard/security/advisories/new)**

Please do not open a public issue for anything in the first list above.

Include:

- the smallest reproducer you can manage (the offline fake clients in `examples/`
  are usually enough — no API key needed),
- the `guard.report()` output for the run,
- and which guarantee you believe was broken.

## What to expect

- **Acknowledgement** within 3 working days.
- **An assessment** — accepted, needs more information, or out of scope with a
  reason — within 10 working days.
- **Credit** in the release notes and the advisory, unless you would rather stay
  anonymous.

Fixes for guard bypasses are released as patch versions as soon as they are ready,
without waiting for a scheduled release.

## Scope note

`agent-guard` makes no network requests, stores nothing, and runs no background
threads. A report of "agent-guard is sending data somewhere" would itself be the
most severe bug in this project's history — and is a good reason to report
privately first.
