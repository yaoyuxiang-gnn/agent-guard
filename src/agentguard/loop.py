"""Runaway-loop detection.

An agent that has stopped making progress rarely stops on its own. It calls the
same tool with the same arguments, or bounces between two tools forever, burning
tokens and dollars on every pass. A budget cap eventually catches that — but only
once the money is gone. Loop detection catches it *while* it is happening, and
tells you why.

Four stdlib-only detectors ship with agent-guard, and the set is pluggable:

================== ==========================================================
Detector           Catches
================== ==========================================================
``RepeatDetector`` The same call, identical arguments, N times in a window.
``CycleDetector``  A short repeating pattern: A, B, A, B, ...
``SimilarityDetector`` Near-duplicates: ``search python`` / ``search python ``
``NoProgressDetector`` A progress marker that never changes.
================== ==========================================================

Detectors are intentionally *explainable*: each returns a human-readable
:class:`LoopVerdict`, because "why did you kill my agent?" is the first question
anyone asks::

    from agentguard import Guard, LoopDetected

    guard = Guard(loop_detection=True)
    try:
        with guard:
            for _ in range(10):
                guard.observe(Guard.call_signature("search", {"q": "weather"}))
    except LoopDetected as exc:
        print(exc.detail)   # 'the same call appeared 3 times in the last 3 steps: ...'

The defaults are tuned so that each detector owns a distinct, reachable regime:
a repeat fires on the third identical call, a cycle on the second full pass of an
alternating pattern, similarity on the fourth near-identical call, and stagnation
after six unchanged progress markers. Change any of them per guard.
"""

from __future__ import annotations

import difflib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._util import short_hash, stable_json
from .exceptions import GuardConfigError

__all__ = [
    "LoopVerdict",
    "Detector",
    "RepeatDetector",
    "CycleDetector",
    "SimilarityDetector",
    "NoProgressDetector",
    "LoopMonitor",
    "call_signature",
    "default_detectors",
    "default_progress_detectors",
]

# Signatures can embed whole tool payloads; comparisons and logs only ever need
# a fraction of one, so keep the fingerprint bounded to avoid quadratic
# similarity work. The bound is deliberately well below the point where
# ``difflib.SequenceMatcher.ratio()`` starts to dilute: on a 2,000-character
# string a genuine difference of twenty characters still scores above 0.99, so a
# "95% similar" threshold applied to unbounded text stops meaning anything.
_MAX_SIGNATURE_LEN = 256

#: Length of the content digest folded into a truncated signature.
_SIGNATURE_DIGEST_LEN = 12

#: How much of a payload a truncated signature keeps before the digest.
_SIGNATURE_HEAD_LEN = 64

#: How much of the *end* of a payload a truncated signature keeps after it.
#: Arguments a tool is keyed on — an ``id``, a ``path``, a ``cursor`` — sort late
#: in the canonical JSON, and the keys a call site controls are ordered first, so
#: a head-only truncation drops precisely the part that distinguishes two calls.
_SIGNATURE_TAIL_LEN = 64

#: How much of each signature :class:`SimilarityDetector` compares.
#:
#: Short enough to discriminate, long enough to see the digest that
#: :func:`call_signature` folds into a truncated fingerprint. A percentage
#: threshold loses its meaning as the text it is applied to grows — on a
#: two-thousand-character string a genuine difference of twenty characters still
#: scores 0.99 — so the comparison is bounded rather than the ratio.
_DEFAULT_COMPARE_CHARS = 128


def call_signature(name: str, args: Any = None, *, max_len: int = _MAX_SIGNATURE_LEN) -> str:
    """Build a stable fingerprint for a tool call.

    ``args`` is canonicalised with sorted JSON keys, so two calls that differ only
    in dictionary insertion order produce the *same* signature — which is exactly
    the false negative that makes naive loop detectors useless.

    A signature longer than ``max_len`` is truncated to bound comparison cost, and
    the truncation is **collision-free**: it keeps a head, keeps a tail, and folds
    the digest of the whole payload in between — digest first, so that it lands
    inside the window :class:`SimilarityDetector` compares. Cutting the tail off
    and stopping there is a false-positive machine: an agent indexing documents
    that share a boilerplate body would collapse every call onto one fingerprint
    and be reported as a repeat, stopping the healthy work this library exists to
    protect. The digest is what makes "different call, different fingerprint" true
    for payloads of any length, however similar they look.

    >>> call_signature("search", {"q": "a", "n": 1}) == call_signature("search", {"n": 1, "q": "a"})
    True
    >>> call_signature("search", None)
    'search()'
    >>> len(call_signature("big", {"blob": "x" * 5000})) <= 256
    True
    >>> # Two long payloads that differ only at the end stay distinguishable.
    >>> call_signature("b", {"x": "y" * 5000}) != call_signature("b", {"x": "y" * 4999 + "z"})
    True
    """
    rendered = "" if args is None else stable_json(args)
    signature = f"{name}({rendered})"
    if max_len and len(signature) > max_len:
        digest = short_hash(signature, length=_SIGNATURE_DIGEST_LEN)
        # Room for the fixed overhead, then whatever is left is split head/tail.
        head = min(_SIGNATURE_HEAD_LEN, max(1, max_len // 3))
        tail = min(_SIGNATURE_TAIL_LEN, max(1, max_len // 3))
        marker = f"...<{len(signature) - max_len} more chars, digest {digest}>..."
        budget = max(1, max_len - len(marker))
        head, tail = min(head, budget), min(tail, max(0, budget - head))
        signature = f"{signature[:head]}{marker}{signature[len(signature) - tail :]}"
    return signature


def _preview(signature: str, limit: int = 72) -> str:
    """Shorten a signature for inclusion in an error message."""
    if len(signature) <= limit:
        return signature
    return signature[:limit] + "..."


def _squeeze(signature: str) -> str:
    """Drop whitespace, so two spellings of one call compare equal.

    ``"search(\\"python asyncio\\")"`` and ``"search(\\"python asyncio \\")"`` are the
    same query typed twice; treating them as equal is the whole point of the
    similarity detector. Kept whitespace-only so it cannot hide a real difference:
    any change to a payload character still shows up.
    """
    return "".join(signature.split())


@dataclass(frozen=True, slots=True)
class LoopVerdict:
    """A detector's conclusion that the agent is stuck.

    >>> verdict = LoopVerdict(kind="repeat", detail="same call 3x", count=3)
    >>> verdict.kind
    'repeat'
    """

    #: Detector name, e.g. ``"repeat"`` or ``"cycle"``.
    kind: str
    #: One-line, human-readable explanation. Surfaced verbatim in the exception.
    detail: str
    #: The fingerprint that triggered the verdict, when there is a single one.
    signature: str | None = None
    #: How many times the offending pattern was observed.
    count: int = 0
    #: Agent step at which the verdict was reached.
    step: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "signature": self.signature,
            "count": self.count,
            "step": self.step,
        }


class Detector:
    """Base class for loop detectors.

    A detector is a tiny state machine fed one signature per observation. Return a
    :class:`LoopVerdict` to stop the run, or ``None`` to keep going.

    Subclassing is the intended extension point — a detector only needs
    :meth:`observe`, and optionally :meth:`reset` if it keeps state::

        class ScreamingDetector(Detector):
            name = "screaming"

            def observe(self, signature, step):
                if "ERROR" in signature:
                    return LoopVerdict(self.name, "agent is screaming", signature, 1, step)
                return None
    """

    #: Stable identifier, used as :attr:`LoopVerdict.kind`.
    name: str = "detector"

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        """Consume one observation. Return a verdict to trip the guard."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear accumulated state. Called when a guard is reused."""
        return None

    def get_state(self) -> dict[str, Any]:
        """Return this detector's sliding windows as JSON-friendly data.

        Used by :meth:`agentguard.Guard.snapshot`, so a checkpointed run resumes
        with the observations it had already seen. A loop that spans a checkpoint
        is still a loop, and forgetting the window would let it start over.

        The default raises :class:`NotImplementedError` rather than returning
        ``{}``, because a custom detector that silently reported "no state" would
        look checkpointed while losing the very history it exists to keep. The
        guard skips detectors that do not implement this, and warns once.

        >>> RepeatDetector(max_repeats=2, window=4).get_state()
        {'recent': []}
        """
        raise NotImplementedError

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore state previously produced by :meth:`get_state`.

        The inverse of :meth:`get_state`, and optional in the same way: a
        detector that cannot be restored raises rather than pretending.

        >>> detector = RepeatDetector(max_repeats=2, window=4)
        >>> detector.set_state({"recent": ["a", "a"]})
        >>> detector.observe("a", 3).kind
        'repeat'
        """
        raise NotImplementedError


class RepeatDetector(Detector):
    """Trip when the identical call repeats too often inside a sliding window.

    The cheapest and highest-precision detector: byte-identical tool calls with
    byte-identical arguments are essentially never legitimate after the first few.

    >>> detector = RepeatDetector(max_repeats=2, window=6)
    >>> detector.observe("search(q=1)", 1) is None
    True
    >>> detector.observe("search(q=1)", 2).kind
    'repeat'
    """

    name = "repeat"

    def __init__(self, *, max_repeats: int = 3, window: int = 12) -> None:
        if max_repeats < 2:
            raise GuardConfigError(f"max_repeats must be >= 2, got {max_repeats}")
        if window < max_repeats:
            raise GuardConfigError(f"window ({window}) must be >= max_repeats ({max_repeats})")
        self.max_repeats = max_repeats
        self.window = window
        self._recent: deque[str] = deque(maxlen=window)

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        self._recent.append(signature)
        count = self._recent.count(signature)
        if count >= self.max_repeats:
            return LoopVerdict(
                kind=self.name,
                detail=(
                    f"the same call appeared {count} times in the last "
                    f"{len(self._recent)} steps: {_preview(signature)}"
                ),
                signature=signature,
                count=count,
                step=step,
            )
        return None

    def reset(self) -> None:
        self._recent.clear()

    def get_state(self) -> dict[str, Any]:
        """The sliding window of signatures, oldest first.

        >>> detector = RepeatDetector(max_repeats=2, window=4)
        >>> _ = detector.observe("a", 1)
        >>> detector.get_state()
        {'recent': ['a']}
        """
        return {"recent": list(self._recent)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        recent = state.get("recent") or []
        self._recent.clear()
        # Re-append through the deque so maxlen still governs, and a hand-written
        # state file cannot grow the window past what this detector was built for.
        self._recent.extend(str(item) for item in recent)


class CycleDetector(Detector):
    """Trip on a short repeating pattern such as ``A, B, A, B``.

    Catches the two-tool ping-pong that a pure repeat detector misses: read a
    file, write a file, read the same file, write the same file, forever.

    ``repeats`` defaults to **2**, not 3. With 3 the detector would need six
    observations to fire, while :class:`RepeatDetector` fires on the fifth — so
    the cycle would always be reported as a repeat and this detector would never
    be reachable in practice. Two full repetitions of an argument-identical cycle
    is already a strong signal: nothing changed between the first and second pass.

    >>> detector = CycleDetector(min_cycle=2, max_cycle=3, repeats=2)
    >>> for i, sig in enumerate(["read", "write"] * 2, start=1):
    ...     verdict = detector.observe(sig, i)
    >>> verdict.kind
    'cycle'
    """

    name = "cycle"

    def __init__(
        self,
        *,
        min_cycle: int = 2,
        max_cycle: int = 4,
        repeats: int = 2,
    ) -> None:
        if min_cycle < 2:
            raise GuardConfigError(
                f"min_cycle must be >= 2 (period-1 loops are RepeatDetector's job), got {min_cycle}"
            )
        if max_cycle < min_cycle:
            raise GuardConfigError(f"max_cycle ({max_cycle}) must be >= min_cycle ({min_cycle})")
        if repeats < 2:
            raise GuardConfigError(f"repeats must be >= 2, got {repeats}")
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.repeats = repeats
        self._recent: deque[str] = deque(maxlen=max_cycle * repeats + 2)

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        self._recent.append(signature)
        sequence = list(self._recent)

        for period in range(self.min_cycle, self.max_cycle + 1):
            needed = period * self.repeats
            if len(sequence) < needed:
                continue
            tail = sequence[-needed:]
            pattern = tail[:period]
            # All-identical patterns are repeats, not cycles; let RepeatDetector
            # own that signal so the two never report the same event twice.
            if len(set(pattern)) == 1:
                continue
            if all(tail[i] == pattern[i % period] for i in range(needed)):
                return LoopVerdict(
                    kind=self.name,
                    detail=(
                        f"a {period}-step pattern repeated {self.repeats} times: "
                        + " -> ".join(_preview(p, 32) for p in pattern)
                    ),
                    signature=signature,
                    count=self.repeats,
                    step=step,
                )
        return None

    def reset(self) -> None:
        self._recent.clear()

    def get_state(self) -> dict[str, Any]:
        """The sliding window of signatures, oldest first."""
        return {"recent": list(self._recent)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        recent = state.get("recent") or []
        self._recent.clear()
        self._recent.extend(str(item) for item in recent)


class SimilarityDetector(Detector):
    """Trip on *near*-duplicate calls that differ only trivially.

    Real agents rarely repeat themselves byte-for-byte. They search for
    ``"python asyncio"``, then ``"python asyncio "``, then ``"asyncio python"``.
    Similarity is measured with :class:`difflib.SequenceMatcher` over a bounded
    prefix, so this stays stdlib-only and cheap enough for a hot loop.

    The default threshold of ``0.95`` with ``max_similar=3`` is deliberately
    conservative: it needs four separate near-identical calls. Lower it if your
    agent paraphrases aggressively; raise it if legitimate queries in your domain
    share long prefixes (searching for successive library versions, for example).

    ``compare_chars`` is how much of each signature is compared. Raising it does
    not make the detector more sensitive, it makes it *less*: on a
    two-thousand-character string a real difference of twenty characters still
    scores 0.99, so a percentage threshold stops discriminating the longer the
    text it is applied to gets.

    >>> detector = SimilarityDetector(threshold=0.9, window=8, max_similar=1)
    >>> detector.observe("search(q=python)", 1) is None
    True
    >>> detector.observe("search(q=python )", 2).kind
    'similarity'
    """

    name = "similarity"

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        window: int = 8,
        max_similar: int = 3,
        compare_chars: int = _DEFAULT_COMPARE_CHARS,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise GuardConfigError(f"threshold must be in (0, 1], got {threshold}")
        if window < 1:
            raise GuardConfigError(f"window must be >= 1, got {window}")
        if max_similar < 1:
            raise GuardConfigError(f"max_similar must be >= 1, got {max_similar}")
        if compare_chars < 1:
            raise GuardConfigError(f"compare_chars must be >= 1, got {compare_chars}")
        self.threshold = threshold
        self.window = window
        self.max_similar = max_similar
        self.compare_chars = compare_chars
        self._recent: deque[str] = deque(maxlen=window)

    def _key(self, signature: str) -> str:
        """The bounded, digest-bearing form this detector actually compares.

        Two problems make comparing raw signatures unreliable, and both come down
        to the same thing: a percentage threshold over text whose length the caller
        controls stops measuring what it claims to.

        *Length dilution.* On a two-thousand-character call, changing one argument
        character leaves 99.9% of the text identical, so *every* pair of similar
        calls scores above a 95% bar. An agent indexing successive documents — same
        boilerplate body, a different ``id`` — is doing healthy work, and it was
        being reported as a loop on the fourth document.

        *Length dependence.* The same two calls can score above or below the bar
        depending only on how much payload surrounds the difference.

        Folding the digest of the whole signature in fixes both: any change to the
        payload at all moves the key, whether the signature was 30 characters or
        thirty thousand, so the ratio measures similarity of *calls* rather than
        similarity of whatever text happened to survive truncation. Whitespace-only
        differences still score 1.0, which keeps the paraphrase case — the one this
        detector exists for — working exactly as documented.
        """
        squeezed = _squeeze(signature)
        digest = short_hash(squeezed, length=_SIGNATURE_DIGEST_LEN)
        return f"{squeezed[: self.compare_chars]}|{digest}"

    def _ratio(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        key = self._key(signature)
        similar = [k for k in self._recent if self._ratio(k, key) >= self.threshold]
        self._recent.append(key)
        if len(similar) >= self.max_similar:
            return LoopVerdict(
                kind=self.name,
                detail=(
                    f"{len(similar) + 1} near-identical calls "
                    f"(>= {self.threshold:.0%} similar) in the last "
                    f"{len(self._recent)} steps: {_preview(signature)}"
                ),
                signature=signature,
                count=len(similar) + 1,
                step=step,
            )
        return None

    def reset(self) -> None:
        self._recent.clear()

    def get_state(self) -> dict[str, Any]:
        """The window of compared signatures, oldest first."""
        return {"recent": list(self._recent)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        recent = state.get("recent") or []
        self._recent.clear()
        self._recent.extend(str(item) for item in recent)


class NoProgressDetector(Detector):
    """Trip when an explicitly reported progress marker stops changing.

    Unlike the other detectors this one is fed by :meth:`agentguard.Guard.progress`
    rather than by tool calls, because only your agent knows what "progress" means
    — a row count, an HTTP cursor, a test-pass tally.

    >>> detector = NoProgressDetector(max_stagnant=3)
    >>> for i, value in enumerate(["a", "b", "b", "b"], start=1):
    ...     verdict = detector.observe(value, i)
    >>> verdict.kind
    'no-progress'
    """

    name = "no-progress"

    def __init__(self, *, max_stagnant: int = 6) -> None:
        if max_stagnant < 2:
            raise GuardConfigError(f"max_stagnant must be >= 2, got {max_stagnant}")
        self.max_stagnant = max_stagnant
        self._last: str | None = None
        self._same = 0

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        if signature == self._last:
            self._same += 1
        else:
            self._last = signature
            self._same = 1
        if self._same >= self.max_stagnant:
            return LoopVerdict(
                kind=self.name,
                detail=(
                    f"the progress marker did not change for {self._same} "
                    f"consecutive observations: {_preview(signature)}"
                ),
                signature=signature,
                count=self._same,
                step=step,
            )
        return None

    def reset(self) -> None:
        self._last = None
        self._same = 0

    def get_state(self) -> dict[str, Any]:
        """The last progress marker and how many times it has repeated.

        >>> detector = NoProgressDetector(max_stagnant=3)
        >>> _ = detector.observe("rows=0", 1)
        >>> detector.get_state()
        {'last': 'rows=0', 'same': 1}
        """
        return {"last": self._last, "same": self._same}

    def set_state(self, state: Mapping[str, Any]) -> None:
        last = state.get("last")
        self._last = None if last is None else str(last)
        same = state.get("same") or 0
        self._same = int(same)
        # A state file claiming stagnation it never observed must not trip on the
        # next observation alone; the count is clamped to what this detector can
        # legitimately have accumulated, mirroring ``max_stagnant`` at the top.
        if self._last is None:
            self._same = 0
        self._same = max(0, min(self._same, self.max_stagnant))


class LoopMonitor:
    """Runs a sequence of detectors and reports the first verdict.

    Every detector sees every observation — even after one of them trips — so a
    guard that chooses to log-and-continue keeps its detectors in sync.

    >>> monitor = LoopMonitor([RepeatDetector(max_repeats=2, window=4)])
    >>> monitor.observe("a", 1) is None
    True
    >>> monitor.observe("a", 2).kind
    'repeat'
    >>> monitor.observe("a", 3).count
    3
    """

    __slots__ = ("_detectors",)

    def __init__(self, detectors: Sequence[Detector] | None = None) -> None:
        self._detectors: list[Detector] = list(
            detectors if detectors is not None else default_detectors()
        )

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return tuple(self._detectors)

    def __bool__(self) -> bool:
        return bool(self._detectors)

    def __len__(self) -> int:
        return len(self._detectors)

    def observe(self, signature: str, step: int) -> LoopVerdict | None:
        """Feed one observation to every detector; return the first verdict."""
        first: LoopVerdict | None = None
        for detector in self._detectors:
            verdict = detector.observe(signature, step)
            if verdict is not None and first is None:
                first = verdict
        return first

    def reset(self) -> None:
        for detector in self._detectors:
            detector.reset()

    def get_states(self) -> list[dict[str, Any] | None]:
        """One entry per detector, in order; ``None`` where state is unsupported.

        A custom detector that does not implement :meth:`Detector.get_state` yields
        ``None`` rather than an empty dict, so a checkpoint records "this detector's
        history was *not* kept" instead of implying there was none to keep.
        """
        states: list[dict[str, Any] | None] = []
        for detector in self._detectors:
            try:
                states.append(detector.get_state())
            except NotImplementedError:
                states.append(None)
        return states

    def set_states(self, states: Sequence[Mapping[str, Any] | None]) -> None:
        """Restore per-detector state, by position.

        Extra entries are ignored and missing ones leave that detector as it is:
        the checkpoint comes from a detector list that may not match the one being
        restored, and refusing to restore a budget over a detector-list mismatch
        would be the wrong trade.
        """
        for detector, state in zip(self._detectors, states, strict=False):
            if state is None:
                continue
            try:
                detector.set_state(state)
            except NotImplementedError:
                continue


def default_detectors() -> list[Detector]:
    """Detector set used when ``Guard(loop_detection=True)``.

    Ordered cheapest-and-most-precise first, so the reported verdict is the most
    specific explanation available.
    """
    return [RepeatDetector(), CycleDetector(), SimilarityDetector()]


def default_progress_detectors() -> list[Detector]:
    """Detector set used for the :meth:`agentguard.Guard.progress` channel."""
    return [NoProgressDetector()]
