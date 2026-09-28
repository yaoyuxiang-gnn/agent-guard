"""Human-readable and machine-readable run reports.

The text report is the artefact people screenshot, paste into an issue, and read
at 2am when an agent burned $40 overnight. It is therefore designed to answer
three questions in the first two lines: *how much did this cost*, *how close was
I to the cap*, and *what would have stopped it*.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._util import format_duration, format_percent, format_tokens, format_usd
from .loop import LoopVerdict
from .pricing import PRICING_AS_OF
from .tracker import ModelSummary, Usage

__all__ = ["LimitStatus", "Report", "supports_unicode"]


_UNICODE_PROBE = "█░│─"
_unicode_cache: bool | None = None


def supports_unicode() -> bool:
    """Whether the current stdout can encode box-drawing characters.

    Windows consoles redirected to a file default to a legacy code page, where
    printing ``█`` raises :class:`UnicodeEncodeError`. Rather than crash a report,
    agent-guard falls back to ASCII. Force it either way with ``AGENT_GUARD_ASCII``.
    """
    global _unicode_cache
    if _unicode_cache is None:
        override = os.environ.get("AGENT_GUARD_ASCII", "").strip().lower()
        if override in ("1", "true", "yes"):
            _unicode_cache = False
        elif override in ("0", "false", "no"):
            _unicode_cache = True
        else:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            try:
                _UNICODE_PROBE.encode(encoding)
            except (UnicodeEncodeError, LookupError, TypeError):
                _unicode_cache = False
            else:
                _unicode_cache = True
    return _unicode_cache


@dataclass(frozen=True, slots=True)
class LimitStatus:
    """One configured limit and how close the run came to it.

    >>> status = LimitStatus(name="budget", used=0.5, limit=1.0, unit="usd")
    >>> status.fraction
    0.5
    >>> status.exceeded
    False
    """

    name: str
    used: float
    limit: float
    unit: str

    @property
    def fraction(self) -> float | None:
        """Used / limit, or ``None`` when the limit is zero and undefined."""
        if self.limit <= 0:
            return None
        return self.used / self.limit

    @property
    def exceeded(self) -> bool:
        return self.used > self.limit

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    def _format(self, value: float) -> str:
        if self.unit == "usd":
            return format_usd(value)
        if self.unit == "tokens":
            return format_tokens(int(value))
        if self.unit == "steps":
            return str(int(value))
        return format_duration(value)

    def render(self, *, bar_width: int = 16, ascii_only: bool = False) -> str:
        """Render as ``NAME  used / limit  pct  [bar]``."""
        filled, empty = ("#", ".") if ascii_only else ("█", "░")
        fraction = self.fraction
        clamped = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
        cells = int(round(clamped * bar_width))
        bar = filled * cells + empty * (bar_width - cells)
        pair = f"{self._format(self.used)} / {self._format(self.limit)}"
        marker = "!" if self.exceeded else " "
        return (
            f"{marker} {self.name:<8} {pair:<26} "
            f"{format_percent(fraction):>7}  [{bar}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "used": self.used,
            "limit": self.limit,
            "unit": self.unit,
            "fraction": self.fraction,
            "remaining": self.remaining,
            "exceeded": self.exceeded,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """An immutable snapshot of a guard's run.

    Produced by :meth:`agent_guard.Guard.report`; safe to store, log or serialise
    after the guard itself has been garbage collected.
    """

    name: str | None
    elapsed_s: float
    steps: int
    calls: int
    usage: Usage
    cost_usd: float
    limits: tuple[LimitStatus, ...] = ()
    by_model: tuple[ModelSummary, ...] = ()
    unpriced_models: tuple[str, ...] = ()
    unpriced_calls: int = 0
    trip: LoopVerdict | None = None
    tripped_reason: str | None = None
    pricing_as_of: str = PRICING_AS_OF

    # -- text ----------------------------------------------------------------

    def render(self, *, width: int = 64, ascii_only: bool | None = None) -> str:
        """Render the report as plain text.

        Deterministic: it never reads the clock or the environment beyond the
        one-time Unicode probe, so it is safe to assert on in tests.
        """
        if ascii_only is None:
            ascii_only = not supports_unicode()
        rule = "-" * width if ascii_only else "─" * width

        lines: list[str] = []
        title = "agent-guard" + (f"  {self.name}" if self.name else "")
        lines.append(title)
        lines.append(("=" if ascii_only else "═") * width)

        lines.append(
            f"  {'wall time':<12}{format_duration(self.elapsed_s):<16}"
            f"{'steps':<10}{self.steps}"
        )
        lines.append(
            f"  {'llm calls':<12}{self.calls:<16}"
            f"{'tokens':<10}{format_tokens(self.usage.total_tokens)}"
            f"  (in {format_tokens(self.usage.input_tokens)}"
            f" / out {format_tokens(self.usage.output_tokens)})"
        )
        if self.usage.cached_input_tokens:
            lines.append(
                f"  {'':<12}{'':<16}{'cached':<10}"
                f"{format_tokens(self.usage.cached_input_tokens)} input tokens"
            )

        if self.limits:
            lines.append("")
            lines.append("  limits")
            for status in self.limits:
                lines.append("  " + status.render(ascii_only=ascii_only))

        if self.by_model:
            lines.append("")
            lines.append("  by model")
            name_width = min(22, max(len(s.model) for s in self.by_model))
            for summary in self.by_model:
                calls = f"{summary.calls} call" + ("s" if summary.calls != 1 else "")
                cost = format_usd(summary.cost_usd)
                if summary.unpriced_calls:
                    cost = "unpriced"
                lines.append(
                    f"    {summary.model:<{name_width}}  {calls:>8}  {cost:>10}  "
                    f"{format_tokens(summary.input_tokens):>10} in"
                    f" / {format_tokens(summary.output_tokens)} out"
                )

        if self.unpriced_calls:
            lines.append("")
            lines.append(
                f"  ! {self.unpriced_calls} call(s) had no known price and are "
                f"excluded from the budget:"
            )
            for model in self.unpriced_models:
                lines.append(f"      {model}")
            lines.append(
                "    Pass Guard(pricing={...}) to include them."
            )

        if self.trip is not None:
            lines.append("")
            lines.append(f"  tripped: loop [{self.trip.kind}] {self.trip.detail}")
        elif self.tripped_reason:
            lines.append("")
            lines.append(f"  tripped: {self.tripped_reason}")

        lines.append("")
        lines.append(rule)
        lines.append(f"  prices as of {self.pricing_as_of} (indicative only)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    # -- machine -------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, suitable for CI artefacts and dashboards."""
        return {
            "name": self.name,
            "elapsed_s": round(self.elapsed_s, 6),
            "steps": self.steps,
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 8),
            "usage": self.usage.as_dict(),
            "limits": [s.as_dict() for s in self.limits],
            "by_model": [s.as_dict() for s in self.by_model],
            "unpriced_calls": self.unpriced_calls,
            "unpriced_models": list(self.unpriced_models),
            "trip": self.trip.as_dict() if self.trip else None,
            "tripped_reason": self.tripped_reason,
            "pricing_as_of": self.pricing_as_of,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Report:
        """Rebuild a report from :meth:`as_dict` output.

        This is what lets the ``agent-guard report`` CLI render a JSON file that
        was written by a different process — typically one that had the guard but
        not a terminal.

        >>> report = Report(name="demo", elapsed_s=1.0, steps=2, calls=1,
        ...                 usage=Usage(input_tokens=10, output_tokens=5), cost_usd=0.01)
        >>> round(Report.from_dict(report.as_dict()).cost_usd, 4)
        0.01
        """
        usage_data = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens", 0)),
            output_tokens=int(usage_data.get("output_tokens", 0)),
            cached_input_tokens=int(usage_data.get("cached_input_tokens", 0)),
            reasoning_tokens=int(usage_data.get("reasoning_tokens", 0)),
        )

        limits = tuple(
            LimitStatus(
                name=str(item["name"]),
                used=float(item["used"]),
                limit=float(item["limit"]),
                unit=str(item.get("unit", "")),
            )
            for item in data.get("limits") or ()
        )

        by_model = tuple(
            ModelSummary(
                model=str(item["model"]),
                calls=int(item["calls"]),
                input_tokens=int(item["input_tokens"]),
                output_tokens=int(item["output_tokens"]),
                cost_usd=float(item["cost_usd"]),
                unpriced_calls=int(item.get("unpriced_calls", 0)),
            )
            for item in data.get("by_model") or ()
        )

        trip_data = data.get("trip")
        trip = (
            LoopVerdict(
                kind=str(trip_data.get("kind", "loop")),
                detail=str(trip_data.get("detail", "")),
                signature=trip_data.get("signature"),
                count=int(trip_data.get("count", 0)),
                step=trip_data.get("step"),
            )
            if trip_data
            else None
        )

        return cls(
            name=data.get("name"),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            steps=int(data.get("steps", 0)),
            calls=int(data.get("calls", 0)),
            usage=usage,
            cost_usd=float(data.get("cost_usd", 0.0)),
            limits=limits,
            by_model=by_model,
            unpriced_models=tuple(data.get("unpriced_models") or ()),
            unpriced_calls=int(data.get("unpriced_calls", 0)),
            trip=trip,
            tripped_reason=data.get("tripped_reason"),
            pricing_as_of=str(data.get("pricing_as_of", PRICING_AS_OF)),
        )


def build_limits(
    *,
    cost_usd: float,
    max_usd: float | None,
    total_tokens: int,
    max_tokens: int | None,
    steps: int,
    max_steps: int | None,
    elapsed_s: float,
    max_seconds: float | None,
) -> tuple[LimitStatus, ...]:
    """Assemble the limit rows for a report, skipping unconfigured limits."""
    statuses: list[LimitStatus] = []
    if max_usd is not None:
        statuses.append(LimitStatus("budget", cost_usd, max_usd, "usd"))
    if max_tokens is not None:
        statuses.append(LimitStatus("tokens", float(total_tokens), float(max_tokens), "tokens"))
    if max_steps is not None:
        statuses.append(LimitStatus("steps", float(steps), float(max_steps), "steps"))
    if max_seconds is not None:
        statuses.append(LimitStatus("time", elapsed_s, max_seconds, "seconds"))
    return tuple(statuses)
