"""Small formatting and hashing helpers shared across agent-guard.

Everything here is standard library only and free of side effects, so it is safe
to import from any layer of the package.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = [
    "format_usd",
    "format_tokens",
    "format_duration",
    "format_percent",
    "stable_json",
    "short_hash",
]


def format_usd(value: float | None) -> str:
    """Render a dollar amount with just enough precision to be readable.

    Small amounts keep more decimals, because "$0.0000" tells a user nothing
    when they are debugging why a cheap agent still blew its cap.

    >>> format_usd(1.5)
    '$1.5'
    >>> format_usd(0.00042)
    '$0.00042'
    >>> format_usd(None)
    'n/a'
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value != 0 and abs(value) < 0.01:
        return f"${value:.5f}".rstrip("0").rstrip(".")
    return f"${value:.4f}".rstrip("0").rstrip(".")


def format_tokens(count: int | None) -> str:
    """Render a token count with thousands separators.

    >>> format_tokens(184203)
    '184,203'
    >>> format_tokens(None)
    'n/a'
    """
    if count is None:
        return "n/a"
    return f"{count:,}"


def format_duration(seconds: float | None) -> str:
    """Render a duration compactly: ``840ms``, ``12.4s``, ``3m 05s``.

    >>> format_duration(0.84)
    '840ms'
    >>> format_duration(12.42)
    '12.4s'
    >>> format_duration(185)
    '3m 05s'
    """
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {rest:02d}s"


def format_percent(fraction: float | None, *, width: int = 1) -> str:
    """Render a 0..1 fraction as a percentage.

    >>> format_percent(0.4821)
    '48.2%'
    >>> format_percent(None)
    'n/a'
    """
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.{width}f}%"


def stable_json(value: Any) -> str:
    """Serialise a value to JSON with a deterministic key order.

    Used to fingerprint tool arguments, so that two calls that differ only in
    dictionary insertion order hash to the *same* loop-detection signature.
    Falls back to :func:`repr` for values JSON cannot represent (sets, custom
    objects, framework message types).
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_fallback)
    except (TypeError, ValueError):
        return repr(value)


def _fallback(obj: Any) -> str:
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return stable_json(method())
            except Exception:  # noqa: BLE001 - best effort only
                continue
    return repr(obj)


def short_hash(text: str, *, length: int = 12) -> str:
    """Return a short, stable hex digest of ``text``.

    >>> short_hash("search:python", length=8) == short_hash("search:python", length=8)
    True
    >>> len(short_hash("search:python", length=8))
    8
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()[:length]
