"""Test package for agent-guard.

Adding ``src/`` to ``sys.path`` here means the suite runs against a checkout with
no install step at all::

    python -m unittest discover -s tests -t .
    pytest

pytest gets the same effect from ``pythonpath = ["src"]`` in ``pyproject.toml``;
this shim exists so the stdlib runner works too.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
