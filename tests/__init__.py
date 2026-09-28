"""Test package for agent-guard.

Adding ``src/`` to ``sys.path`` here means the suite runs against a checkout with
no install step at all::

    python -m unittest discover -s tests -t .
    pytest

pytest gets the same effect from ``pythonpath = ["src"]`` in ``pyproject.toml``;
this shim exists so the stdlib runner works too.

It also pins ``$AGENTGUARD_CONFIG`` to ``none``, so a developer's personal
``agentguard.json`` (or ``~/.config/agentguard/pricing.json``) can never change
what the suite asserts. Tests that are *about* configuration set the variable
themselves or pass ``config_path=`` / ``env=`` explicitly. Doing this here rather
than in a ``conftest.py`` matters: ``conftest.py`` is only loaded by pytest, while
``make test`` runs ``python -m unittest discover``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("AGENTGUARD_CONFIG", "none")
