"""Single source of truth for the package version.

Kept in its own module so that ``pyproject.toml`` (via
``[tool.hatch.version]``) and ``agentguard.__version__`` can never drift apart.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
