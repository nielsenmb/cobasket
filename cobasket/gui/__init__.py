"""PyQt dashboard components for Cobasket.

The PyQt dependency is imported lazily so report-formatting helpers remain
available in non-GUI environments.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CobasketDashboard", "main"]


def __getattr__(name: str) -> Any:
    """Load PyQt dashboard objects only when explicitly requested."""
    if name in __all__:
        from .dashboard import CobasketDashboard, main

        return {"CobasketDashboard": CobasketDashboard, "main": main}[name]
    raise AttributeError(name)
