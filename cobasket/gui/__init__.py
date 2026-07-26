"""PyQt dashboard components for Cobasket.

The PyQt dependency is imported lazily so report-formatting helpers remain
available in non-GUI environments.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CobasketDashboard", "EditableCobasketDashboard", "main"]


def __getattr__(name: str) -> Any:
    """Load PyQt dashboard objects only when explicitly requested."""
    if name == "CobasketDashboard":
        from .dashboard import CobasketDashboard

        return CobasketDashboard
    if name in {"EditableCobasketDashboard", "main"}:
        from .editable_dashboard import EditableCobasketDashboard, main

        return {
            "EditableCobasketDashboard": EditableCobasketDashboard,
            "main": main,
        }[name]
    raise AttributeError(name)
