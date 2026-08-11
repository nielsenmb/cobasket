"""PyQt dashboard components for Cobasket.

The PyQt dependency is imported lazily so report-formatting helpers remain
available in non-GUI environments.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CobasketDashboard", "EditableCobasketDashboard", "GuidedCobasketDashboard", "main"]


def __getattr__(name: str) -> Any:
    """Load PyQt dashboard objects only when explicitly requested."""
    if name == "CobasketDashboard":
        from .dashboard import CobasketDashboard

        return CobasketDashboard
    if name == "EditableCobasketDashboard":
        from .editable_dashboard import EditableCobasketDashboard

        return EditableCobasketDashboard
    if name in {"GuidedCobasketDashboard", "main"}:
        from .guided_dashboard import GuidedCobasketDashboard, main

        return {
            "GuidedCobasketDashboard": GuidedCobasketDashboard,
            "main": main,
        }[name]
    raise AttributeError(name)
