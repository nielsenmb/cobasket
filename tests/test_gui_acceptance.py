"""Headless acceptance tests for top-level Cobasket GUI actions."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.editable_dashboard import EditableCobasketDashboard


def _report_payload() -> dict[str, object]:
    """Return a minimal saved-report payload for path-resolution tests."""
    return {
        "generated_at_utc": "2026-08-11T10:00:00+00:00",
        "latest_price_date": "2026-08-10T00:00:00",
        "cash": 100.0,
        "invested_value": 0.0,
        "total_value": 100.0,
        "tickers": [],
        "basket_diagnostics": [],
        "warnings": [],
        "metadata": {},
    }


def test_history_path_uses_saved_report_directory_when_no_config(tmp_path):
    """A display-only session should look for history beside its saved report."""
    app = QApplication.instance() or QApplication([])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report_payload()), encoding="utf-8")

    window = EditableCobasketDashboard()
    window.report_path_edit.setText(str(report_path))
    assert window._history_path() == tmp_path / "cobasket_history.sqlite"
    window.close()
    app.processEvents()


def test_all_standalone_menu_dialog_constructors_are_guarded(monkeypatch):
    """Constructor failures in optional workflows must not terminate the dashboard."""
    app = QApplication.instance() or QApplication([])
    window = EditableCobasketDashboard()
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "cobasket.gui.editable_dashboard.QMessageBox.critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    class BrokenDialog:
        """Synthetic dialog that fails during construction."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("synthetic constructor failure")

    actions = (
        ("StrategyExperimentDialog", window.open_strategy_experiments),
        ("RepeatedWalkForwardDialog", window.open_repeated_walk_forward),
        ("ContinuousWalkForwardDialog", window.open_continuous_walk_forward),
        ("ValidationDialog", window.open_validation),
        ("RecommendationHistoryDialog", window.open_recommendation_history),
    )
    for class_name, callback in actions:
        monkeypatch.setattr(f"cobasket.gui.editable_dashboard.{class_name}", BrokenDialog)
        callback()

    assert len(messages) == len(actions)
    assert all("synthetic constructor failure" in message for _, message in messages)
    window.close()
    app.processEvents()
