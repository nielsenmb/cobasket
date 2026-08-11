"""Optional headless smoke tests for the PyQt dashboard."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.dashboard import CobasketDashboard, classify_cobasket_json
from cobasket.gui.guided_dashboard import GuidedCobasketDashboard
from cobasket.gui.models import portfolio_report_from_dict


def _report_payload():
    """Return a minimal serialized report payload for GUI tests."""
    return {
        "generated_at_utc": "2026-07-26T12:00:00+00:00",
        "latest_price_date": "2026-07-25T00:00:00",
        "cash": 100.0,
        "invested_value": 200.0,
        "total_value": 300.0,
        "tickers": [
            {
                "ticker": "AAPL",
                "held_quantity": 1.0,
                "current_price": 200.0,
                "market_value": 200.0,
                "evidence_score": 0.5,
                "evidence_confidence": 0.7,
                "probability_outperform": 0.65,
                "probability_lower": 0.52,
                "probability_upper": 0.76,
                "calibration_sample_count": 30,
                "recommendation": "Add",
                "explanation": "Positive relative evidence.",
                "basket_memberships": [["AAPL", "MSFT"]],
                "native_currency": "USD",
                "base_currency": "USD",
                "fx_rate_to_base": 1.0,
                "warnings": [],
            }
        ],
        "basket_diagnostics": [],
        "warnings": [],
        "metadata": {"base_currency": "USD"},
    }


def test_dashboard_displays_report_without_network_access():
    """The dashboard should populate from a serialized report offline."""
    app = QApplication.instance() or QApplication([])
    window = CobasketDashboard()
    window.set_report(portfolio_report_from_dict(_report_payload()))
    assert window.table.rowCount() == 1
    assert window.total_value_label.text() == "Total value: $300.00"
    window.close()
    app.processEvents()


def test_guided_dashboard_constructs_workflow_menu():
    """The guided dashboard should construct its Workflow menu without error."""
    app = QApplication.instance() or QApplication([])
    window = GuidedCobasketDashboard()
    menus = [action.text() for action in window.menuBar().actions()]
    assert "Workflow" in menus
    window.close()
    app.processEvents()


def test_json_classifier_distinguishes_portfolio_and_report(tmp_path):
    """Report output must never be interpreted as a portfolio configuration."""
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report_payload()), encoding="utf-8")
    config_path = tmp_path / "portfolio.json"
    config_path.write_text(
        json.dumps({"holdings": {}, "cash": 100.0, "watchlist_path": "watchlist.json"}),
        encoding="utf-8",
    )
    assert classify_cobasket_json(report_path) == "report"
    assert classify_cobasket_json(config_path) == "portfolio"


def test_dashboard_uses_separate_config_and_report_fields():
    """The dashboard should expose distinct paths for analysis and saved reports."""
    app = QApplication.instance() or QApplication([])
    window = CobasketDashboard()
    assert window.path_edit is not window.report_path_edit
    assert "portfolio" in window.path_edit.placeholderText().lower()
    assert "report" in window.report_path_edit.placeholderText().lower()
    window.close()
    app.processEvents()
