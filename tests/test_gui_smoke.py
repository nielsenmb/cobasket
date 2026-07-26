"""Optional headless smoke test for the PyQt dashboard."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.dashboard import CobasketDashboard
from cobasket.gui.models import portfolio_report_from_dict


def test_dashboard_displays_report_without_network_access():
    """The dashboard should populate from a serialized report offline."""
    app = QApplication.instance() or QApplication([])
    window = CobasketDashboard()
    report = portfolio_report_from_dict(
        {
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
                    "warnings": [],
                }
            ],
            "basket_diagnostics": [],
            "warnings": [],
            "metadata": {},
        }
    )
    window.set_report(report)
    assert window.table.rowCount() == 1
    assert window.total_value_label.text() == "Total value: $300.00"
    window.close()
    app.processEvents()
