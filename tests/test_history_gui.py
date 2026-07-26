"""Headless smoke test for the recommendation-history dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from cobasket.gui.history_dialog import RecommendationHistoryDialog
from cobasket.history import RecommendationHistoryStore
from cobasket.workflow import PortfolioReport, TickerReport


def test_history_dialog_renders_stored_report(tmp_path):
    """The timeline should render a stored ticker without network access."""
    app = QApplication.instance() or QApplication([])
    ticker = TickerReport(
        ticker="AAPL",
        held_quantity=1.0,
        current_price=100.0,
        market_value=100.0,
        evidence_score=0.2,
        evidence_confidence=0.8,
        probability_outperform=0.65,
        probability_lower=0.55,
        probability_upper=0.74,
        calibration_sample_count=40,
        recommendation="Buy",
        explanation="Synthetic GUI test.",
        basket_memberships=(("AAPL", "MSFT"),),
    )
    report = PortfolioReport(
        generated_at_utc="2026-01-01T12:00:00+00:00",
        latest_price_date="2026-01-01",
        cash=1000.0,
        invested_value=100.0,
        total_value=1100.0,
        tickers=(ticker,),
        basket_diagnostics=(),
        warnings=(),
    )
    path = tmp_path / "history.sqlite"
    RecommendationHistoryStore(path).record_report(report)

    dialog = RecommendationHistoryDialog(path, ticker="AAPL")
    assert dialog.ticker_combo.currentText() == "AAPL"
    assert dialog.history_table.rowCount() == 1
    assert "Buy" in dialog.summary.text()
    dialog.close()
    app.processEvents()
