"""Headless smoke tests for the recommendation-history dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from cobasket.gui.history_dialog import RecommendationHistoryDialog
from cobasket.history import RecommendationHistoryStore
from cobasket.workflow import PortfolioReport, TickerReport


def _make_report(probability: float | None) -> PortfolioReport:
    """Create a single-ticker history report with optional calibration."""
    ticker = TickerReport(
        ticker="AAPL",
        held_quantity=1.0,
        current_price=100.0,
        market_value=100.0,
        evidence_score=0.2,
        evidence_confidence=0.8,
        probability_outperform=probability,
        probability_lower=0.55 if probability is not None else None,
        probability_upper=0.74 if probability is not None else None,
        calibration_sample_count=40 if probability is not None else None,
        recommendation="Buy",
        explanation="Synthetic GUI test.",
        basket_memberships=(("AAPL", "MSFT"),),
    )
    return PortfolioReport(
        generated_at_utc="2026-01-01T12:00:00+00:00",
        latest_price_date="2026-01-01",
        cash=1000.0,
        invested_value=100.0,
        total_value=1100.0,
        tickers=(ticker,),
        basket_diagnostics=(),
        warnings=(),
    )


def test_history_dialog_renders_stored_report(tmp_path):
    """The timeline should render a calibrated ticker without network access."""
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "history.sqlite"
    RecommendationHistoryStore(path).record_report(_make_report(0.65))

    dialog = RecommendationHistoryDialog(path, ticker="AAPL")
    assert dialog.ticker_combo.currentText() == "AAPL"
    assert dialog.history_table.rowCount() == 1
    assert "Buy" in dialog.summary.text()
    dialog.close()
    app.processEvents()


def test_history_dialog_renders_uncalibrated_report(tmp_path):
    """Missing probabilities should display safely rather than crashing Qt."""
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "history.sqlite"
    RecommendationHistoryStore(path).record_report(_make_report(None))

    dialog = RecommendationHistoryDialog(path, ticker="AAPL")
    assert dialog.history_table.rowCount() == 1
    assert dialog.history_table.item(0, 2).text() == "—"
    assert "No calibrated probabilities" in dialog.figure.axes[0].texts[0].get_text()
    dialog.close()
    app.processEvents()
