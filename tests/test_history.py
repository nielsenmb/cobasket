"""Tests for persistent recommendation and decision history."""

from __future__ import annotations

import pandas as pd
import pytest

from cobasket.history import RecommendationHistoryStore
from cobasket.workflow import PortfolioReport, TickerReport


def _report(timestamp: str, recommendation: str, probability: float) -> PortfolioReport:
    """Build a compact report fixture with one ticker."""
    ticker = TickerReport(
        ticker="AAPL",
        held_quantity=1.0,
        current_price=100.0,
        market_value=100.0,
        evidence_score=probability - 0.5,
        evidence_confidence=0.8,
        probability_outperform=probability,
        probability_lower=max(0.0, probability - 0.1),
        probability_upper=min(1.0, probability + 0.1),
        calibration_sample_count=50,
        recommendation=recommendation,
        explanation="Synthetic test report.",
        basket_memberships=(("AAPL", "MSFT"),),
    )
    return PortfolioReport(
        generated_at_utc=timestamp,
        latest_price_date=timestamp[:10],
        cash=1000.0,
        invested_value=100.0,
        total_value=1100.0,
        tickers=(ticker,),
        basket_diagnostics=(),
        warnings=(),
    )


def test_record_report_is_idempotent_and_detects_transition(tmp_path):
    """Repeated snapshots should not duplicate, while changed labels form transitions."""
    store = RecommendationHistoryStore(tmp_path / "history.sqlite")
    first = _report("2026-01-01T12:00:00+00:00", "Hold", 0.52)
    second = _report("2026-01-02T12:00:00+00:00", "Buy", 0.66)
    store.record_report(first)
    store.record_report(first)
    store.record_report(second)

    history = store.ticker_history("aapl")
    assert len(history) == 2
    transitions = store.transitions("AAPL")
    assert len(transitions) == 1
    assert transitions[0].previous == "Hold"
    assert transitions[0].current == "Buy"


def test_record_action_preserves_user_decision(tmp_path):
    """User actions should remain separate from model recommendations."""
    store = RecommendationHistoryStore(tmp_path / "history.sqlite")
    action_id = store.record_action(
        "aapl",
        "2026-01-02T13:00:00+00:00",
        "no action",
        note="Waiting for more evidence",
    )
    actions = store.actions("AAPL")
    assert action_id > 0
    assert actions.loc[0, "action"] == "no action"
    assert actions.loc[0, "note"] == "Waiting for more evidence"


def test_update_outcomes_adds_available_forward_returns(tmp_path):
    """Stored recommendations should receive forward returns without duplication."""
    store = RecommendationHistoryStore(tmp_path / "history.sqlite")
    store.record_report(_report("2026-01-01T12:00:00+00:00", "Buy", 0.65))
    index = pd.date_range("2026-01-01", periods=8, freq="B")
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 104, 105, 106, 107, 108]}, index=index)

    assert store.update_outcomes(prices, horizons=(1, 5)) == 2
    assert store.update_outcomes(prices, horizons=(1, 5)) == 0
    outcomes = store.outcome_history("AAPL")
    assert set(outcomes["horizon"]) == {1, 5}
    one_day = outcomes.loc[outcomes["horizon"] == 1, "forward_return"].iloc[0]
    assert one_day == pytest.approx(0.01)


def test_tickers_lists_symbols_with_history(tmp_path):
    """The history store should expose symbols represented by report snapshots."""
    store = RecommendationHistoryStore(tmp_path / "history.sqlite")
    store.record_report(_report("2026-01-01T12:00:00+00:00", "Hold", 0.5))
    assert store.tickers() == ("AAPL",)
