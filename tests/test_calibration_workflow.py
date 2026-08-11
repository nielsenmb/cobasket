"""Tests for watchlist-level probability calibration."""

from __future__ import annotations

import json

import pandas as pd

from cobasket.calibration_cli import _update_portfolio_calibration
from cobasket.calibration_workflow import calibrate_watchlist


class FakeManager:
    """Return deterministic prices without network access."""

    def prices(self, tickers, period="2y", **kwargs):
        """Return an aligned synthetic price table."""
        index = pd.date_range("2020-01-01", periods=12, freq="D")
        return pd.DataFrame({ticker: range(10, 22) for ticker in tickers}, index=index, dtype=float)


def _records(tickers: list[str], start_score: float) -> pd.DataFrame:
    """Create deterministic walk-forward evidence records."""
    rows = []
    for i, ticker in enumerate(tickers):
        score = start_score + 0.1 * i
        rows.append(
            {
                "evaluation_date": pd.Timestamp("2020-01-05"),
                "future_date": pd.Timestamp("2020-01-07"),
                "ticker": ticker,
                "score": score,
                "confidence": abs(score),
                "z_score": score,
                "weight": 0.5,
                "trace_ratio": 1.2,
                "asset_return": 0.02 if i == 0 else 0.0,
                "basket_return": 0.01,
                "excess_return": 0.01 if i == 0 else -0.01,
                "outperformed": 1 if i == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_calibrate_watchlist_pools_baskets(monkeypatch, tmp_path):
    """Calibration should pool only leakage-safe records generated per basket."""
    watchlist = tmp_path / "watchlist.json"
    watchlist.write_text(
        json.dumps({"name": "test", "baskets": [["AAA", "BBB"], ["CCC", "DDD"]]}),
        encoding="utf-8",
    )
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(
        json.dumps(
            {
                "holdings": {},
                "cash": 100.0,
                "watchlist_path": "watchlist.json",
                "calibration_path": None,
                "period": "5y",
                "z_window": 5,
                "min_trace_ratio": 1.0,
                "max_price_age_days": 7.0,
            }
        ),
        encoding="utf-8",
    )

    def fake_walk_forward(prices, **kwargs):
        start = -0.7 if list(prices.columns)[0] == "AAA" else 0.3
        return _records(list(prices.columns), start)

    monkeypatch.setattr("cobasket.calibration_workflow.walk_forward_evidence", fake_walk_forward)
    result = calibrate_watchlist(portfolio, train_window=5, horizon=2, data_manager=FakeManager())

    assert len(result.records) == 4
    assert set(result.records["basket"]) == {"AAA, BBB", "CCC, DDD"}
    assert result.basket_summary["status"].tolist() == ["ok", "ok"]
    assert int(result.calibration.table["sample_count"].sum()) == 4
    assert result.calibration.horizon == 2


def test_update_portfolio_writes_calibration_path(tmp_path):
    """The CLI helper should link a saved calibration into the portfolio file."""
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(json.dumps({"holdings": {}, "watchlist_path": "watchlist.json"}), encoding="utf-8")
    calibration = tmp_path / "probability_calibration.json"
    calibration.write_text("{}", encoding="utf-8")

    _update_portfolio_calibration(portfolio, calibration)
    payload = json.loads(portfolio.read_text(encoding="utf-8"))
    assert payload["calibration_path"] == str(calibration.resolve())
