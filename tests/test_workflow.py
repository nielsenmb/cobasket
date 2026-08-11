"""Tests for the end-to-end portfolio reporting workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cobasket.basket_validation import BasketValidationProfile, BasketValidationSet
from cobasket.evidence import BasketWatchlist, ProbabilityCalibration
from cobasket.workflow import PortfolioAnalyzer, PortfolioConfig


@dataclass
class _Metadata:
    """Minimal fake download metadata."""

    source: str = "test"


class _FakeDataManager:
    """Return a fixed price table without network access."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices
        self.last_metadata = _Metadata()

    def prices(self, tickers, **kwargs):
        """Return requested fixed-price columns."""
        return self._prices.loc[:, list(tickers)]


def _prices() -> pd.DataFrame:
    """Create a reproducible synthetic cointegrated pair."""
    rng = np.random.default_rng(42)
    n = 260
    common = 100 + np.cumsum(rng.normal(0, 0.5, n))
    stationary = rng.normal(0, 0.4, n)
    index = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="B")
    return pd.DataFrame({"AAA": common + stationary, "BBB": common - stationary}, index=index)


def _calibration() -> ProbabilityCalibration:
    """Create a small deterministic calibration table."""
    table = pd.DataFrame(
        {
            "score_lower": [-1.0, 0.0],
            "score_upper": [0.0, 1.0],
            "sample_count": [40, 40],
            "successes": [12, 28],
            "probability_mean": [0.30, 0.70],
            "probability_lower": [0.22, 0.60],
            "probability_upper": [0.40, 0.78],
        }
    )
    return ProbabilityCalibration(table=table, score_edges=(-1.0, 0.0, 1.0), horizon=20)


def _validation(status: str) -> BasketValidationSet:
    """Create one deterministic basket validation profile."""
    profile = BasketValidationProfile(
        basket=("AAA", "BBB"),
        status=status,
        current_trace_ratio=1.2,
        accepted_evaluations=30,
        possible_evaluations=40,
        acceptance_rate=0.75,
        weight_stability=0.9,
        score_return_correlation=0.2 if status == "validated" else -0.1,
        positive_outperform_rate=0.7 if status == "validated" else 0.4,
        negative_outperform_rate=0.4 if status == "validated" else 0.6,
        calibration_contrast=0.3 if status == "validated" else -0.2,
        records=60,
        reasons=("test profile",),
    )
    return BasketValidationSet(
        generated_at_utc="2026-08-11T12:00:00+00:00",
        train_window=252,
        z_window=30,
        horizon=20,
        step=20,
        min_trace_ratio=0.01,
        profiles=(profile,),
    )


def test_portfolio_config_round_trip(tmp_path):
    """Portfolio configuration should preserve normalized holdings."""
    config = PortfolioConfig(
        holdings={"aaa": 2.5, "bbb": 0.0},
        cash=500.0,
        watchlist_path="watchlist.json",
        validation_path="validation.json",
    )
    path = config.save(tmp_path / "portfolio.json")
    loaded = PortfolioConfig.load(path)
    assert loaded.holdings == {"AAA": 2.5, "BBB": 0.0}
    assert loaded.cash == 500.0
    assert loaded.validation_path == "validation.json"


def test_probability_calibration_round_trip(tmp_path):
    """Calibration JSON should reconstruct the probability lookup."""
    calibration = _calibration()
    path = calibration.save(tmp_path / "calibration.json")
    loaded = ProbabilityCalibration.load(path)
    assert loaded.score_edges == calibration.score_edges
    assert loaded.lookup(0.5)["probability_mean"] == 0.70


def test_analyzer_generates_serializable_report(tmp_path):
    """The analyzer should combine prices, evidence, calibration, and holdings."""
    watchlist = BasketWatchlist(baskets=(("AAA", "BBB"),), name="Test")
    watchlist_path = watchlist.save(tmp_path / "watchlist.json")
    calibration_path = _calibration().save(tmp_path / "calibration.json")
    config = PortfolioConfig(
        holdings={"AAA": 2.0, "BBB": 0.0},
        cash=1000.0,
        watchlist_path=str(watchlist_path),
        calibration_path=str(calibration_path),
        z_window=30,
        min_trace_ratio=0.01,
    )
    report = PortfolioAnalyzer(_FakeDataManager(_prices())).run(config)
    assert {item.ticker for item in report.tickers} == {"AAA", "BBB"}
    assert report.total_value > report.cash
    assert all(item.probability_outperform is not None for item in report.tickers)
    output = report.save(tmp_path / "report.json")
    assert output.exists()
    assert not report.table().empty


def test_zero_holding_ticker_remains_in_report(tmp_path):
    """Selling all shares must not remove a watched ticker from analysis."""
    watchlist_path = BasketWatchlist(baskets=(("AAA", "BBB"),)).save(
        tmp_path / "watchlist.json"
    )
    config = PortfolioConfig(
        holdings={"AAA": 0.0, "BBB": 1.0},
        watchlist_path=str(watchlist_path),
        z_window=30,
        min_trace_ratio=0.01,
    )
    report = PortfolioAnalyzer(_FakeDataManager(_prices())).run(config)
    aaa = next(item for item in report.tickers if item.ticker == "AAA")
    assert aaa.held_quantity == 0.0
    assert aaa.market_value == 0.0


def test_missing_calibration_emits_warning(tmp_path):
    """Raw evidence fallback should be explicit in report warnings."""
    watchlist_path = BasketWatchlist(baskets=(("AAA", "BBB"),)).save(
        tmp_path / "watchlist.json"
    )
    config = PortfolioConfig(
        holdings={},
        watchlist_path=str(watchlist_path),
        z_window=30,
        min_trace_ratio=0.01,
    )
    report = PortfolioAnalyzer(_FakeDataManager(_prices())).run(config)
    assert any("No probability calibration" in warning for warning in report.warnings)
    assert all(item.probability_outperform is None for item in report.tickers)


def test_weak_validation_profile_gates_live_actions(tmp_path):
    """A weak basket should retain diagnostics but suppress actionable labels."""
    watchlist_path = BasketWatchlist(baskets=(("AAA", "BBB"),)).save(
        tmp_path / "watchlist.json"
    )
    calibration_path = _calibration().save(tmp_path / "calibration.json")
    validation_path = _validation("weak").save(tmp_path / "validation.json")
    config = PortfolioConfig(
        holdings={"AAA": 0.0, "BBB": 1.0},
        watchlist_path=str(watchlist_path),
        calibration_path=str(calibration_path),
        validation_path=str(validation_path),
        z_window=30,
        min_trace_ratio=0.01,
    )
    report = PortfolioAnalyzer(_FakeDataManager(_prices())).run(config)
    for item in report.tickers:
        expected = "Hold" if item.held_quantity > 0 else "Wait"
        assert item.recommendation == expected
        assert item.probability_outperform is not None
        assert item.basket_validation[0]["status"] == "weak"
        assert any("No supporting basket" in warning for warning in item.warnings)
