"""Tests for leakage-free basket strategy simulation helpers."""

import numpy as np
import pandas as pd

from cobasket.strategy_simulation import (
    BasketStrategyConfig,
    equal_weight_benchmark,
    expanding_calibrated_probabilities,
    run_basket_strategy_simulation,
)


def synthetic_prices(n: int = 520) -> pd.DataFrame:
    """Return a stable two-asset cointegrated price table."""
    rng = np.random.default_rng(42)
    common = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    stationary = rng.normal(0.0, 1.0, n)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"AAA": common + stationary, "BBB": common - stationary}, index=index)


def test_expanding_calibration_is_neutral_before_outcomes_mature():
    """Early decisions must not use outcomes that occur in the future."""
    dates = pd.date_range("2024-01-01", periods=4, freq="10D")
    records = pd.DataFrame(
        {
            "evaluation_date": dates,
            "future_date": dates + pd.Timedelta(days=20),
            "ticker": ["AAA"] * 4,
            "score": [0.8, 0.8, 0.8, 0.8],
            "outperformed": [1, 1, 1, 1],
        }
    )
    probabilities = expanding_calibrated_probabilities(records, horizon=20, min_samples=2)
    assert probabilities.iloc[0, 0] == 0.5
    assert probabilities.iloc[1, 0] == 0.5
    assert probabilities.iloc[-1, 0] > 0.5


def test_equal_weight_benchmark_starts_at_requested_capital():
    """The comparison portfolio should share the strategy's starting value."""
    benchmark = equal_weight_benchmark(synthetic_prices(20), 10_000.0)
    assert np.isclose(benchmark.iloc[0], 10_000.0)


def test_complete_strategy_returns_profit_and_trade_outputs():
    """A complete walk-forward simulation should produce inspectable outputs."""
    result = run_basket_strategy_simulation(
        synthetic_prices(),
        config=BasketStrategyConfig(
            train_window=120,
            z_window=30,
            horizon=10,
            step=10,
            min_calibration_samples=4,
            initial_cash=10_000.0,
        ),
    )
    assert not result.records.empty
    assert not result.probabilities.empty
    assert np.isfinite(result.summary["profit"])
    assert np.isclose(result.backtest.equity.iloc[0], 10_000.0)
    assert result.benchmark_equity.index.equals(result.backtest.equity.index)
