"""Tests for historical policy and calibration validation."""

import numpy as np
import pandas as pd

from cobasket.validation import build_validation_result, equal_weight_benchmark


def synthetic_inputs():
    """Return deterministic prices, probabilities, and binary outcomes."""
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    x = np.linspace(100.0, 120.0, len(index))
    prices = pd.DataFrame({"AAA": x, "BBB": 100.0 + 0.5 * (x - 100.0)}, index=index)
    probabilities = pd.DataFrame(
        {"AAA": [0.75, 0.25, 0.75], "BBB": [0.25, 0.75, 0.25]},
        index=index[[10, 30, 50]],
    )
    outcomes = pd.DataFrame(
        {
            "probability_outperform": [0.75, 0.25, 0.75, 0.25],
            "outperformed": [1, 0, 1, 0],
        }
    )
    return prices, probabilities, outcomes


def test_equal_weight_benchmark_starts_at_requested_value():
    """The comparison curve should share the policy's initial scale."""
    prices, _, _ = synthetic_inputs()
    benchmark = equal_weight_benchmark(prices, 10_000.0)
    assert benchmark.iloc[0] == 10_000.0
    assert benchmark.index.equals(prices.index)


def test_build_validation_result_contains_all_dashboard_series():
    """The combined result should expose performance and calibration outputs."""
    prices, probabilities, outcomes = synthetic_inputs()
    result = build_validation_result(prices, probabilities, outcomes=outcomes)
    assert result.backtest.equity.index.equals(prices.index)
    assert result.drawdown.max() <= 0.0
    assert result.invested_fraction.between(0.0, 1.0).all()
    assert result.calibration is not None
    assert result.calibration.sample_count == len(outcomes)
    assert len(result.backtest.trades) > 0


def test_build_validation_result_accepts_policy_only_inputs():
    """Calibration plots should remain optional for policy backtests."""
    prices, probabilities, _ = synthetic_inputs()
    result = build_validation_result(prices, probabilities)
    assert result.calibration is None
