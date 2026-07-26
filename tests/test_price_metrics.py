"""Tests for leakage-safe price metrics and rule-engine integration."""

import numpy as np
import pandas as pd
import pytest

from cobasket.price_metrics import (
    PriceMetricConfig,
    build_price_metrics,
    compare_incremental_metric_strategies,
    merge_metric_tables,
    momentum_score,
    trailing_momentum,
    trailing_percentile,
    trailing_trend_distance,
    trailing_volatility,
    trend_score,
)
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


def _prices() -> pd.DataFrame:
    index = pd.date_range("2022-01-03", periods=320, freq="B")
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 * np.exp(0.0010 * x + 0.015 * np.sin(x / 11.0)),
            "BBB": 100.0 * np.exp(0.0003 * x + 0.010 * np.cos(x / 9.0)),
        },
        index=index,
    )


def test_momentum_is_trailing_return_and_bounded_score():
    """Momentum should use only the current and lagged prices."""
    prices = _prices()
    raw = trailing_momentum(prices, window=20)
    expected = prices.iloc[20] / prices.iloc[0] - 1.0
    pd.testing.assert_series_equal(raw.iloc[20], expected, check_names=False)
    score = momentum_score(prices, window=20, scale=0.10)
    assert score.abs().max().max() <= 1.0
    assert score.iloc[:20].isna().all().all()


def test_trend_uses_trailing_moving_average():
    """Trend distance should equal price divided by its trailing mean minus one."""
    prices = _prices()
    distance = trailing_trend_distance(prices, window=30)
    date = prices.index[29]
    expected = prices.loc[date] / prices.iloc[:30].mean() - 1.0
    pd.testing.assert_series_equal(distance.loc[date], expected, check_names=False)
    score = trend_score(prices, window=30, scale=0.05)
    assert score.abs().max().max() <= 1.0


def test_volatility_is_annualised_trailing_return_scatter():
    """Volatility should match an explicitly calculated trailing sample standard deviation."""
    prices = _prices()
    result = trailing_volatility(prices, window=20, periods_per_year=252)
    returns = prices["AAA"].pct_change(fill_method=None)
    expected = returns.iloc[1:21].std(ddof=1) * np.sqrt(252)
    assert result["AAA"].iloc[20] == pytest.approx(expected)


def test_trailing_percentile_does_not_change_past_when_future_is_appended():
    """Appending future observations must not alter earlier percentile values."""
    prices = _prices()
    volatility = trailing_volatility(prices.iloc[:250], window=20)
    original = trailing_percentile(volatility, window=80)
    extended_volatility = trailing_volatility(prices, window=20)
    extended = trailing_percentile(extended_volatility, window=80)
    pd.testing.assert_frame_equal(original, extended.loc[original.index])


def test_build_price_metrics_returns_rule_ready_tables():
    """All generated tables should align and high-volatility flags should be binary."""
    prices = _prices()
    metrics = build_price_metrics(
        prices,
        config=PriceMetricConfig(
            momentum_window=20,
            trend_window=30,
            volatility_window=15,
            volatility_baseline_window=60,
        ),
    )
    assert set(metrics) == {
        "momentum",
        "momentum_return",
        "trend",
        "trend_distance",
        "volatility",
        "volatility_percentile",
        "high_volatility",
    }
    for table in metrics.values():
        assert table.index.equals(prices.index)
        assert table.columns.equals(prices.columns)
    flags = metrics["high_volatility"].stack().dropna().unique()
    assert set(flags).issubset({0.0, 1.0})


def test_merge_metric_tables_rejects_duplicate_names():
    """Metric name collisions should be explicit rather than silently overwritten."""
    table = _prices()
    with pytest.raises(ValueError, match="duplicate metric names"):
        merge_metric_tables({"momentum": table}, {"momentum": table})


def test_incremental_strategy_comparison_uses_same_metric_history():
    """A momentum filter should be comparable with a probability-only strategy."""
    prices = _prices()
    dates = prices.index[100::20]
    probability = pd.DataFrame(0.65, index=dates, columns=prices.columns)
    price_metrics = build_price_metrics(
        prices,
        config=PriceMetricConfig(
            momentum_window=20,
            trend_window=30,
            volatility_window=15,
            volatility_baseline_window=60,
        ),
    )
    base = StrategyRules(
        name="probability only",
        rules=(
            StrategyRule(
                "buy",
                (MetricCondition("probability", ">=", 0.60),),
                0.20,
            ),
        ),
        transaction_cost_bps=0.0,
    )
    filtered = StrategyRules(
        name="probability and momentum",
        rules=(
            StrategyRule(
                "buy",
                (
                    MetricCondition("probability", ">=", 0.60),
                    MetricCondition("momentum", ">", 0.0),
                ),
                0.20,
            ),
        ),
        transaction_cost_bps=0.0,
    )
    summary, results = compare_incremental_metric_strategies(
        prices,
        {"probability": probability},
        price_metrics,
        (base, filtered),
        initial_cash=10_000.0,
    )
    assert list(summary.index) == ["probability only", "probability and momentum"]
    assert set(results) == set(summary.index)
    assert not results["probability only"].decisions.empty


def test_appending_future_prices_does_not_change_existing_metrics():
    """All completed metric rows should remain identical when future prices are added."""
    prices = _prices()
    config = PriceMetricConfig(
        momentum_window=20,
        trend_window=30,
        volatility_window=15,
        volatility_baseline_window=60,
    )
    short = build_price_metrics(prices.iloc[:240], config=config)
    full = build_price_metrics(prices, config=config)
    for name, table in short.items():
        pd.testing.assert_frame_equal(table, full[name].loc[table.index])
