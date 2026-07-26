"""Tests for declarative strategy rules and historical comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobasket.strategy_rules import (
    MetricCondition,
    StrategyRule,
    StrategyRules,
    compare_rule_strategies,
    run_rule_strategy_backtest,
)


def _prices() -> pd.DataFrame:
    """Return deterministic two-asset prices for rule tests."""
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100, 101, 103, 105, 107, 106, 104, 102],
            "BBB": [100, 100, 101, 101, 102, 102, 103, 103],
        },
        index=index,
        dtype=float,
    )


def _strategy(name: str = "probability rules") -> StrategyRules:
    """Build a compact buy, reduce, and sell strategy."""
    return StrategyRules(
        name=name,
        transaction_cost_bps=0.0,
        rules=(
            StrategyRule(
                "sell",
                (MetricCondition("probability", "<=", 0.30),),
                target_weight=0.0,
            ),
            StrategyRule(
                "strong buy",
                (
                    MetricCondition("probability", ">=", 0.70),
                    MetricCondition("stable", "==", True),
                ),
                target_weight=0.50,
            ),
            StrategyRule(
                "buy",
                (
                    MetricCondition("probability", ">=", 0.60),
                    MetricCondition("stable", "==", True),
                ),
                target_weight=0.25,
            ),
            StrategyRule(
                "reduce",
                (
                    MetricCondition("probability", "<=", 0.40),
                    MetricCondition("is_held", "==", True),
                ),
                target_weight=0.10,
            ),
        ),
    )


def test_first_matching_rule_has_priority() -> None:
    """More specific rules should win when they appear earlier."""
    action, weight = _strategy().decide(
        {"probability": 0.75, "stable": True, "is_held": False},
        current_weight=0.0,
    )
    assert action == "strong buy"
    assert weight == pytest.approx(0.50)


def test_missing_metric_does_not_trigger_rule() -> None:
    """Unavailable metrics should fail their condition rather than trade."""
    action, weight = _strategy().decide(
        {"probability": 0.75},
        current_weight=0.20,
    )
    assert action == "hold"
    assert weight == pytest.approx(0.20)


def test_strategy_json_round_trip(tmp_path) -> None:
    """Saved strategies should preserve order, thresholds, and costs."""
    original = _strategy()
    path = original.save(tmp_path / "strategy.json")
    restored = StrategyRules.load(path)
    assert restored == original
    assert restored.rules[0].action == "sell"


def test_backtest_executes_decisions_on_next_observation() -> None:
    """A decision dated on one bar should trade at the following bar."""
    prices = _prices()
    probability = pd.DataFrame(
        {"AAA": [0.75, 0.20], "BBB": [0.50, 0.50]},
        index=prices.index[[1, 4]],
    )
    stable = pd.DataFrame(
        {"AAA": [1.0, 1.0], "BBB": [1.0, 1.0]},
        index=probability.index,
    )
    result = run_rule_strategy_backtest(
        prices,
        {"probability": probability, "stable": stable},
        _strategy(),
        initial_cash=1000.0,
    )
    trades = result.backtest.trades
    assert trades.iloc[0]["date"] == prices.index[2]
    assert trades.iloc[0]["action"] == "strong buy"
    assert trades.iloc[-1]["date"] == prices.index[5]
    assert trades.iloc[-1]["action"] == "sell"


def test_rule_backtest_supports_reentry_after_complete_sale() -> None:
    """A fully sold ticker should remain eligible for a later buy rule."""
    prices = _prices()
    dates = prices.index[[0, 2, 4]]
    probability = pd.DataFrame(
        {"AAA": [0.75, 0.20, 0.75], "BBB": [0.50, 0.50, 0.50]},
        index=dates,
    )
    stable = pd.DataFrame(1.0, index=dates, columns=prices.columns)
    result = run_rule_strategy_backtest(
        prices,
        {"probability": probability, "stable": stable},
        _strategy(),
        initial_cash=1000.0,
    )
    aaa = result.backtest.trades.loc[result.backtest.trades["ticker"] == "AAA"]
    assert list(aaa["side"]) == ["buy", "sell", "buy"]


def test_compare_rule_strategies_uses_identical_inputs() -> None:
    """Strategy comparison should return one summary and result per rule set."""
    prices = _prices()
    dates = prices.index[[0, 3]]
    probability = pd.DataFrame(
        {"AAA": [0.75, 0.20], "BBB": [0.50, 0.50]},
        index=dates,
    )
    stable = pd.DataFrame(1.0, index=dates, columns=prices.columns)
    conservative = StrategyRules(
        name="conservative",
        transaction_cost_bps=0.0,
        rules=(
            StrategyRule(
                "buy",
                (MetricCondition("probability", ">=", 0.80),),
                target_weight=0.20,
            ),
        ),
    )
    summary, results = compare_rule_strategies(
        prices,
        {"probability": probability, "stable": stable},
        (_strategy("active"), conservative),
        initial_cash=1000.0,
    )
    assert set(summary.index) == {"active", "conservative"}
    assert set(results) == {"active", "conservative"}
    assert summary.loc["active", "trade_count"] > summary.loc["conservative", "trade_count"]


def test_invalid_operator_is_rejected() -> None:
    """Unknown comparison syntax should fail during configuration."""
    with pytest.raises(ValueError, match="unsupported operator"):
        MetricCondition("probability", "approximately", 0.5)
