"""Tests for continuous walk-forward strategy deployment."""

import numpy as np
import pandas as pd
import pytest

from cobasket.continuous_walk_forward import (
    ContinuousDeploymentConfig,
    run_continuous_walk_forward,
)
from cobasket.repeated_walk_forward import WalkForwardConfig
from cobasket.strategy_experiments import StrategyExperimentConfig
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


def _buy_strategy(name: str = "buy") -> StrategyRules:
    """Return a simple half-weight strategy for synthetic tests."""
    return StrategyRules(
        name=name,
        rules=(
            StrategyRule(
                action="buy",
                conditions=(MetricCondition("probability", ">=", 0.5),),
                target_weight=0.5,
            ),
        ),
        transaction_cost_bps=5.0,
    )


def _cash_strategy(name: str = "cash") -> StrategyRules:
    """Return a strategy that never opens a position."""
    return StrategyRules(
        name=name,
        rules=(
            StrategyRule(
                action="sell",
                conditions=(MetricCondition("probability", "<", 0.0),),
                target_weight=0.0,
            ),
        ),
    )


def _inputs(periods: int = 80) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build deterministic rising prices and positive probabilities."""
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 150.0, periods),
            "BBB": np.linspace(80.0, 104.0, periods),
        },
        index=index,
    )
    probability = pd.DataFrame(0.8, index=index, columns=prices.columns)
    return prices, {"probability": probability}


def _walk_forward() -> WalkForwardConfig:
    """Return compact non-overlapping folds for tests."""
    return WalkForwardConfig(
        train_observations=20,
        validation_observations=10,
        test_observations=10,
        step_observations=10,
    )


def _experiment() -> StrategyExperimentConfig:
    """Return interval-size settings compatible with compact folds."""
    return StrategyExperimentConfig(
        selection_metric="total_return",
        initial_cash=10_000.0,
        minimum_train_observations=5,
        minimum_validation_observations=5,
        minimum_test_observations=5,
    )


def test_continuous_account_carries_positions_between_folds():
    """A position should survive fold boundaries under the retain policy."""
    prices, metrics = _inputs()
    result = run_continuous_walk_forward(
        prices,
        metrics,
        [_buy_strategy()],
        walk_forward=_walk_forward(),
        experiment=_experiment(),
        deployment=ContinuousDeploymentConfig(boundary_policy="retain"),
    )

    assert len(result.selections) >= 2
    second_start = pd.Timestamp(result.selections.iloc[1]["test_start"])
    assert result.positions.loc[second_start].sum() > 0.0
    assert result.equity.iloc[-1]["strategy"] > result.equity.iloc[0]["strategy"]
    assert not result.trades.empty
    assert set(result.metrics.index) == {"strategy", "equal_weight", "cash"}


def test_decisions_execute_on_next_observation():
    """The first rule decision must not trade at the same closing price."""
    prices, metrics = _inputs()
    result = run_continuous_walk_forward(
        prices,
        metrics,
        [_buy_strategy()],
        walk_forward=_walk_forward(),
        experiment=_experiment(),
    )

    first_decision = pd.Timestamp(result.decisions.iloc[0]["date"])
    first_trade = pd.Timestamp(result.trades.iloc[0]["date"])
    expected = prices.index[prices.index.get_loc(first_decision) + 1]
    assert first_trade == expected


def test_boundary_liquidation_is_applied_when_strategy_changes(monkeypatch):
    """The liquidation policy should close holdings at a changed strategy boundary."""
    prices, metrics = _inputs(periods=60)
    index = prices.index
    buy = _buy_strategy("first")
    second = _buy_strategy("second")
    selections = [
        {
            "fold": 1,
            "train_start": index[0],
            "train_end": index[19],
            "validation_start": index[20],
            "validation_end": index[29],
            "test_start": index[30],
            "test_end": index[39],
            "selected_strategy": buy.name,
            "strategy": buy,
        },
        {
            "fold": 2,
            "train_start": index[10],
            "train_end": index[29],
            "validation_start": index[30],
            "validation_end": index[39],
            "test_start": index[40],
            "test_end": index[49],
            "selected_strategy": second.name,
            "strategy": second,
        },
    ]

    monkeypatch.setattr(
        "cobasket.continuous_walk_forward._select_strategies",
        lambda *args, **kwargs: (selections, []),
    )
    result = run_continuous_walk_forward(
        prices,
        metrics,
        [buy, second],
        walk_forward=_walk_forward(),
        experiment=_experiment(),
        deployment=ContinuousDeploymentConfig(boundary_policy="liquidate"),
    )

    boundary_trades = result.trades[result.trades["action"] == "boundary_liquidation"]
    assert not boundary_trades.empty
    assert (boundary_trades["date"] == index[40]).all()
    assert any("liquidated" in warning for warning in result.warnings)


def test_future_test_prices_do_not_change_strategy_selection():
    """Changing later test prices must not alter earlier validation selections."""
    prices, metrics = _inputs()
    strategies = [_buy_strategy(), _cash_strategy()]
    original = run_continuous_walk_forward(
        prices,
        metrics,
        strategies,
        walk_forward=_walk_forward(),
        experiment=_experiment(),
    )

    changed = prices.copy()
    final_start = pd.Timestamp(original.selections.iloc[-1]["test_start"])
    changed.loc[final_start:, "AAA"] *= np.linspace(
        1.0,
        0.5,
        len(changed.loc[final_start:]),
    )
    repeated = run_continuous_walk_forward(
        changed,
        metrics,
        strategies,
        walk_forward=_walk_forward(),
        experiment=_experiment(),
    )

    pd.testing.assert_series_equal(
        original.selections["selected_strategy"],
        repeated.selections["selected_strategy"],
    )


def test_overlapping_test_folds_are_rejected_for_continuous_account():
    """One account cannot assign two selected strategies to the same date."""
    prices, metrics = _inputs()
    with pytest.raises(ValueError, match="non-overlapping"):
        run_continuous_walk_forward(
            prices,
            metrics,
            [_buy_strategy()],
            walk_forward=WalkForwardConfig(
                train_observations=20,
                validation_observations=10,
                test_observations=10,
                step_observations=5,
                allow_overlapping_tests=True,
            ),
            experiment=_experiment(),
        )


def test_result_exports_all_account_tables(tmp_path):
    """Continuous results should export reproducible account state."""
    prices, metrics = _inputs()
    result = run_continuous_walk_forward(
        prices,
        metrics,
        [_buy_strategy()],
        walk_forward=_walk_forward(),
        experiment=_experiment(),
    )
    output = result.save(tmp_path / "continuous")

    assert (output / "continuous_equity.csv").exists()
    assert (output / "continuous_trades.csv").exists()
    assert (output / "continuous_selections.csv").exists()
    assert (output / "continuous_walk_forward.json").exists()
