"""Tests for repeated chronological strategy experiments."""

import numpy as np
import pandas as pd
import pytest

from cobasket.repeated_walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
    run_repeated_walk_forward,
)
from cobasket.strategy_experiments import StrategyExperimentConfig
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


def _prices_and_metrics(n: int = 180):
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    x = np.arange(n, dtype=float)
    prices = pd.DataFrame(
        {
            "AAA": 100.0 * np.exp(0.0015 * x),
            "BBB": 100.0 * np.exp(0.0004 * x),
        },
        index=index,
    )
    probability = pd.DataFrame(
        {
            "AAA": np.where((x // 30) % 2 == 0, 0.75, 0.55),
            "BBB": np.where((x // 30) % 2 == 0, 0.55, 0.75),
        },
        index=index,
    )
    return prices, {"probability": probability}


def _strategies():
    conservative = StrategyRules(
        name="conservative",
        rules=(
            StrategyRule(
                "buy",
                (MetricCondition("probability", ">=", 0.70),),
                0.25,
            ),
        ),
        transaction_cost_bps=0.0,
    )
    permissive = StrategyRules(
        name="permissive",
        rules=(
            StrategyRule(
                "buy",
                (MetricCondition("probability", ">=", 0.50),),
                0.25,
            ),
        ),
        transaction_cost_bps=0.0,
    )
    return conservative, permissive


def test_generate_walk_forward_splits_are_ordered_and_non_overlapping():
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    config = WalkForwardConfig(
        train_observations=40,
        validation_observations=20,
        test_observations=20,
        step_observations=20,
    )
    splits = generate_walk_forward_splits(index, config=config)
    assert len(splits) == 5
    for previous, current in zip(splits, splits[1:]):
        assert previous.test[1] < current.test[0]
        assert current.train[0] > previous.train[0]


def test_overlapping_tests_require_explicit_opt_in():
    with pytest.raises(ValueError, match="overlapping"):
        WalkForwardConfig(
            train_observations=40,
            validation_observations=20,
            test_observations=20,
            step_observations=10,
        )


def test_repeated_walk_forward_reports_folds_and_selection_frequency():
    prices, metrics = _prices_and_metrics()
    strategies = _strategies()
    result = run_repeated_walk_forward(
        prices,
        metrics,
        strategies,
        walk_forward=WalkForwardConfig(
            train_observations=50,
            validation_observations=25,
            test_observations=25,
            step_observations=25,
        ),
        experiment=StrategyExperimentConfig(
            selection_metric="total_return",
            initial_cash=10_000.0,
            minimum_train_observations=20,
            minimum_validation_observations=20,
            minimum_test_observations=20,
        ),
    )
    assert len(result.folds) == len(result.fold_table)
    assert result.selection_frequency["selected_folds"].sum() == len(result.folds)
    assert set(result.selection_frequency.index) == {item.name for item in strategies}
    assert list(result.compounded_equity.columns) == ["strategy", "equal_weight", "cash"]
    assert len(result.compounded_equity) == len(result.folds) + 1
    assert np.isfinite(result.fold_table["excess_return"]).all()


def test_future_data_after_final_test_end_cannot_change_existing_result():
    prices, metrics = _prices_and_metrics(180)
    strategies = _strategies()
    walk = WalkForwardConfig(
        train_observations=50,
        validation_observations=25,
        test_observations=25,
        step_observations=25,
    )
    experiment = StrategyExperimentConfig(
        selection_metric="total_return",
        minimum_train_observations=20,
        minimum_validation_observations=20,
        minimum_test_observations=20,
    )
    baseline = run_repeated_walk_forward(
        prices.iloc[:175],
        {name: table.iloc[:175] for name, table in metrics.items()},
        strategies,
        walk_forward=walk,
        experiment=experiment,
    )
    modified_prices = prices.copy()
    modified_prices.iloc[175:] *= 20.0
    modified_metrics = {name: table.copy() for name, table in metrics.items()}
    modified_metrics["probability"].iloc[175:] = 0.99
    modified = run_repeated_walk_forward(
        modified_prices.iloc[:175],
        {name: table.iloc[:175] for name, table in modified_metrics.items()},
        strategies,
        walk_forward=walk,
        experiment=experiment,
    )
    pd.testing.assert_frame_equal(baseline.fold_table, modified.fold_table)
