"""Tests for controlled train/validation/test strategy experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobasket.strategy_experiments import (
    ExperimentSplit,
    StrategyExperimentConfig,
    bounded_parameter_grid,
    run_strategy_experiment,
    strategies_from_grid,
)
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


def _strategy(name: str, metric: str, threshold: float) -> StrategyRules:
    """Build a one-rule full-allocation strategy for tests."""
    return StrategyRules(
        name=name,
        rules=(
            StrategyRule(
                action="buy",
                conditions=(MetricCondition(metric, ">=", threshold),),
                target_weight=1.0,
            ),
        ),
        transaction_cost_bps=0.0,
    )


def _experiment_data():
    """Create prices and metrics with different validation/test preferences."""
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 160.0, len(index))}, index=index)
    probability = pd.DataFrame(0.0, index=index, columns=["AAA"])
    momentum = pd.DataFrame(0.0, index=index, columns=["AAA"])
    probability.loc[index[40:80], "AAA"] = 0.8
    momentum.loc[index[80:], "AAA"] = 0.8
    metrics = {"probability": probability, "momentum": momentum}
    split = ExperimentSplit(
        train=(index[0], index[39]),
        validation=(index[40], index[79]),
        test=(index[80], index[-1]),
    )
    return prices, metrics, split


def test_fractional_split_is_ordered_and_disjoint():
    """Fractional splitting should cover the ordered sample without overlap."""
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    split = ExperimentSplit.from_fractions(index, train_fraction=0.5, validation_fraction=0.25)
    assert split.train == (index[0], index[49])
    assert split.validation == (index[50], index[74])
    assert split.test == (index[75], index[-1])


def test_invalid_overlapping_split_is_rejected():
    """An interval used twice would leak information and must be rejected."""
    with pytest.raises(ValueError, match="ordered and disjoint"):
        ExperimentSplit(
            train=("2020-01-01", "2020-06-01"),
            validation=("2020-05-01", "2020-09-01"),
            test=("2020-10-01", "2020-12-01"),
        )


def test_parameter_grid_has_a_hard_combination_limit():
    """Large searches should fail rather than silently encourage overfitting."""
    with pytest.raises(ValueError, match="contains 12 combinations"):
        bounded_parameter_grid(
            {"buy": [0.55, 0.60, 0.65], "momentum": [0.0, 0.1], "trend": [0.0, 0.1]},
            maximum_combinations=10,
        )


def test_strategy_factory_requires_unique_names():
    """Results cannot be compared reliably when names collide."""
    def factory(threshold):
        return _strategy("duplicate", "probability", threshold)

    with pytest.raises(ValueError, match="unique names"):
        strategies_from_grid(factory, {"threshold": [0.5, 0.6]})


def test_validation_selects_strategy_without_consulting_test_results():
    """The test interval must not influence the chosen candidate."""
    prices, metrics, split = _experiment_data()
    probability = _strategy("probability", "probability", 0.6)
    momentum = _strategy("momentum", "momentum", 0.6)
    result = run_strategy_experiment(
        prices,
        metrics,
        (probability, momentum),
        split,
        config=StrategyExperimentConfig(selection_metric="total_return"),
    )
    assert result.selected_strategy.name == "probability"
    assert set(result.test_table.index) == {
        "probability",
        "benchmark_equal_weight",
        "benchmark_cash",
    }
    assert "momentum" not in result.interval_results["test"]


def test_appending_future_data_cannot_change_selection_or_existing_test_result():
    """Data after the declared test end must be ignored by the experiment."""
    prices, metrics, split = _experiment_data()
    strategies = (
        _strategy("probability", "probability", 0.6),
        _strategy("momentum", "momentum", 0.6),
    )
    original = run_strategy_experiment(prices, metrics, strategies, split)

    future_index = pd.date_range(prices.index[-1] + pd.offsets.BDay(), periods=20, freq="B")
    extended_prices = pd.concat([
        prices,
        pd.DataFrame({"AAA": np.linspace(10.0, 500.0, len(future_index))}, index=future_index),
    ])
    extended_metrics = {
        name: pd.concat([table, pd.DataFrame(1.0, index=future_index, columns=table.columns)])
        for name, table in metrics.items()
    }
    extended = run_strategy_experiment(extended_prices, extended_metrics, strategies, split)
    assert extended.selected_strategy.name == original.selected_strategy.name
    pd.testing.assert_frame_equal(extended.test_table, original.test_table)


def test_experiment_exports_reproducible_tables_and_strategy(tmp_path: Path):
    """Saved experiments should contain the split, tables, and selected rules."""
    prices, metrics, split = _experiment_data()
    result = run_strategy_experiment(
        prices,
        metrics,
        (
            _strategy("probability", "probability", 0.6),
            _strategy("momentum", "momentum", 0.6),
        ),
        split,
    )
    output = result.save(tmp_path / "experiment")
    assert (output / "train_results.csv").exists()
    assert (output / "validation_results.csv").exists()
    assert (output / "test_results.csv").exists()
    assert (output / "selected_strategy.json").exists()
    assert (output / "experiment.json").exists()


def test_candidate_limit_is_enforced():
    """An experiment should reject an oversized candidate family."""
    prices, metrics, split = _experiment_data()
    strategies = tuple(
        _strategy(f"strategy_{index}", "probability", 0.5 + index * 0.01)
        for index in range(4)
    )
    with pytest.raises(ValueError, match="limit is 3"):
        run_strategy_experiment(
            prices,
            metrics,
            strategies,
            split,
            config=StrategyExperimentConfig(maximum_candidates=3),
        )
