"""Controlled train/validation/test experiments for declarative strategies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.strategy_rules import (
    RuleBacktestResult,
    StrategyRules,
    run_rule_strategy_backtest,
)


@dataclass(frozen=True)
class ExperimentSplit:
    """Chronological train, validation, and test intervals.

    Parameters
    ----------
    train
        Training interval, inclusive at both ends.
    validation
        Validation interval used to compare candidate strategies.
    test
        Untouched interval used once for final evaluation.
    """

    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]

    def __post_init__(self) -> None:
        """Validate chronological, non-overlapping intervals."""
        intervals = tuple((pd.Timestamp(start), pd.Timestamp(end)) for start, end in (
            self.train,
            self.validation,
            self.test,
        ))
        for start, end in intervals:
            if start > end:
                raise ValueError("split interval start must not follow its end")
        if not intervals[0][1] < intervals[1][0] or not intervals[1][1] < intervals[2][0]:
            raise ValueError("train, validation, and test intervals must be ordered and disjoint")
        object.__setattr__(self, "train", intervals[0])
        object.__setattr__(self, "validation", intervals[1])
        object.__setattr__(self, "test", intervals[2])

    @classmethod
    def from_fractions(
        cls,
        index: pd.Index,
        *,
        train_fraction: float = 0.50,
        validation_fraction: float = 0.25,
    ) -> "ExperimentSplit":
        """Construct chronological intervals from fractional sample sizes.

        Parameters
        ----------
        index
            Ordered observation dates.
        train_fraction
            Fraction assigned to training.
        validation_fraction
            Fraction assigned to validation. The remainder is the test interval.

        Returns
        -------
        ExperimentSplit
            Chronological, non-overlapping intervals.
        """
        dates = pd.DatetimeIndex(index).sort_values().unique()
        if len(dates) < 12:
            raise ValueError("at least 12 observations are required for a three-way split")
        if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
            raise ValueError("split fractions must lie strictly between zero and one")
        if train_fraction + validation_fraction >= 1.0:
            raise ValueError("training and validation fractions must leave a test interval")
        train_end = max(1, int(np.floor(len(dates) * train_fraction)))
        validation_end = max(train_end + 1, int(np.floor(len(dates) * (train_fraction + validation_fraction))))
        if validation_end >= len(dates):
            raise ValueError("split fractions leave no test observations")
        return cls(
            train=(dates[0], dates[train_end - 1]),
            validation=(dates[train_end], dates[validation_end - 1]),
            test=(dates[validation_end], dates[-1]),
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return a JSON-compatible representation."""
        return {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in {
                "train": self.train,
                "validation": self.validation,
                "test": self.test,
            }.items()
        }


@dataclass(frozen=True)
class StrategyExperimentConfig:
    """Settings controlling strategy selection and experiment safeguards.

    Parameters
    ----------
    selection_metric
        Validation metric maximized when choosing the final strategy.
    initial_cash
        Starting capital supplied independently to each interval backtest.
    maximum_candidates
        Maximum number of strategies permitted in one experiment.
    minimum_validation_observations
        Minimum number of price observations in the validation interval.
    minimum_test_observations
        Minimum number of price observations in the final test interval.
    """

    selection_metric: str = "sharpe_ratio"
    initial_cash: float = 10_000.0
    maximum_candidates: int = 25
    minimum_validation_observations: int = 20
    minimum_test_observations: int = 20

    def __post_init__(self) -> None:
        """Validate capital, limits, and selection metric."""
        allowed = {
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "maximum_drawdown",
            "annualized_volatility",
        }
        if self.selection_metric not in allowed:
            raise ValueError(f"unsupported selection metric: {self.selection_metric}")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if self.minimum_validation_observations < 2 or self.minimum_test_observations < 2:
            raise ValueError("minimum interval sizes must be at least two")


@dataclass(frozen=True)
class StrategyExperimentResult:
    """Outputs from one controlled strategy-selection experiment.

    Parameters
    ----------
    split
        Chronological intervals used by the experiment.
    validation_table
        Candidate performance on the validation interval.
    test_table
        Selected-strategy and benchmark performance on the test interval.
    selected_strategy
        Strategy chosen using validation data only.
    interval_results
        Detailed candidate results keyed by interval and strategy name.
    warnings
        Overfitting and data-quality warnings.
    """

    split: ExperimentSplit
    validation_table: pd.DataFrame
    test_table: pd.DataFrame
    selected_strategy: StrategyRules
    interval_results: Mapping[str, Mapping[str, RuleBacktestResult]]
    warnings: tuple[str, ...]

    def save(self, directory: str | Path) -> Path:
        """Write experiment tables and metadata to a directory.

        Parameters
        ----------
        directory
            Output directory.

        Returns
        -------
        pathlib.Path
            Created output directory.
        """
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.validation_table.to_csv(output / "validation_results.csv")
        self.test_table.to_csv(output / "test_results.csv")
        self.selected_strategy.save(output / "selected_strategy.json")
        metadata = {
            "split": self.split.to_dict(),
            "selected_strategy": self.selected_strategy.name,
            "warnings": list(self.warnings),
        }
        (output / "experiment.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


def bounded_parameter_grid(
    parameters: Mapping[str, Sequence[object]],
    *,
    maximum_combinations: int = 25,
) -> tuple[dict[str, object], ...]:
    """Expand a small parameter grid while enforcing a hard search limit.

    Parameters
    ----------
    parameters
        Mapping from parameter name to candidate values.
    maximum_combinations
        Hard limit on generated combinations.

    Returns
    -------
    tuple of dict
        Cartesian-product parameter combinations.
    """
    if maximum_combinations < 1:
        raise ValueError("maximum_combinations must be positive")
    if not parameters:
        return ({},)
    names = tuple(parameters)
    values = tuple(tuple(parameters[name]) for name in names)
    if any(len(items) == 0 for items in values):
        raise ValueError("every grid parameter must provide at least one value")
    count = int(np.prod([len(items) for items in values]))
    if count > maximum_combinations:
        raise ValueError(
            f"parameter grid contains {count} combinations; limit is {maximum_combinations}"
        )
    return tuple(dict(zip(names, items)) for items in product(*values))


def strategies_from_grid(
    factory: Callable[..., StrategyRules],
    parameters: Mapping[str, Sequence[object]],
    *,
    maximum_combinations: int = 25,
) -> tuple[StrategyRules, ...]:
    """Build uniquely named strategies from a bounded parameter grid."""
    strategies = tuple(
        factory(**combination)
        for combination in bounded_parameter_grid(
            parameters,
            maximum_combinations=maximum_combinations,
        )
    )
    if len({strategy.name for strategy in strategies}) != len(strategies):
        raise ValueError("strategy factory must produce unique names")
    return strategies


def _slice_tables(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    interval: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Slice prices and metrics to one inclusive interval without filling data."""
    start, end = interval
    sliced_prices = prices.loc[start:end]
    sliced_metrics = {name: table.loc[start:end] for name, table in metrics.items()}
    return sliced_prices, sliced_metrics


def _summary_row(result: RuleBacktestResult) -> dict[str, float]:
    """Return standard metrics plus explicit profit and ending value."""
    metrics = dict(result.backtest.metrics)
    equity = result.backtest.equity
    metrics["ending_value"] = float(equity.iloc[-1])
    metrics["profit"] = float(equity.iloc[-1] - equity.iloc[0])
    return metrics


def _equal_weight_benchmark(prices: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Construct an equal-weight buy-and-hold benchmark."""
    units = (initial_cash / prices.shape[1]) / prices.iloc[0]
    return (prices * units).sum(axis=1).rename("equal_weight")


def _benchmark_rows(prices: pd.DataFrame, initial_cash: float) -> list[dict[str, object]]:
    """Build cash and equal-weight benchmark summary rows."""
    from cobasket.evidence.policy_backtest import _performance_metrics

    equal_weight = _equal_weight_benchmark(prices, initial_cash)
    cash = pd.Series(initial_cash, index=prices.index, name="cash")
    rows: list[dict[str, object]] = []
    for name, equity in (("benchmark_equal_weight", equal_weight), ("benchmark_cash", cash)):
        metrics = _performance_metrics(equity, 252)
        rows.append({
            "strategy": name,
            **metrics,
            "ending_value": float(equity.iloc[-1]),
            "profit": float(equity.iloc[-1] - equity.iloc[0]),
            "transaction_costs": 0.0,
            "trade_count": 0.0,
        })
    return rows


def run_strategy_experiment(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    split: ExperimentSplit,
    *,
    config: StrategyExperimentConfig | None = None,
) -> StrategyExperimentResult:
    """Select a strategy on validation data and evaluate it once on test data.

    Candidate rankings use the validation interval only. Test results are not
    consulted when choosing the selected strategy.
    """
    config = config or StrategyExperimentConfig()
    clean = prices.astype(float).dropna(how="any").sort_index()
    if clean.empty or clean.shape[1] < 1 or (clean <= 0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    if not strategies:
        raise ValueError("at least one strategy is required")
    if len(strategies) > config.maximum_candidates:
        raise ValueError(
            f"experiment has {len(strategies)} candidates; limit is {config.maximum_candidates}"
        )
    if len({strategy.name for strategy in strategies}) != len(strategies):
        raise ValueError("strategy names must be unique")

    validation_prices, validation_metrics = _slice_tables(clean, metrics, split.validation)
    test_prices, test_metrics = _slice_tables(clean, metrics, split.test)
    if len(validation_prices) < config.minimum_validation_observations:
        raise ValueError("validation interval is too short")
    if len(test_prices) < config.minimum_test_observations:
        raise ValueError("test interval is too short")

    validation_results: dict[str, RuleBacktestResult] = {}
    validation_rows: list[dict[str, object]] = []
    for strategy in strategies:
        result = run_rule_strategy_backtest(
            validation_prices,
            validation_metrics,
            strategy,
            initial_cash=config.initial_cash,
        )
        validation_results[strategy.name] = result
        validation_rows.append({"strategy": strategy.name, **_summary_row(result)})
    validation_table = pd.DataFrame(validation_rows).set_index("strategy")

    scores = validation_table[config.selection_metric].copy()
    if config.selection_metric in {"maximum_drawdown", "annualized_volatility"}:
        selected_name = str(scores.idxmax() if config.selection_metric == "maximum_drawdown" else scores.idxmin())
    else:
        selected_name = str(scores.idxmax())
    selected_strategy = next(item for item in strategies if item.name == selected_name)

    selected_test = run_rule_strategy_backtest(
        test_prices,
        test_metrics,
        selected_strategy,
        initial_cash=config.initial_cash,
    )
    test_rows = [{"strategy": selected_name, **_summary_row(selected_test)}]
    test_rows.extend(_benchmark_rows(test_prices, config.initial_cash))
    test_table = pd.DataFrame(test_rows).set_index("strategy")

    warnings: list[str] = []
    if len(strategies) > 10:
        warnings.append(
            "More than 10 candidate strategies were compared; the best validation result may reflect multiple-testing noise."
        )
    validation_length = len(validation_prices)
    if len(strategies) > max(3, validation_length // 20):
        warnings.append(
            "The number of candidate strategies is large relative to the validation sample."
        )
    if selected_test.backtest.metrics["trade_count"] == 0:
        warnings.append("The selected strategy made no trades in the test interval.")

    return StrategyExperimentResult(
        split=split,
        validation_table=validation_table.sort_values(
            config.selection_metric,
            ascending=config.selection_metric in {"annualized_volatility"},
        ),
        test_table=test_table,
        selected_strategy=selected_strategy,
        interval_results={
            "validation": validation_results,
            "test": {selected_name: selected_test},
        },
        warnings=tuple(warnings),
    )
