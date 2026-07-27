"""Repeated walk-forward evaluation for declarative trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.strategy_experiments import (
    ExperimentSplit,
    StrategyExperimentConfig,
    StrategyExperimentResult,
    run_strategy_experiment,
)
from cobasket.strategy_rules import StrategyRules


@dataclass(frozen=True)
class WalkForwardConfig:
    """Window lengths and safeguards for repeated strategy evaluation.

    Parameters
    ----------
    train_observations
        Observations in each training interval.
    validation_observations
        Observations used to select one candidate strategy per fold.
    test_observations
        Untouched observations used to score the selected strategy.
    step_observations
        Number of observations between successive fold starts. By default this
        should be at least ``test_observations`` to avoid overlapping test data.
    expanding_train
        Expand the training start back to the first observation while moving
        validation and test intervals forward.
    allow_overlapping_tests
        Permit test intervals to overlap. This is disabled by default because
        overlapping folds are correlated and can exaggerate effective sample size.
    """

    train_observations: int = 504
    validation_observations: int = 126
    test_observations: int = 126
    step_observations: int = 126
    expanding_train: bool = False
    allow_overlapping_tests: bool = False

    def __post_init__(self) -> None:
        """Validate fold lengths and overlap assumptions."""
        sizes = (
            self.train_observations,
            self.validation_observations,
            self.test_observations,
            self.step_observations,
        )
        if min(sizes) < 2:
            raise ValueError("all walk-forward window sizes must be at least two")
        if not self.allow_overlapping_tests and self.step_observations < self.test_observations:
            raise ValueError(
                "step_observations must be at least test_observations unless overlapping tests are enabled"
            )


@dataclass(frozen=True)
class WalkForwardFold:
    """One chronological strategy-selection and evaluation fold."""

    number: int
    split: ExperimentSplit
    result: StrategyExperimentResult


@dataclass(frozen=True)
class RepeatedWalkForwardResult:
    """Aggregated results from repeated chronological experiments."""

    folds: tuple[WalkForwardFold, ...]
    fold_table: pd.DataFrame
    selection_frequency: pd.DataFrame
    compounded_equity: pd.DataFrame
    warnings: tuple[str, ...]

    def save(self, directory: str | Path) -> Path:
        """Write aggregate tables, fold metadata, and selected strategies."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.fold_table.to_csv(output / "walk_forward_folds.csv", index=False)
        self.selection_frequency.to_csv(output / "selection_frequency.csv")
        self.compounded_equity.to_csv(output / "compounded_equity.csv")
        fold_metadata = []
        for fold in self.folds:
            strategy_path = output / f"fold_{fold.number:03d}_strategy.json"
            fold.result.selected_strategy.save(strategy_path)
            fold_metadata.append(
                {
                    "fold": fold.number,
                    "split": fold.split.to_dict(),
                    "selected_strategy": fold.result.selected_strategy.name,
                    "warnings": list(fold.result.warnings),
                }
            )
        payload = {"folds": fold_metadata, "warnings": list(self.warnings)}
        (output / "walk_forward.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return output


def generate_walk_forward_splits(
    index: pd.Index,
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[ExperimentSplit, ...]:
    """Generate chronological train/validation/test folds from an index."""
    config = config or WalkForwardConfig()
    dates = pd.DatetimeIndex(index).sort_values().unique()
    required = (
        config.train_observations
        + config.validation_observations
        + config.test_observations
    )
    if len(dates) < required:
        raise ValueError(
            f"at least {required} observations are required; received {len(dates)}"
        )

    splits: list[ExperimentSplit] = []
    test_end = required
    while test_end <= len(dates):
        test_start = test_end - config.test_observations
        validation_start = test_start - config.validation_observations
        train_start = 0 if config.expanding_train else validation_start - config.train_observations
        if train_start < 0:
            break
        splits.append(
            ExperimentSplit(
                train=(dates[train_start], dates[validation_start - 1]),
                validation=(dates[validation_start], dates[test_start - 1]),
                test=(dates[test_start], dates[test_end - 1]),
            )
        )
        test_end += config.step_observations
    if not splits:
        raise ValueError("no complete walk-forward folds could be generated")
    return tuple(splits)


def _fold_row(fold: WalkForwardFold) -> dict[str, object]:
    """Convert one completed fold into an aggregate summary row."""
    selected = fold.result.selected_strategy.name
    test = fold.result.test_table
    strategy = test.loc[selected]
    benchmark = test.loc["benchmark_equal_weight"]
    cash = test.loc["benchmark_cash"]
    return {
        "fold": fold.number,
        "train_start": fold.split.train[0],
        "train_end": fold.split.train[1],
        "validation_start": fold.split.validation[0],
        "validation_end": fold.split.validation[1],
        "test_start": fold.split.test[0],
        "test_end": fold.split.test[1],
        "selected_strategy": selected,
        "test_total_return": float(strategy["total_return"]),
        "test_annualized_return": float(strategy["annualized_return"]),
        "test_sharpe_ratio": float(strategy["sharpe_ratio"]),
        "test_maximum_drawdown": float(strategy["maximum_drawdown"]),
        "test_trade_count": float(strategy["trade_count"]),
        "benchmark_total_return": float(benchmark["total_return"]),
        "cash_total_return": float(cash["total_return"]),
        "excess_return": float(strategy["total_return"] - benchmark["total_return"]),
    }


def _compounded_curves(table: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Compound non-overlapping fold returns into comparable equity curves."""
    curves = pd.DataFrame(index=pd.RangeIndex(len(table) + 1, name="fold_boundary"))
    curves.loc[0, "strategy"] = initial_cash
    curves.loc[0, "equal_weight"] = initial_cash
    curves.loc[0, "cash"] = initial_cash
    for i, row in table.iterrows():
        curves.loc[i + 1, "strategy"] = curves.loc[i, "strategy"] * (
            1.0 + row["test_total_return"]
        )
        curves.loc[i + 1, "equal_weight"] = curves.loc[i, "equal_weight"] * (
            1.0 + row["benchmark_total_return"]
        )
        curves.loc[i + 1, "cash"] = curves.loc[i, "cash"] * (
            1.0 + row["cash_total_return"]
        )
    return curves.astype(float)


def run_repeated_walk_forward(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    *,
    walk_forward: WalkForwardConfig | None = None,
    experiment: StrategyExperimentConfig | None = None,
) -> RepeatedWalkForwardResult:
    """Select and test strategies repeatedly across successive market regimes.

    Each fold is a complete controlled experiment. Candidate ranking uses only
    that fold's validation interval, and its test interval is not consulted until
    after one strategy has been selected.
    """
    walk_forward = walk_forward or WalkForwardConfig()
    experiment = experiment or StrategyExperimentConfig()
    clean = prices.astype(float).dropna(how="any").sort_index()
    splits = generate_walk_forward_splits(clean.index, config=walk_forward)
    folds: list[WalkForwardFold] = []
    warnings: list[str] = []
    for number, split in enumerate(splits, start=1):
        result = run_strategy_experiment(
            clean,
            metrics,
            strategies,
            split,
            config=experiment,
        )
        folds.append(WalkForwardFold(number=number, split=split, result=result))
        warnings.extend(f"Fold {number}: {item}" for item in result.warnings)

    table = pd.DataFrame(_fold_row(fold) for fold in folds)
    counts = table["selected_strategy"].value_counts().rename("selected_folds")
    frequency = counts.to_frame()
    frequency["selection_fraction"] = frequency["selected_folds"] / len(table)
    strategy_names = {strategy.name for strategy in strategies}
    missing = sorted(strategy_names.difference(frequency.index))
    for name in missing:
        frequency.loc[name] = [0, 0.0]
    frequency = frequency.sort_values("selected_folds", ascending=False)

    if len(frequency[frequency["selected_folds"] > 0]) > 1:
        warnings.append(
            "Different strategies were selected across folds; preferred rules are regime-dependent."
        )
    positive_excess = float((table["excess_return"] > 0.0).mean())
    if positive_excess < 0.5:
        warnings.append(
            "The selected strategy beat equal-weight buy-and-hold in fewer than half of test folds."
        )
    if walk_forward.allow_overlapping_tests:
        warnings.append(
            "Test intervals overlap, so fold results are correlated and must not be treated as independent."
        )

    compounded = _compounded_curves(table, experiment.initial_cash)
    return RepeatedWalkForwardResult(
        folds=tuple(folds),
        fold_table=table,
        selection_frequency=frequency,
        compounded_equity=compounded,
        warnings=tuple(dict.fromkeys(warnings)),
    )
