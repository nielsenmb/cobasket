"""Continuous walk-forward deployment with persistent cash and holdings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.evidence.policy_backtest import _performance_metrics
from cobasket.repeated_walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
)
from cobasket.strategy_experiments import StrategyExperimentConfig, run_strategy_experiment
from cobasket.strategy_rules import StrategyRules


@dataclass(frozen=True)
class ContinuousDeploymentConfig:
    """Execution settings for a continuously managed walk-forward account.

    Parameters
    ----------
    initial_cash
        Capital available at the start of the first test interval.
    boundary_policy
        Behaviour when a newly selected strategy differs from the preceding one.
        ``"retain"`` keeps current holdings until the new rules alter them.
        ``"liquidate"`` sells all holdings at the first observation of the new test
        interval before applying the newly selected strategy.
    periods_per_year
        Sampling frequency used for annualized performance statistics.
    """

    initial_cash: float = 10_000.0
    boundary_policy: str = "retain"
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        """Validate capital and boundary behaviour."""
        if self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be positive")
        if self.boundary_policy not in {"retain", "liquidate"}:
            raise ValueError("boundary_policy must be 'retain' or 'liquidate'")
        if self.periods_per_year < 2:
            raise ValueError("periods_per_year must be at least two")


@dataclass(frozen=True)
class ContinuousWalkForwardResult:
    """Outputs from one continuously managed walk-forward simulation.

    Parameters
    ----------
    equity
        Continuous strategy, equal-weight, and cash benchmark values.
    cash
        Strategy cash balance through time.
    positions
        Share quantities held through time.
    weights
        End-of-observation asset weights.
    trades
        Executed trades, including boundary liquidations.
    decisions
        Rule decisions made during each test interval.
    selections
        Fold boundaries and the validation-selected strategy for each fold.
    metrics
        Summary performance statistics for the strategy and benchmarks.
    warnings
        Interpretation and data-quality warnings.
    """

    equity: pd.DataFrame
    cash: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    warnings: tuple[str, ...]

    def save(self, directory: str | Path) -> Path:
        """Write continuous deployment outputs to a directory."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(output / "continuous_equity.csv")
        self.cash.to_csv(output / "continuous_cash.csv")
        self.positions.to_csv(output / "continuous_positions.csv")
        self.weights.to_csv(output / "continuous_weights.csv")
        self.trades.to_csv(output / "continuous_trades.csv", index=False)
        self.decisions.to_csv(output / "continuous_decisions.csv", index=False)
        self.selections.to_csv(output / "continuous_selections.csv", index=False)
        self.metrics.to_csv(output / "continuous_metrics.csv")
        (output / "continuous_walk_forward.json").write_text(
            json.dumps({"warnings": list(self.warnings)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


def _prepare_metrics(
    metrics: Mapping[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Align metric tables without filling missing or future values."""
    if not metrics:
        raise ValueError("at least one metric table is required")
    prepared: dict[str, pd.DataFrame] = {}
    for name, table in metrics.items():
        if not name.strip():
            raise ValueError("metric names must not be empty")
        prepared[name] = table.astype(float).sort_index().reindex(
            index=index,
            columns=tickers,
        )
    return prepared


def _select_strategies(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    walk_forward: WalkForwardConfig,
    experiment: StrategyExperimentConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    """Select one strategy per fold using validation data only."""
    selections: list[dict[str, object]] = []
    warnings: list[str] = []
    for number, split in enumerate(
        generate_walk_forward_splits(prices.index, config=walk_forward),
        start=1,
    ):
        result = run_strategy_experiment(
            prices,
            metrics,
            strategies,
            split,
            config=experiment,
        )
        selections.append(
            {
                "fold": number,
                "train_start": split.train[0],
                "train_end": split.train[1],
                "validation_start": split.validation[0],
                "validation_end": split.validation[1],
                "test_start": split.test[0],
                "test_end": split.test[1],
                "selected_strategy": result.selected_strategy.name,
                "strategy": result.selected_strategy,
            }
        )
        warnings.extend(f"Fold {number}: {item}" for item in result.warnings)
    return selections, warnings


def _summary_metrics(equity: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Return standard metrics plus ending value and profit."""
    metrics = _performance_metrics(equity, periods_per_year)
    metrics["ending_value"] = float(equity.iloc[-1])
    metrics["profit"] = float(equity.iloc[-1] - equity.iloc[0])
    return metrics


def run_continuous_walk_forward(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    *,
    walk_forward: WalkForwardConfig | None = None,
    experiment: StrategyExperimentConfig | None = None,
    deployment: ContinuousDeploymentConfig | None = None,
) -> ContinuousWalkForwardResult:
    """Simulate one account while strategies are reselected through time.

    Each fold selects a strategy from its validation interval. The selected strategy
    then controls the same account during that fold's test interval. Cash and holdings
    are carried into later folds and through any gaps between test intervals.

    Decisions use metric values available on the current observation and execute at
    the next available price observation. Strategy selection never consults the fold's
    test interval.

    Parameters
    ----------
    prices
        Positive aligned prices with dates in rows and tickers in columns.
    metrics
        Named ticker-by-date metric tables used by the declarative rules.
    strategies
        Candidate strategies considered independently in each fold.
    walk_forward
        Fold lengths and overlap safeguards.
    experiment
        Validation selection metric and candidate safeguards.
    deployment
        Starting capital and fold-boundary behaviour.

    Returns
    -------
    ContinuousWalkForwardResult
        Continuous portfolio state, selections, trades, and performance.
    """
    walk_forward = walk_forward or WalkForwardConfig()
    deployment = deployment or ContinuousDeploymentConfig()
    experiment = experiment or StrategyExperimentConfig(
        initial_cash=deployment.initial_cash,
    )
    if experiment.initial_cash != deployment.initial_cash:
        raise ValueError("experiment and deployment initial_cash must match")
    if not strategies:
        raise ValueError("at least one candidate strategy is required")
    if len({item.name for item in strategies}) != len(strategies):
        raise ValueError("strategy names must be unique")

    clean = prices.astype(float).dropna(how="any").sort_index()
    if clean.empty or clean.shape[1] < 1 or (clean <= 0.0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    tickers = list(clean.columns)
    prepared = _prepare_metrics(metrics, clean.index, tickers)
    selections, warnings = _select_strategies(
        clean,
        prepared,
        strategies,
        walk_forward,
        experiment,
    )
    strategy_lookup = {item.name: item for item in strategies}
    first_test = pd.Timestamp(selections[0]["test_start"])
    last_test = pd.Timestamp(selections[-1]["test_end"])
    simulation_prices = clean.loc[first_test:last_test]

    active_by_date: dict[pd.Timestamp, dict[str, object]] = {}
    boundary_by_date: dict[pd.Timestamp, dict[str, object]] = {}
    for selection in selections:
        start = pd.Timestamp(selection["test_start"])
        end = pd.Timestamp(selection["test_end"])
        for date in simulation_prices.loc[start:end].index:
            if date in active_by_date:
                raise ValueError("continuous deployment requires non-overlapping test intervals")
            active_by_date[pd.Timestamp(date)] = selection
        boundary_by_date[start] = selection

    shares = pd.Series(0.0, index=tickers)
    cash_value = float(deployment.initial_cash)
    pending: pd.DataFrame | None = None
    pending_strategy: StrategyRules | None = None
    current_strategy_name: str | None = None
    equity_rows: list[float] = []
    cash_rows: list[float] = []
    position_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for date, price in simulation_prices.iterrows():
        date = pd.Timestamp(date)
        boundary = boundary_by_date.get(date)
        new_name = None if boundary is None else str(boundary["selected_strategy"])
        strategy_changed = new_name is not None and new_name != current_strategy_name

        if strategy_changed and deployment.boundary_policy == "liquidate" and shares.sum() > 0.0:
            strategy_for_cost = strategy_lookup[new_name]
            for ticker in tickers:
                quantity = float(shares.loc[ticker])
                if quantity <= 0.0:
                    continue
                notional = quantity * float(price.loc[ticker])
                cost = notional * strategy_for_cost.transaction_cost_bps / 10_000.0
                cash_value += notional - cost
                shares.loc[ticker] = 0.0
                trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": "sell",
                        "quantity": quantity,
                        "price": float(price.loc[ticker]),
                        "notional": notional,
                        "cost": cost,
                        "action": "boundary_liquidation",
                        "strategy": new_name,
                        "fold": int(boundary["fold"]),
                    }
                )
            pending = None
            pending_strategy = None

        if new_name is not None:
            current_strategy_name = new_name

        if pending is not None and pending_strategy is not None:
            portfolio_before = cash_value + float((shares * price).sum())
            current_values = shares * price
            desired = pending["target_weight"].astype(float).copy()
            if float(desired.sum()) > 1.0:
                desired /= float(desired.sum())
            trade_values = desired * portfolio_before - current_values
            for ticker in sorted(tickers, key=lambda item: trade_values.loc[item]):
                trade_value = float(trade_values.loc[ticker])
                if abs(trade_value) < pending_strategy.minimum_trade_value:
                    continue
                if trade_value > 0.0:
                    denominator = 1.0 + pending_strategy.transaction_cost_bps / 10_000.0
                    trade_value = min(trade_value, cash_value / denominator)
                quantity = trade_value / float(price.loc[ticker])
                cost = abs(trade_value) * pending_strategy.transaction_cost_bps / 10_000.0
                shares.loc[ticker] += quantity
                cash_value -= trade_value + cost
                trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": "buy" if trade_value > 0.0 else "sell",
                        "quantity": abs(quantity),
                        "price": float(price.loc[ticker]),
                        "notional": abs(trade_value),
                        "cost": cost,
                        "action": str(pending.loc[ticker, "action"]),
                        "strategy": pending_strategy.name,
                        "fold": int(pending.loc[ticker, "fold"]),
                    }
                )
            pending = None
            pending_strategy = None

        equity_value = cash_value + float((shares * price).sum())
        current_weights = shares * price / equity_value
        equity_rows.append(equity_value)
        cash_rows.append(cash_value)
        position_rows.append(shares.copy())
        weight_rows.append(current_weights.copy())

        active = active_by_date.get(date)
        if active is None:
            continue
        strategy = strategy_lookup[str(active["selected_strategy"])]
        rows: list[dict[str, object]] = []
        for ticker in tickers:
            context: dict[str, float | bool] = {
                name: float(frame.loc[date, ticker])
                for name, frame in prepared.items()
                if pd.notna(frame.loc[date, ticker])
            }
            context["current_weight"] = float(current_weights.loc[ticker])
            context["is_held"] = bool(current_weights.loc[ticker] > 0.0)
            action, target = strategy.decide(context, float(current_weights.loc[ticker]))
            rows.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "target_weight": target,
                    "fold": int(active["fold"]),
                }
            )
            decisions.append(
                {
                    "date": date,
                    "fold": int(active["fold"]),
                    "strategy": strategy.name,
                    "ticker": ticker,
                    "action": action,
                    "target_weight": target,
                    "current_weight": float(current_weights.loc[ticker]),
                    **context,
                }
            )
        pending = pd.DataFrame(rows).set_index("ticker")
        pending_strategy = strategy

    index = simulation_prices.index
    strategy_equity = pd.Series(equity_rows, index=index, name="strategy")
    cash_series = pd.Series(cash_rows, index=index, name="cash")
    positions = pd.DataFrame(position_rows, index=index)
    weights = pd.DataFrame(weight_rows, index=index)

    benchmark_units = (deployment.initial_cash / len(tickers)) / simulation_prices.iloc[0]
    benchmark = (simulation_prices * benchmark_units).sum(axis=1).rename("equal_weight")
    cash_benchmark = pd.Series(deployment.initial_cash, index=index, name="cash_benchmark")
    equity = pd.concat([strategy_equity, benchmark, cash_benchmark], axis=1)

    trade_table = pd.DataFrame(
        trades,
        columns=[
            "date", "ticker", "side", "quantity", "price", "notional",
            "cost", "action", "strategy", "fold",
        ],
    )
    decision_table = pd.DataFrame(decisions)
    selection_table = pd.DataFrame(selections).drop(columns="strategy")
    metric_rows = {
        "strategy": _summary_metrics(strategy_equity, deployment.periods_per_year),
        "equal_weight": _summary_metrics(benchmark, deployment.periods_per_year),
        "cash": _summary_metrics(cash_benchmark, deployment.periods_per_year),
    }
    metric_table = pd.DataFrame(metric_rows).T
    metric_table["transaction_costs"] = 0.0
    metric_table["trade_count"] = 0.0
    metric_table.loc["strategy", "transaction_costs"] = (
        float(trade_table["cost"].sum()) if not trade_table.empty else 0.0
    )
    metric_table.loc["strategy", "trade_count"] = float(len(trade_table))

    selected_names = selection_table["selected_strategy"]
    if selected_names.nunique() > 1:
        warnings.append(
            "The selected strategy changes across folds; continuous performance includes model-reselection turnover."
        )
    if deployment.boundary_policy == "liquidate" and selected_names.nunique() > 1:
        warnings.append(
            "Holdings are liquidated when the selected strategy changes, increasing turnover and transaction costs."
        )
    if strategy_equity.iloc[-1] <= benchmark.iloc[-1]:
        warnings.append(
            "The continuous deployment did not finish above equal-weight buy-and-hold."
        )

    return ContinuousWalkForwardResult(
        equity=equity,
        cash=cash_series,
        positions=positions,
        weights=weights,
        trades=trade_table,
        decisions=decision_table,
        selections=selection_table,
        metrics=metric_table,
        warnings=tuple(dict.fromkeys(warnings)),
    )
