"""End-to-end walk-forward simulation of a long-only basket strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cobasket.evidence import (
    LongOnlyPolicy,
    PolicyBacktestResult,
    fit_probability_calibration,
    run_long_only_policy_backtest,
    walk_forward_evidence,
)


@dataclass(frozen=True)
class BasketStrategyConfig:
    """Parameters controlling walk-forward evidence and portfolio decisions.

    Parameters
    ----------
    train_window
        Number of preceding price observations used to refit the basket.
    z_window
        Number of observations used to standardize the fitted spread.
    horizon
        Future observations used to define calibration outcomes.
    step
        Observations between consecutive strategy evaluations.
    min_trace_ratio
        Minimum Johansen trace ratio accepted as evidence of cointegration.
    min_calibration_samples
        Mature historical outcomes required before probabilities can depart
        from the neutral value of 0.5.
    initial_cash
        Starting capital for the simulated portfolio.
    policy
        Rules converting calibrated probabilities into target allocations.
    """

    train_window: int = 252
    z_window: int = 60
    horizon: int = 20
    step: int = 5
    min_trace_ratio: float = 1.0
    min_calibration_samples: int = 30
    initial_cash: float = 10_000.0
    policy: LongOnlyPolicy = LongOnlyPolicy()

    def __post_init__(self) -> None:
        """Validate window, calibration, and capital settings."""
        if self.train_window < self.z_window:
            raise ValueError("train_window must be at least z_window")
        if self.horizon < 1 or self.step < 1:
            raise ValueError("horizon and step must be positive")
        if self.min_trace_ratio <= 0:
            raise ValueError("min_trace_ratio must be positive")
        if self.min_calibration_samples < 1:
            raise ValueError("min_calibration_samples must be positive")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")


@dataclass(frozen=True)
class BasketStrategyResult:
    """Outputs from one complete basket strategy simulation.

    Parameters
    ----------
    records
        Walk-forward evidence and realized future outcomes.
    probabilities
        Probabilities available at each historical decision date.
    backtest
        Long-only policy simulation driven by ``probabilities``.
    benchmark_equity
        Equal-weight buy-and-hold equity using the same starting capital.
    summary
        Compact strategy and benchmark performance statistics.
    """

    records: pd.DataFrame
    probabilities: pd.DataFrame
    backtest: PolicyBacktestResult
    benchmark_equity: pd.Series
    summary: dict[str, float]


def expanding_calibrated_probabilities(
    records: pd.DataFrame,
    *,
    horizon: int,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Convert walk-forward evidence into leakage-free historical probabilities.

    At each evaluation date, calibration uses only records whose outcome date is
    on or before that date. Evidence with insufficient mature history receives
    the neutral probability 0.5.

    Parameters
    ----------
    records
        Output from :func:`cobasket.evidence.walk_forward_evidence`.
    horizon
        Forecast horizon represented by the records.
    min_samples
        Minimum mature outcomes required before fitting calibration.

    Returns
    -------
    pandas.DataFrame
        Evaluation dates in rows, tickers in columns, and probabilities in
        ``[0, 1]``.
    """
    required = {"evaluation_date", "future_date", "ticker", "score", "outperformed"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"records are missing required columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("records must not be empty")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    table = records.copy()
    table["evaluation_date"] = pd.to_datetime(table["evaluation_date"])
    table["future_date"] = pd.to_datetime(table["future_date"])
    tickers = sorted(table["ticker"].astype(str).unique())
    rows: list[pd.Series] = []

    for date, current in table.groupby("evaluation_date", sort=True):
        mature = table.loc[table["future_date"] <= date]
        probabilities = pd.Series(0.5, index=tickers, name=date, dtype=float)
        if len(mature) >= min_samples:
            calibration = fit_probability_calibration(mature, horizon=horizon)
            for item in current.itertuples(index=False):
                probabilities.loc[str(item.ticker)] = float(
                    calibration.lookup(float(item.score))["probability_mean"]
                )
        rows.append(probabilities)

    return pd.DataFrame(rows).sort_index()


def equal_weight_benchmark(prices: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Construct an equal-weight buy-and-hold benchmark equity curve.

    Parameters
    ----------
    prices
        Positive aligned prices with assets in columns.
    initial_cash
        Starting capital assigned equally across assets.

    Returns
    -------
    pandas.Series
        Benchmark portfolio value through time.
    """
    clean = prices.astype(float).dropna(how="any").sort_index()
    if clean.empty or (clean <= 0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    units = (initial_cash / clean.shape[1]) / clean.iloc[0]
    return (clean * units).sum(axis=1).rename("equal_weight_benchmark")


def run_basket_strategy_simulation(
    prices: pd.DataFrame,
    *,
    config: BasketStrategyConfig | None = None,
) -> BasketStrategyResult:
    """Generate historical evidence, calibrate it, and simulate the strategy.

    Parameters
    ----------
    prices
        Aligned adjusted prices for one candidate basket.
    config
        Walk-forward and long-only policy settings.

    Returns
    -------
    BasketStrategyResult
        Forecast records, probabilities, portfolio history, benchmark, and
        summary statistics.
    """
    config = config or BasketStrategyConfig()
    clean = prices.astype(float).dropna(how="any").sort_index()
    if clean.shape[1] < 2:
        raise ValueError("a basket strategy requires at least two assets")

    records = walk_forward_evidence(
        clean,
        train_window=config.train_window,
        z_window=config.z_window,
        horizon=config.horizon,
        step=config.step,
        min_trace_ratio=config.min_trace_ratio,
    )
    if records.empty:
        raise ValueError("no valid walk-forward basket evaluations were produced")
    probabilities = expanding_calibrated_probabilities(
        records,
        horizon=config.horizon,
        min_samples=config.min_calibration_samples,
    )
    backtest = run_long_only_policy_backtest(
        clean,
        probabilities,
        policy=config.policy,
        initial_cash=config.initial_cash,
    )
    benchmark = equal_weight_benchmark(clean, config.initial_cash)
    strategy_profit = float(backtest.equity.iloc[-1] - config.initial_cash)
    benchmark_profit = float(benchmark.iloc[-1] - config.initial_cash)
    summary = dict(backtest.metrics)
    summary.update(
        {
            "starting_value": float(config.initial_cash),
            "ending_value": float(backtest.equity.iloc[-1]),
            "profit": strategy_profit,
            "benchmark_ending_value": float(benchmark.iloc[-1]),
            "benchmark_profit": benchmark_profit,
            "excess_profit": strategy_profit - benchmark_profit,
            "evaluation_count": float(probabilities.shape[0]),
        }
    )
    return BasketStrategyResult(records, probabilities, backtest, benchmark, summary)
