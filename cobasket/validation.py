"""Historical policy and probability-calibration validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cobasket.evidence import (
    CalibrationDiagnostics,
    LongOnlyPolicy,
    PolicyBacktestResult,
    calibration_diagnostics,
    run_long_only_policy_backtest,
)


@dataclass(frozen=True)
class ValidationResult:
    """Combined historical policy and calibration validation output.

    Parameters
    ----------
    backtest
        Long-only policy simulation result.
    benchmark_equity
        Equal-weight buy-and-hold benchmark scaled to initial portfolio equity.
    drawdown
        Fractional drawdown of the simulated policy from its prior peak.
    invested_fraction
        Fraction of total equity invested rather than held as cash.
    calibration
        Probability-calibration diagnostics, when outcome records were supplied.
    """

    backtest: PolicyBacktestResult
    benchmark_equity: pd.Series
    drawdown: pd.Series
    invested_fraction: pd.Series
    calibration: CalibrationDiagnostics | None = None


def equal_weight_benchmark(prices: pd.DataFrame, initial_value: float) -> pd.Series:
    """Construct an equal-weight buy-and-hold benchmark.

    Parameters
    ----------
    prices
        Positive aligned price histories.
    initial_value
        Benchmark value at the first observation.

    Returns
    -------
    pandas.Series
        Equal-weight benchmark equity curve.
    """
    clean = prices.astype(float).sort_index().dropna(how="any")
    if clean.empty or (clean <= 0.0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    normalized = clean.divide(clean.iloc[0])
    return (normalized.mean(axis=1) * float(initial_value)).rename("benchmark_equity")


def build_validation_result(
    prices: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    outcomes: pd.DataFrame | None = None,
    policy: LongOnlyPolicy | None = None,
    initial_cash: float = 10_000.0,
    periods_per_year: int = 252,
    n_bins: int = 10,
) -> ValidationResult:
    """Run the policy backtest and optional calibration diagnostics.

    Parameters
    ----------
    prices
        Price observations with dates in rows and tickers in columns.
    probabilities
        Calibrated ticker probabilities indexed by forecast date.
    outcomes
        Optional table containing ``probability_outperform`` and
        ``outperformed`` columns from walk-forward evaluation.
    policy
        Probability-to-position decision policy.
    initial_cash
        Starting portfolio value.
    periods_per_year
        Observations per year for annualized backtest metrics.
    n_bins
        Number of reliability-diagram probability bins.

    Returns
    -------
    ValidationResult
        Curves, trades, metrics, and optional calibration diagnostics.
    """
    backtest = run_long_only_policy_backtest(
        prices,
        probabilities,
        policy=policy,
        initial_cash=initial_cash,
        periods_per_year=periods_per_year,
    )
    benchmark = equal_weight_benchmark(prices.reindex(backtest.equity.index), initial_cash)
    drawdown = (backtest.equity / backtest.equity.cummax() - 1.0).rename("drawdown")
    invested = (1.0 - backtest.cash / backtest.equity).clip(0.0, 1.0).rename(
        "invested_fraction"
    )

    diagnostics = None
    if outcomes is not None:
        required = {"probability_outperform", "outperformed"}
        missing = required.difference(outcomes.columns)
        if missing:
            raise ValueError(f"outcomes is missing columns: {sorted(missing)}")
        diagnostics = calibration_diagnostics(
            outcomes["probability_outperform"],
            outcomes["outperformed"],
            n_bins=n_bins,
        )

    return ValidationResult(backtest, benchmark, drawdown, invested, diagnostics)
