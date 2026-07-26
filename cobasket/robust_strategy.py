"""Robustness-aware filtering for basket selection and historical strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from cobasket.evidence import (
    BasketCandidate,
    run_long_only_policy_backtest,
    walk_forward_evidence,
)
from cobasket.robustness import rolling_basket_robustness
from cobasket.strategy_simulation import (
    BasketStrategyConfig,
    BasketStrategyResult,
    equal_weight_benchmark,
    expanding_calibrated_probabilities,
)


@dataclass(frozen=True)
class RobustnessGateConfig:
    """Thresholds controlling whether a basket may trigger decisions.

    Parameters
    ----------
    enabled
        Apply the robustness gate when ``True``.
    window
        Number of preceding observations in each robustness fit.
    step
        Separation between rolling robustness fits.
    min_trace_ratio
        Minimum acceptable Johansen trace-statistic ratio.
    max_half_life
        Maximum acceptable spread mean-reversion half-life in observations.
    max_weight_drift
        Maximum acceptable L1 drift between consecutive weight vectors.
    minimum_stable_fraction
        Minimum fraction of successful historical windows classified as stable.
    """

    enabled: bool = True
    window: int = 126
    step: int = 21
    min_trace_ratio: float = 1.0
    max_half_life: float = 120.0
    max_weight_drift: float = 0.50
    minimum_stable_fraction: float = 0.60

    def __post_init__(self) -> None:
        """Validate robustness-gate settings."""
        if self.window < 20 or self.step < 1:
            raise ValueError("robustness window must be at least 20 and step must be positive")
        if self.min_trace_ratio <= 0 or self.max_half_life <= 0 or self.max_weight_drift < 0:
            raise ValueError("robustness thresholds must be positive")
        if not 0.0 <= self.minimum_stable_fraction <= 1.0:
            raise ValueError("minimum_stable_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class RobustnessAwareStrategyResult:
    """Filtered and unfiltered simulations for the same basket strategy."""

    filtered: BasketStrategyResult
    unfiltered: BasketStrategyResult
    stability: pd.DataFrame
    comparison: dict[str, float]


def filter_candidate_baskets_by_robustness(
    candidates: Sequence[BasketCandidate],
    prices: pd.DataFrame,
    *,
    gate: RobustnessGateConfig | None = None,
) -> tuple[BasketCandidate, ...]:
    """Retain candidate baskets whose latest relationship is historically stable.

    Parameters
    ----------
    candidates
        Candidate baskets produced by the universe-selection stage.
    prices
        Aligned price table containing all candidate tickers.
    gate
        Stability thresholds used to accept or reject candidates.

    Returns
    -------
    tuple of BasketCandidate
        Candidates satisfying both the latest-window and stable-fraction limits.
    """
    gate = gate or RobustnessGateConfig()
    if not gate.enabled:
        return tuple(candidates)
    accepted: list[BasketCandidate] = []
    for candidate in candidates:
        missing = set(candidate.tickers).difference(prices.columns)
        if missing:
            continue
        try:
            result = rolling_basket_robustness(
                prices.loc[:, list(candidate.tickers)],
                window=gate.window,
                step=gate.step,
                min_trace_ratio=gate.min_trace_ratio,
                max_half_life=gate.max_half_life,
                max_weight_drift=gate.max_weight_drift,
            )
        except (ValueError, ArithmeticError):
            continue
        if not result.break_detected and result.stable_fraction >= gate.minimum_stable_fraction:
            accepted.append(candidate)
    return tuple(accepted)


def historical_stability_table(
    prices: pd.DataFrame,
    evaluation_dates: pd.Index,
    *,
    gate: RobustnessGateConfig,
) -> pd.DataFrame:
    """Calculate leakage-free stability diagnostics at historical decision dates.

    Only prices observed on or before each evaluation date are used.
    """
    clean = prices.astype(float).dropna(how="any").sort_index()
    rows: list[dict[str, object]] = []
    for date in pd.DatetimeIndex(evaluation_dates):
        history = clean.loc[:date]
        if len(history) < gate.window:
            rows.append({"date": date, "stable": False, "reason": "insufficient_history"})
            continue
        try:
            result = rolling_basket_robustness(
                history,
                window=gate.window,
                step=gate.step,
                min_trace_ratio=gate.min_trace_ratio,
                max_half_life=gate.max_half_life,
                max_weight_drift=gate.max_weight_drift,
            )
            stable = bool(
                not result.break_detected
                and result.stable_fraction >= gate.minimum_stable_fraction
            )
            rows.append({
                "date": date,
                "stable": stable,
                "trace_ratio": result.latest_trace_ratio,
                "half_life": result.latest_half_life,
                "weight_drift": result.latest_weight_drift,
                "stable_fraction": result.stable_fraction,
                "reason": "stable" if stable else "; ".join(result.warnings),
            })
        except (ValueError, ArithmeticError):
            rows.append({"date": date, "stable": False, "reason": "robustness_fit_failed"})
    return pd.DataFrame(rows).set_index("date").sort_index()


def run_robustness_aware_strategy(
    prices: pd.DataFrame,
    *,
    strategy: BasketStrategyConfig | None = None,
    gate: RobustnessGateConfig | None = None,
) -> RobustnessAwareStrategyResult:
    """Compare a basket strategy with and without leakage-free stability gating."""
    strategy = strategy or BasketStrategyConfig()
    gate = gate or RobustnessGateConfig()
    clean = prices.astype(float).dropna(how="any").sort_index()
    records = walk_forward_evidence(
        clean,
        train_window=strategy.train_window,
        z_window=strategy.z_window,
        horizon=strategy.horizon,
        step=strategy.step,
        min_trace_ratio=strategy.min_trace_ratio,
    )
    if records.empty:
        raise ValueError("no valid walk-forward basket evaluations were produced")
    probabilities = expanding_calibrated_probabilities(
        records,
        horizon=strategy.horizon,
        min_samples=strategy.min_calibration_samples,
    )
    stability = historical_stability_table(clean, probabilities.index, gate=gate)
    filtered_probabilities = probabilities.copy()
    if gate.enabled:
        unstable = stability.index[~stability["stable"].astype(bool)]
        filtered_probabilities.loc[filtered_probabilities.index.intersection(unstable)] = 0.5

    unfiltered_backtest = run_long_only_policy_backtest(
        clean, probabilities, policy=strategy.policy, initial_cash=strategy.initial_cash
    )
    filtered_backtest = run_long_only_policy_backtest(
        clean, filtered_probabilities, policy=strategy.policy, initial_cash=strategy.initial_cash
    )
    benchmark = equal_weight_benchmark(clean, strategy.initial_cash)

    def assemble(backtest, probs) -> BasketStrategyResult:
        strategy_profit = float(backtest.equity.iloc[-1] - strategy.initial_cash)
        benchmark_profit = float(benchmark.iloc[-1] - strategy.initial_cash)
        summary = dict(backtest.metrics)
        summary.update({
            "starting_value": float(strategy.initial_cash),
            "ending_value": float(backtest.equity.iloc[-1]),
            "profit": strategy_profit,
            "benchmark_ending_value": float(benchmark.iloc[-1]),
            "benchmark_profit": benchmark_profit,
            "excess_profit": strategy_profit - benchmark_profit,
            "evaluation_count": float(probs.shape[0]),
        })
        return BasketStrategyResult(records, probs, backtest, benchmark, summary)

    filtered = assemble(filtered_backtest, filtered_probabilities)
    unfiltered = assemble(unfiltered_backtest, probabilities)
    comparison = {
        "filtered_profit": filtered.summary["profit"],
        "unfiltered_profit": unfiltered.summary["profit"],
        "profit_difference": filtered.summary["profit"] - unfiltered.summary["profit"],
        "filtered_maximum_drawdown": filtered.summary["maximum_drawdown"],
        "unfiltered_maximum_drawdown": unfiltered.summary["maximum_drawdown"],
        "filtered_trade_count": filtered.summary["trade_count"],
        "unfiltered_trade_count": unfiltered.summary["trade_count"],
        "stable_evaluation_fraction": float(stability["stable"].mean()),
    }
    return RobustnessAwareStrategyResult(filtered, unfiltered, stability, comparison)
