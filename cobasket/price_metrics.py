"""Leakage-safe momentum, trend, and volatility metrics for trading rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.strategy_rules import RuleBacktestResult, StrategyRules, compare_rule_strategies


@dataclass(frozen=True)
class PriceMetricConfig:
    """Configuration for trailing price-based metrics.

    Parameters
    ----------
    momentum_window
        Observations used for trailing compounded return.
    trend_window
        Observations used for the trailing simple moving average.
    volatility_window
        Observations used for trailing return volatility.
    volatility_baseline_window
        Earlier volatility observations used to estimate the current volatility
        percentile. This is an expanding/trailing historical comparison and never
        uses future values.
    periods_per_year
        Sampling frequency used to annualise volatility.
    momentum_scale
        Positive scale controlling how quickly momentum is mapped into ``[-1, 1]``.
    trend_scale
        Positive scale controlling how quickly fractional trend distance is mapped
        into ``[-1, 1]``.
    high_volatility_percentile
        Historical percentile above which ``high_volatility`` is true.
    """

    momentum_window: int = 60
    trend_window: int = 100
    volatility_window: int = 20
    volatility_baseline_window: int = 252
    periods_per_year: int = 252
    momentum_scale: float = 0.15
    trend_scale: float = 0.10
    high_volatility_percentile: float = 0.90

    def __post_init__(self) -> None:
        """Validate trailing-window and scaling settings."""
        windows = (
            self.momentum_window,
            self.trend_window,
            self.volatility_window,
            self.volatility_baseline_window,
            self.periods_per_year,
        )
        if any(value < 2 for value in windows):
            raise ValueError("all windows and periods_per_year must be at least two")
        if self.momentum_scale <= 0.0 or self.trend_scale <= 0.0:
            raise ValueError("metric scales must be positive")
        if not 0.0 < self.high_volatility_percentile < 1.0:
            raise ValueError("high_volatility_percentile must lie in (0, 1)")


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, chronological positive price table."""
    clean = prices.astype(float).sort_index()
    if clean.empty or clean.shape[1] < 1:
        raise ValueError("prices must contain at least one ticker")
    if clean.index.has_duplicates:
        raise ValueError("price index must not contain duplicate dates")
    if (clean.dropna(how="all") <= 0.0).any().any():
        raise ValueError("finite prices must be strictly positive")
    return clean


def trailing_momentum(prices: pd.DataFrame, *, window: int = 60) -> pd.DataFrame:
    """Calculate trailing compounded returns.

    Parameters
    ----------
    prices
        Positive price table with dates in rows and tickers in columns.
    window
        Number of price intervals over which return is measured.

    Returns
    -------
    pandas.DataFrame
        Fractional trailing return. A value of ``0.10`` means the price is 10%
        above its value ``window`` observations earlier.
    """
    if window < 1:
        raise ValueError("window must be positive")
    clean = _clean_prices(prices)
    return clean.pct_change(periods=window, fill_method=None).rename_axis(index="date")


def momentum_score(
    prices: pd.DataFrame,
    *,
    window: int = 60,
    scale: float = 0.15,
) -> pd.DataFrame:
    """Map trailing return smoothly into the bounded interval ``[-1, 1]``.

    The hyperbolic tangent prevents extreme historical returns from dominating
    rule thresholds while preserving sign and ordering.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    raw = trailing_momentum(prices, window=window)
    return np.tanh(raw / scale)


def trailing_trend_distance(prices: pd.DataFrame, *, window: int = 100) -> pd.DataFrame:
    """Calculate fractional distance from a trailing simple moving average.

    Positive values mean the current price is above its trailing baseline; negative
    values mean it is below. The moving average includes only the current and
    preceding observations.
    """
    if window < 2:
        raise ValueError("window must be at least two")
    clean = _clean_prices(prices)
    baseline = clean.rolling(window=window, min_periods=window).mean()
    return clean.divide(baseline) - 1.0


def trend_score(
    prices: pd.DataFrame,
    *,
    window: int = 100,
    scale: float = 0.10,
) -> pd.DataFrame:
    """Map moving-average distance smoothly into ``[-1, 1]``."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    raw = trailing_trend_distance(prices, window=window)
    return np.tanh(raw / scale)


def trailing_volatility(
    prices: pd.DataFrame,
    *,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate trailing annualised volatility from simple returns.

    Volatility is the standard deviation of returns. It describes variability,
    not direction: a rapidly rising and a rapidly falling asset may both have high
    volatility.
    """
    if window < 2 or periods_per_year < 2:
        raise ValueError("window and periods_per_year must be at least two")
    clean = _clean_prices(prices)
    returns = clean.pct_change(fill_method=None)
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def trailing_percentile(table: pd.DataFrame, *, window: int = 252) -> pd.DataFrame:
    """Calculate each value's percentile within its trailing historical window.

    The latest value is ranked only against values available at that date. Missing
    values remain missing and no backward or future filling is performed.
    """
    if window < 2:
        raise ValueError("window must be at least two")

    def percentile(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            return float("nan")
        latest = finite[-1]
        return float(np.mean(finite <= latest))

    return table.rolling(window=window, min_periods=2).apply(percentile, raw=True)


def build_price_metrics(
    prices: pd.DataFrame,
    *,
    config: PriceMetricConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all price metrics expected by the declarative rule engine.

    Parameters
    ----------
    prices
        Positive price table.
    config
        Metric window, scaling, and volatility-threshold settings.

    Returns
    -------
    dict of str to pandas.DataFrame
        Tables named ``momentum``, ``momentum_return``, ``trend``,
        ``trend_distance``, ``volatility``, ``volatility_percentile``, and
        ``high_volatility``. Boolean flags are represented as ``0.0`` and ``1.0``
        so they remain compatible with the existing metric-table interface.
    """
    config = config or PriceMetricConfig()
    raw_momentum = trailing_momentum(prices, window=config.momentum_window)
    bounded_momentum = np.tanh(raw_momentum / config.momentum_scale)
    raw_trend = trailing_trend_distance(prices, window=config.trend_window)
    bounded_trend = np.tanh(raw_trend / config.trend_scale)
    volatility = trailing_volatility(
        prices,
        window=config.volatility_window,
        periods_per_year=config.periods_per_year,
    )
    volatility_percentile = trailing_percentile(
        volatility,
        window=config.volatility_baseline_window,
    )
    high_volatility = (
        volatility_percentile >= config.high_volatility_percentile
    ).where(volatility_percentile.notna()).astype(float)
    return {
        "momentum": bounded_momentum,
        "momentum_return": raw_momentum,
        "trend": bounded_trend,
        "trend_distance": raw_trend,
        "volatility": volatility,
        "volatility_percentile": volatility_percentile,
        "high_volatility": high_volatility,
    }


def merge_metric_tables(
    *metric_sets: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Merge named metric mappings while rejecting accidental name collisions."""
    merged: dict[str, pd.DataFrame] = {}
    for metric_set in metric_sets:
        overlap = set(merged).intersection(metric_set)
        if overlap:
            raise ValueError(f"duplicate metric names: {sorted(overlap)}")
        merged.update(metric_set)
    return merged


def compare_incremental_metric_strategies(
    prices: pd.DataFrame,
    base_metrics: Mapping[str, pd.DataFrame],
    price_metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    *,
    initial_cash: float = 10_000.0,
) -> tuple[pd.DataFrame, dict[str, RuleBacktestResult]]:
    """Compare strategies after joining base and price-derived metrics.

    This helper does not optimise thresholds. It runs the supplied, pre-declared
    strategies on exactly the same prices and metric history so added conditions
    can be evaluated incrementally.
    """
    metrics = merge_metric_tables(base_metrics, price_metrics)
    return compare_rule_strategies(
        prices,
        metrics,
        strategies,
        initial_cash=initial_cash,
    )
