"""Diagnostics for investigating one candidate cointegrated basket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cobasket.cointegration import build_spread, johansen_test
from cobasket.evidence.cointegration import rolling_z_score


@dataclass(frozen=True)
class BasketInvestigation:
    """Computed diagnostics for one basket.

    Parameters
    ----------
    tickers
        Asset symbols in fitted column order.
    prices
        Clean aligned adjusted prices.
    normalized_prices
        Prices divided by their first observation.
    spread
        Fitted cointegrating linear combination.
    z_score
        Rolling standardized spread.
    weights
        L1-normalized Johansen eigenvector.
    trace_ratio
        Rank-zero trace statistic divided by its 95 percent critical value.
    latest_z_score
        Latest finite rolling z-score.
    """

    tickers: tuple[str, ...]
    prices: pd.DataFrame
    normalized_prices: pd.DataFrame
    spread: pd.Series
    z_score: pd.Series
    weights: pd.Series
    trace_ratio: float
    latest_z_score: float


def investigate_basket(prices: pd.DataFrame, *, window: int = 60) -> BasketInvestigation:
    """Fit and summarize a candidate cointegrated basket.

    Parameters
    ----------
    prices
        Adjusted prices with dates in rows and at least two assets in columns.
    window
        Rolling window used to standardize the fitted spread.

    Returns
    -------
    BasketInvestigation
        Prices, spread, z-score, weights, and Johansen diagnostic strength.

    Raises
    ------
    ValueError
        If the price table is invalid or too short for the requested window.
    """
    if window < 2:
        raise ValueError("window must be at least two")
    clean = prices.astype(float).dropna(axis=0, how="any").sort_index()
    if clean.shape[1] < 2:
        raise ValueError("basket investigation requires at least two assets")
    if len(clean) <= window:
        raise ValueError("price history must be longer than the z-score window")
    if (clean <= 0.0).any().any():
        raise ValueError("prices must be strictly positive")

    result = johansen_test(clean, verbose=False)
    spread, weights = build_spread(clean, result)
    z_score = rolling_z_score(spread, window=window)
    finite_z = z_score[np.isfinite(z_score)]
    latest_z = float(finite_z.iloc[-1]) if not finite_z.empty else float("nan")
    critical = float(result.cvt[0, 1])
    trace_ratio = float(result.lr1[0] / critical)
    normalized = clean.divide(clean.iloc[0])

    return BasketInvestigation(
        tickers=tuple(str(column) for column in clean.columns),
        prices=clean,
        normalized_prices=normalized,
        spread=spread,
        z_score=z_score,
        weights=pd.Series(weights, index=clean.columns, name="weight", dtype=float),
        trace_ratio=trace_ratio,
        latest_z_score=latest_z,
    )
