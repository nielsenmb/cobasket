"""Cointegration utilities for related-stock basket analysis."""

from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cobasket.data import fetch_universe


def _coint_johansen_quiet(values: np.ndarray, det_order: int, k_ar_diff: int):
    """Run statsmodels Johansen while suppressing its internal complex cast warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
        return coint_johansen(values, det_order, k_ar_diff)


def johansen_test(
    price_df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    verbose: bool = True,
):
    """Run a Johansen cointegration test on an aligned price table.

    Parameters
    ----------
    price_df
        Price observations with dates in rows and assets in columns.
    det_order
        Deterministic-term setting passed to statsmodels.
    k_ar_diff
        Number of lagged first differences.
    verbose
        Print trace statistics when ``True``.

    Returns
    -------
    statsmodels result
        Johansen test result object.
    """
    values = price_df.astype(float).dropna().to_numpy()
    if values.shape[0] < 5 or values.shape[1] < 2:
        raise ValueError("Johansen test requires at least two series and five observations")
    result = _coint_johansen_quiet(values, det_order, k_ar_diff)
    if verbose:
        for index, ticker in enumerate(price_df.columns):
            _ = ticker
            if index < len(result.lr1):
                print(f"rank {index}: trace={result.lr1[index]:.3f}, crit95={result.cvt[index, 1]:.3f}")
    return result


def normalize_cointegration_weights(weights: Sequence[complex | float]) -> np.ndarray:
    """Convert a Johansen eigenvector to a finite real L1-normalized vector."""
    raw = np.asarray(weights)
    real = np.real_if_close(raw, tol=1000)
    if np.iscomplexobj(real):
        raise ValueError("cointegration weight vector has a material imaginary component")
    real = np.asarray(real, dtype=float)
    if not np.all(np.isfinite(real)):
        raise ValueError("cointegration weight vector contains non-finite values")
    scale = float(np.sum(np.abs(real)))
    if scale <= np.finfo(float).eps:
        raise ValueError("cointegration weight vector has zero norm")
    return real / scale


def build_spread(
    price_df: pd.DataFrame,
    result,
    vector_idx: int = 0,
) -> tuple[pd.Series, np.ndarray]:
    """Construct a real-valued spread from a Johansen eigenvector.

    Parameters
    ----------
    price_df
        Price observations with dates in rows and assets in columns.
    result
        Result returned by :func:`johansen_test` or ``coint_johansen``.
    vector_idx
        Eigenvector index. Index zero corresponds to the strongest estimated
        cointegrating relationship.

    Returns
    -------
    spread
        Weighted linear combination of the input prices.
    weights
        Stable L1-normalized cointegration weights.
    """
    weights = normalize_cointegration_weights(result.evec[:, vector_idx])
    if len(weights) != price_df.shape[1]:
        raise ValueError("weight vector length does not match the number of assets")
    spread_values = price_df.to_numpy(dtype=float) @ weights
    spread = pd.Series(spread_values, index=price_df.index, name="spread", dtype=float)
    return spread, weights


def remove_market_factor(
    returns: pd.DataFrame,
    market_col: str = "SPY",
) -> pd.DataFrame:
    """Remove the linear common-market component from each return series."""
    if market_col not in returns:
        raise KeyError(f"market column {market_col!r} is absent")
    market = returns[market_col].to_numpy(dtype=float)
    market_variance = float(np.var(market))
    if market_variance <= np.finfo(float).eps:
        raise ValueError("market return series has zero variance")
    asset_columns = [column for column in returns.columns if column != market_col]
    assets = returns.loc[:, asset_columns].to_numpy(dtype=float)
    centered_market = market - market.mean()
    centered_assets = assets - assets.mean(axis=0)
    covariance = np.sum(centered_assets * centered_market[:, None], axis=0) / (len(market) - 1)
    betas = covariance / market_variance
    residual_values = assets - market[:, None] * betas[None, :]
    return pd.DataFrame(residual_values, index=returns.index, columns=asset_columns)


def cluster_candidates(
    residual_returns: pd.DataFrame,
    distance_threshold: float = 1.0,
) -> tuple[list[list[str]], np.ndarray, pd.DataFrame]:
    """Cluster assets using correlation distance between residual returns."""
    if residual_returns.shape[1] < 2:
        raise ValueError("clustering requires at least two return series")
    correlation = residual_returns.corr()
    distance = (1.0 - correlation).clip(lower=0.0, upper=2.0)
    distance_values = distance.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(distance_values, 0.0)
    condensed = squareform(distance_values, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    cluster_ids = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for ticker, cluster_id in zip(correlation.columns, cluster_ids):
        clusters.setdefault(int(cluster_id), []).append(str(ticker))
    candidate_baskets = [members for members in clusters.values() if len(members) >= 2]
    return candidate_baskets, linkage_matrix, correlation


def screen_universe(
    tickers: Sequence[str],
    period: str = "2y",
    distance_threshold: float = 0.8,
    min_trace_stat_ratio: float = 1.0,
    max_basket_size: int = 8,
    cache_dir: str = "price_cache",
    market_ticker: str = "SPY",
):
    """Screen a ticker universe for candidate cointegrated baskets.

    Parameters
    ----------
    tickers
        Asset symbols to screen.
    period
        Historical period accepted by ``yfinance``.
    distance_threshold
        Correlation-distance cut used by hierarchical clustering.
    min_trace_stat_ratio
        Required ratio between the Johansen trace statistic and its 95 percent
        critical value.
    max_basket_size
        Maximum number of assets passed to the Johansen test.
    cache_dir
        Directory used by the price-data cache.
    market_ticker
        Market proxy added to the download and removed from residual returns.

    Returns
    -------
    confirmed
        Tuples of basket symbols, Johansen result, trace statistic, and 95
        percent critical value.
    prices
        Downloaded aligned prices, including the market proxy.
    linkage_matrix
        Hierarchical-clustering linkage matrix.
    correlation
        Residual-return correlation matrix.
    """
    prices = fetch_universe(tickers, period, market_ticker=market_ticker, cache_dir=cache_dir)
    returns = prices.pct_change().dropna()
    residuals = remove_market_factor(returns, market_ticker)
    candidate_baskets, linkage_matrix, correlation = cluster_candidates(
        residuals,
        distance_threshold,
    )
    print(f"Found {len(candidate_baskets)} candidate cluster(s) with >=2 members")

    confirmed = []
    for basket in candidate_baskets:
        if len(basket) < 2 or len(basket) > max_basket_size:
            continue
        basket_prices = prices[basket].dropna()
        if len(basket_prices) < 100:
            continue
        try:
            result = _coint_johansen_quiet(basket_prices.to_numpy(dtype=float), 0, 1)
        except Exception as exc:
            print(f"  Skipping {basket}: {exc}")
            continue
        statistic = float(result.lr1[0])
        critical_95 = float(result.cvt[0][1])
        if statistic > critical_95 * min_trace_stat_ratio:
            confirmed.append((basket, result, statistic, critical_95))
            print(f"  CONFIRMED: {basket}  (trace stat {statistic:.1f} > crit {critical_95:.1f})")
        else:
            print(f"  Rejected:  {basket}  (trace stat {statistic:.1f} <= crit {critical_95:.1f})")
    return confirmed, prices, linkage_matrix, correlation
