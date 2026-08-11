"""Cointegration testing, common-factor removal, and basket clustering."""

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
    """Run statsmodels' Johansen test while containing a known cast warning.

    Parameters
    ----------
    values
        Two-dimensional price array with observations in rows.
    det_order
        Deterministic-term setting passed to statsmodels.
    k_ar_diff
        Number of lagged first differences.

    Returns
    -------
    statsmodels.tsa.vector_ar.vecm.JohansenTestResult
        Johansen test result.

    Notes
    -----
    Some numerically valid statsmodels fits emit a ``ComplexWarning`` while
    constructing real-valued trace statistics from intermediate complex-valued
    eigensystem calculations. Cobasket still validates the returned
    cointegration vector separately in :func:`normalize_cointegration_weights`.
    """
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
    """Run the Johansen cointegration test on a price table.

    Parameters
    ----------
    price_df
        Price observations with dates in rows and assets in columns.
    det_order
        Deterministic-term specification passed to
        :func:`statsmodels.tsa.vector_ar.vecm.coint_johansen`. A value of zero
        includes a constant term in the cointegration relation.
    k_ar_diff
        Number of lagged first differences in the vector error-correction
        model.
    verbose
        Print trace statistics and critical values when ``True``.

    Returns
    -------
    statsmodels.tsa.vector_ar.vecm.JohansenTestResult
        Fitted Johansen test result.

    Raises
    ------
    ValueError
        If fewer than two assets or too few observations are supplied.
    """
    if price_df.shape[1] < 2:
        raise ValueError("Johansen testing requires at least two price series")
    if len(price_df) <= k_ar_diff + 2:
        raise ValueError("not enough observations for the requested lag order")

    result = _coint_johansen_quiet(price_df.to_numpy(dtype=float), det_order, k_ar_diff)
    if verbose:
        print("Trace statistic vs critical values (90%, 95%, 99%):")
        for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt)):
            print(f"  r <= {i}: stat={stat:.2f}  crit={crit}")
    return result


def normalize_cointegration_weights(weights: Sequence[complex | float]) -> np.ndarray:
    """Convert a Johansen eigenvector into stable real-valued weights.

    The eigenvector is normalized so that the sum of its absolute components is
    one. This is analogous to normalizing a mode vector by an L1 norm rather
    than dividing by a single component that may be close to zero.

    Parameters
    ----------
    weights
        Raw cointegration-vector components.

    Returns
    -------
    numpy.ndarray
        One-dimensional real-valued vector with unit absolute sum.

    Raises
    ------
    ValueError
        If the vector is empty, non-finite, zero, or contains a material
        imaginary component.
    """
    array = np.asarray(weights)
    array = np.real_if_close(array, tol=1000)
    if np.iscomplexobj(array):
        max_imag = float(np.max(np.abs(np.imag(array))))
        raise ValueError(
            "cointegration weights contain a material imaginary component "
            f"(maximum |imaginary|={max_imag:.3e})"
        )

    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("cointegration weight vector is empty")
    if not np.isfinite(array).all():
        raise ValueError("cointegration weights must be finite")

    scale = np.sum(np.abs(array))
    if scale <= np.finfo(float).eps:
        raise ValueError("cointegration weight vector is numerically zero")
    return array / scale


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

    Raises
    ------
    ValueError
        If the eigenvector cannot be converted to a finite real vector or its
        length does not match the number of assets.
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
    """Remove the linear common-market component from each return series.

    This is comparable to subtracting a common-mode signal from several
    detectors. The residuals retain movements not explained by the selected
    market proxy.

    Parameters
    ----------
    returns
        Fractional or logarithmic returns with assets in columns.
    market_col
        Column used as the common-market proxy.

    Returns
    -------
    pandas.DataFrame
        Residual return series for every non-market column.

    Raises
    ------
    KeyError
        If ``market_col`` is absent.
    ValueError
        If the market series has zero variance.
    """
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
    betas = np.mean(centered_assets * centered_market[:, None], axis=0) / market_variance
    residual_values = assets - market[:, None] * betas[None, :]
    return pd.DataFrame(residual_values, index=returns.index, columns=asset_columns)


def cluster_candidates(
    residual_returns: pd.DataFrame,
    distance_threshold: float = 1.0,
) -> tuple[list[list[str]], np.ndarray, pd.DataFrame]:
    """Cluster assets using correlation distance between residual returns.

    Parameters
    ----------
    residual_returns
        Return series after common-factor removal.
    distance_threshold
        Hierarchical-clustering cut height. Correlation distance is
        ``1 - correlation``; zero means identical movement and two means
        perfectly anti-correlated movement.

    Returns
    -------
    candidate_baskets
        Clusters containing at least two assets.
    linkage_matrix
        SciPy hierarchical-clustering linkage matrix.
    correlation
        Pairwise correlation matrix used to construct distances.

    Raises
    ------
    ValueError
        If fewer than two asset columns are supplied.
    """
    if residual_returns.shape[1] < 2:
        raise ValueError("clustering requires at least two return series")

    correlation = residual_returns.corr()
    distance = (1.0 - correlation).clip(lower=0.0, upper=2.0)
    distance_values = distance.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(distance_values, 0.0)
    condensed = squareform(distance_values, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    cluster_ids = fcluster(
        linkage_matrix,
        t=distance_threshold,
        criterion="distance",
    )
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
):
    """Screen a ticker universe for candidate cointegrated baskets.

    Parameters
    ----------
    tickers
        Asset symbols to screen. ``SPY`` is added internally as a market proxy.
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
    prices = fetch_universe(tickers, period, cache_dir=cache_dir)
    market_col = "SPY"
    returns = prices.pct_change().dropna()
    residuals = remove_market_factor(returns, market_col)

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
            print(
                f"  CONFIRMED: {basket}  "
                f"(trace stat {statistic:.1f} > crit {critical_95:.1f})"
            )
        else:
            print(
                f"  Rejected:  {basket}  "
                f"(trace stat {statistic:.1f} <= crit {critical_95:.1f})"
            )

    return confirmed, prices, linkage_matrix, correlation
