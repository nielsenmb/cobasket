"""PCA-based common-factor decomposition and basket screening."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cobasket.data import fetch_universe


def compute_pca(
    returns: pd.DataFrame,
    n_components: int = 10,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize asset returns and fit principal-component analysis.

    Parameters
    ----------
    returns
        Return observations with dates in rows and assets in columns.
    n_components
        Number of principal components to retain.

    Returns
    -------
    pca
        Fitted scikit-learn PCA estimator.
    loadings
        Asset coordinates in principal-component space.
    scores
        Principal-component time series.
    standardized
        Mean-centered, unit-variance returns supplied to PCA.

    Raises
    ------
    ValueError
        If the requested number of components is invalid or a return column has
        zero variance.
    """
    if returns.empty:
        raise ValueError("returns must not be empty")
    maximum = min(returns.shape)
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must lie between 1 and {maximum}")

    standard_deviation = returns.std()
    if (standard_deviation <= np.finfo(float).eps).any():
        raise ValueError("all return columns must have non-zero variance")
    standardized = (returns - returns.mean()) / standard_deviation

    pca = PCA(n_components=n_components)
    score_values = pca.fit_transform(standardized.to_numpy(dtype=float))
    component_names = [f"PC{i + 1}" for i in range(n_components)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=component_names,
    )
    scores = pd.DataFrame(
        score_values,
        index=returns.index,
        columns=component_names,
    )
    return pca, loadings, scores, standardized


def remove_top_pcs(
    standardized_returns: pd.DataFrame,
    pca: PCA,
    scores: pd.DataFrame,
    n_remove: int = 1,
) -> pd.DataFrame:
    """Subtract reconstructed leading principal components from returns.

    Parameters
    ----------
    standardized_returns
        Mean-centered, unit-variance return table used to fit PCA.
    pca
        Fitted PCA estimator.
    scores
        Principal-component time series returned by :func:`compute_pca`.
    n_remove
        Number of leading components to reconstruct and subtract.

    Returns
    -------
    pandas.DataFrame
        Residual standardized returns after common-factor removal.

    Raises
    ------
    ValueError
        If ``n_remove`` lies outside the fitted component range.
    """
    if not 0 <= n_remove <= pca.n_components_:
        raise ValueError("n_remove must lie between zero and the fitted component count")

    reconstruction = (
        scores.iloc[:, :n_remove].to_numpy(dtype=float)
        @ pca.components_[:n_remove, :]
    )
    residuals = standardized_returns.to_numpy(dtype=float) - reconstruction
    return pd.DataFrame(
        residuals,
        index=standardized_returns.index,
        columns=standardized_returns.columns,
    )


def cluster_by_loadings(
    loadings: pd.DataFrame,
    n_components_for_clustering: int = 5,
    distance_threshold: float = 1.5,
) -> tuple[list[list[str]], np.ndarray]:
    """Cluster assets by Euclidean distance in PCA-loading space.

    Parameters
    ----------
    loadings
        Asset coordinates in principal-component space.
    n_components_for_clustering
        Number of leading loading dimensions used for clustering.
    distance_threshold
        Hierarchical-clustering cut height.

    Returns
    -------
    candidate_baskets
        Clusters containing at least two assets.
    linkage_matrix
        SciPy hierarchical-clustering linkage matrix.

    Raises
    ------
    ValueError
        If too few assets or an invalid number of components is supplied.
    """
    if len(loadings) < 2:
        raise ValueError("clustering requires at least two assets")
    if not 1 <= n_components_for_clustering <= loadings.shape[1]:
        raise ValueError("invalid n_components_for_clustering")

    coordinates = loadings.iloc[:, :n_components_for_clustering]
    distances = pdist(coordinates.to_numpy(dtype=float), metric="euclidean")
    linkage_matrix = linkage(distances, method="average")

    cluster_ids = fcluster(
        linkage_matrix,
        t=distance_threshold,
        criterion="distance",
    )
    clusters: dict[int, list[str]] = {}
    for ticker, cluster_id in zip(loadings.index, cluster_ids):
        clusters.setdefault(int(cluster_id), []).append(str(ticker))

    candidate_baskets = [members for members in clusters.values() if len(members) >= 2]
    return candidate_baskets, linkage_matrix


def pca_screen_universe(
    tickers: Sequence[str],
    period: str = "2y",
    n_components: int = 10,
    n_remove: int = 1,
    n_components_for_clustering: int = 5,
    distance_threshold: float = 1.5,
    max_basket_size: int = 8,
    min_trace_stat_ratio: float = 1.0,
    cache_dir: str = "price_cache",
):
    """Screen a universe using PCA loadings followed by Johansen testing.

    Parameters
    ----------
    tickers
        Asset symbols to screen.
    period
        Historical period accepted by ``yfinance``.
    n_components
        Number of principal components to fit.
    n_remove
        Number of leading common components removed from standardized returns.
    n_components_for_clustering
        Number of loading dimensions used for clustering.
    distance_threshold
        Hierarchical-clustering cut height in loading space.
    max_basket_size
        Maximum number of assets passed to Johansen testing.
    min_trace_stat_ratio
        Required ratio between trace statistic and 95 percent critical value.
    cache_dir
        Directory used by the price-data cache.

    Returns
    -------
    confirmed
        Confirmed basket tuples.
    prices
        Downloaded aligned prices.
    pca
        Fitted PCA estimator.
    loadings
        Asset loading coordinates.
    scores
        Principal-component time series.
    """
    prices = fetch_universe(tickers, period, cache_dir=cache_dir)
    returns = prices.pct_change().dropna()

    pca, loadings, scores, standardized = compute_pca(
        returns,
        n_components=n_components,
    )
    remove_top_pcs(standardized, pca, scores, n_remove=n_remove)

    candidate_baskets, _ = cluster_by_loadings(
        loadings,
        n_components_for_clustering,
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
            result = coint_johansen(basket_prices.to_numpy(dtype=float), 0, 1)
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

    return confirmed, prices, pca, loadings, scores
