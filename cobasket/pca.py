"""
PCA-based factor decomposition: an alternative to the SPY-regression +
correlation-clustering approach in cointegration.py.

Same goal (find candidate baskets to feed into Johansen), different route:
instead of assuming SPY is the right market proxy, let PCA find the
dominant shared modes of variation empirically from the return matrix
itself (rows = days, columns = stocks). Analogous to running PCA on a
matrix of stellar spectra or light curves -- PC1 here is almost always
"the market", later PCs pick up sector/factor structure, and residuals
after removing the top-k PCs are the idiosyncratic part.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from cobasket.data import fetch_universe


def compute_pca(returns, n_components=10):
    """
    Standardize returns (each stock to unit variance -- otherwise
    high-volatility stocks dominate the PCs) and run PCA.

    Returns:
        pca: fitted sklearn PCA object (pca.explained_variance_ratio_ etc.)
        loadings: DataFrame (tickers x components) -- each stock's
                  coordinate on each PC, i.e. how much it's driven by
                  that latent factor
        scores: DataFrame (days x components) -- the latent factor
                time series themselves
        standardized: the standardized returns actually fed to PCA
    """
    std = returns.std()
    standardized = returns / std  # unit-variance standardization

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(standardized.values)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    scores = pd.DataFrame(
        scores,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return pca, loadings, scores, standardized


def remove_top_pcs(standardized_returns, pca, scores, n_remove=1):
    """
    Reconstruct and subtract the top n_remove PCs from the standardized
    returns, leaving the idiosyncratic residual. n_remove=1 removes just
    the market-wide component (usually PC1); raise it to also strip out
    sector-level factors if PC2/PC3 look like sectors on the loadings plot.
    """
    reconstruction = scores.iloc[:, :n_remove].values @ pca.components_[:n_remove, :]
    residuals = standardized_returns.values - reconstruction
    return pd.DataFrame(residuals, index=standardized_returns.index, columns=standardized_returns.columns)


def cluster_by_loadings(loadings, n_components_for_clustering=5, distance_threshold=1.5):
    """
    Cluster stocks on their loading vectors (coordinates in PC-space)
    rather than on pairwise correlation. Stocks with similar loadings on
    PC2..PCk are driven by the same latent factors, even if their raw
    pairwise correlation is muddied by noise.
    """
    coords = loadings.iloc[:, :n_components_for_clustering]
    dist = pdist(coords.values, metric="euclidean")
    Z = linkage(dist, method="average")

    cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")
    clusters = {}
    for ticker, cid in zip(loadings.index, cluster_ids):
        clusters.setdefault(cid, []).append(ticker)

    candidate_baskets = [members for members in clusters.values() if len(members) >= 2]
    return candidate_baskets, Z


def pca_screen_universe(
    tickers,
    period="2y",
    n_components=10,
    n_remove=1,
    n_components_for_clustering=5,
    distance_threshold=1.5,
    max_basket_size=8,
    min_trace_stat_ratio=1.0,
    cache_dir="price_cache",
):
    """
    Full PCA-based screening pipeline: fetch -> PCA -> remove top PCs ->
    cluster on loadings -> confirm each cluster with Johansen.
    Mirrors cointegration.screen_universe's return shape.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    prices = fetch_universe(tickers, period, cache_dir=cache_dir)
    returns = prices.pct_change().dropna()

    pca, loadings, scores, standardized = compute_pca(returns, n_components=n_components)
    residuals = remove_top_pcs(standardized, pca, scores, n_remove=n_remove)

    candidate_baskets, Z = cluster_by_loadings(
        loadings, n_components_for_clustering, distance_threshold
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
            result = coint_johansen(basket_prices.values, 0, 1)
        except Exception as e:
            print(f"  Skipping {basket}: {e}")
            continue

        stat, crit_95 = result.lr1[0], result.cvt[0][1]
        if stat > crit_95 * min_trace_stat_ratio:
            confirmed.append((basket, result, stat, crit_95))
            print(f"  CONFIRMED: {basket}  (trace stat {stat:.1f} > crit {crit_95:.1f})")
        else:
            print(f"  Rejected:  {basket}  (trace stat {stat:.1f} <= crit {crit_95:.1f})")

    return confirmed, prices, pca, loadings, scores
