"""Cointegration testing, market-factor removal, and basket clustering."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cobasket.data import fetch_universe


def johansen_test(price_df, det_order=0, k_ar_diff=1, verbose=True):
    """
    det_order: 0 = no deterministic trend (usual default for prices)
    k_ar_diff: number of lagged differences (start with 1)
    Returns the fitted result; check trace_stat vs crit vals (90/95/99%)
    """
    result = coint_johansen(price_df.values, det_order, k_ar_diff)
    if verbose:
        print("Trace statistic vs critical values (90%, 95%, 99%):")
        for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt)):
            print(f"  r <= {i}: stat={stat:.2f}  crit={crit}")
    return result


def build_spread(price_df, result, vector_idx=0):
    """
    Use the eigenvector corresponding to the strongest cointegrating
    relationship (largest eigenvalue -> index 0) to build the spread.
    """
    weights = result.evec[:, vector_idx]
    weights = weights / weights[0]  # normalize vs first asset
    spread = price_df.values @ weights
    return pd.Series(spread, index=price_df.index, name="spread"), weights


def remove_market_factor(returns, market_col="SPY"):
    """
    Regress each stock's returns on the market return and keep the residual.
    This is the finance equivalent of subtracting a common-mode signal --
    otherwise everything looks 'correlated' just because the whole market
    moves together, drowning out the idiosyncratic co-movement you actually
    want (e.g. two miners moving together because of ore prices specifically).
    """
    market = returns[market_col]
    residuals = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col == market_col:
            continue
        beta = np.cov(returns[col], market)[0, 1] / np.var(market)
        residuals[col] = returns[col] - beta * market
    return residuals


def cluster_candidates(residual_returns, distance_threshold=1.0):
    """
    Hierarchical clustering on 1-correlation distance.
    distance_threshold: cophenetic distance cutoff (lower = tighter clusters).
    Roughly: 0 = identical co-movement, 2 = perfectly anti-correlated.
    Start around 0.6-0.8 and adjust based on cluster sizes returned.
    """
    corr = residual_returns.corr()
    dist = 1 - corr
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")

    cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")
    clusters = {}
    for ticker, cid in zip(corr.columns, cluster_ids):
        clusters.setdefault(cid, []).append(ticker)

    # only keep clusters with >=2 members (singletons aren't baskets)
    candidate_baskets = [members for members in clusters.values() if len(members) >= 2]
    return candidate_baskets, Z, corr


def screen_universe(
    tickers,
    period="2y",
    distance_threshold=0.8,
    min_trace_stat_ratio=1.0,
    max_basket_size=8,
    cache_dir="price_cache",
):
    """
    Full screening pipeline: fetch -> remove market factor -> cluster ->
    confirm each cluster with Johansen. Returns (confirmed, prices, Z, corr)
    where confirmed is a list of (tickers, johansen_result, stat, crit).
    """
    prices = fetch_universe(tickers, period, cache_dir=cache_dir)
    market_col = "SPY"
    returns = prices.pct_change().dropna()
    residuals = remove_market_factor(returns, market_col)

    candidate_baskets, Z, corr = cluster_candidates(residuals, distance_threshold)
    print(f"Found {len(candidate_baskets)} candidate cluster(s) with >=2 members")

    confirmed = []
    for basket in candidate_baskets:
        if len(basket) < 2 or len(basket) > max_basket_size:
            continue  # Johansen gets unstable/slow with too many series
        basket_prices = prices[basket].dropna()
        if len(basket_prices) < 100:
            continue
        try:
            result = coint_johansen(basket_prices.values, 0, 1)
        except Exception as e:
            print(f"  Skipping {basket}: {e}")
            continue

        # trace stat for r<=0 (no cointegration) vs 95% critical value
        stat, crit_95 = result.lr1[0], result.cvt[0][1]
        if stat > crit_95 * min_trace_stat_ratio:
            confirmed.append((basket, result, stat, crit_95))
            print(f"  CONFIRMED: {basket}  (trace stat {stat:.1f} > crit {crit_95:.1f})")
        else:
            print(f"  Rejected:  {basket}  (trace stat {stat:.1f} <= crit {crit_95:.1f})")

    return confirmed, prices, Z, corr
