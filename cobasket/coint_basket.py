"""
Prototype: find a cointegrated basket and build a mean-reverting spread signal.

Requires: pip install yfinance statsmodels numpy pandas matplotlib scipy
"""

import os
import hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

CACHE_DIR = "price_cache"


def _cache_path(tickers, period):
    """Hash the ticker list + period into a stable filename."""
    key = ",".join(sorted(tickers)) + f"|{period}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{digest}.parquet")


def cached_download(tickers, period, force_refresh=False):
    """
    Thin caching wrapper around yf.download. Caches to disk as parquet,
    keyed on the exact ticker set + period requested. Delete price_cache/
    or pass force_refresh=True to bust the cache.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(tickers, period)

    if not force_refresh and os.path.exists(path):
        print(f"  [cache hit] loading {len(tickers)} tickers from {path}")
        return pd.read_parquet(path)

    print(f"  [cache miss] downloading {len(tickers)} tickers...")
    data = yf.download(tickers, period=period)["Close"]
    data.to_parquet(path)
    return data

# --- 1. Pick a basket you believe shares a common driver ---
# e.g. oil majors (shared driver: crude price)
TICKERS = ["XOM", "CVX", "COP", "OXY"]
LOOKBACK = "2y"


# =====================================================================
# SCREENING: find candidate baskets from a large universe
# =====================================================================

def fetch_universe(tickers, period, market_ticker="SPY"):
    """Fetch a universe of tickers plus a market proxy for factor removal."""
    all_tickers = list(tickers) + [market_ticker]
    data = cached_download(all_tickers, period)
    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))  # drop sparse tickers
    data = data.dropna()
    return data


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


def screen_universe(tickers, period="2y", distance_threshold=0.8, min_trace_stat_ratio=1.0):
    """
    Full screening pipeline: fetch -> remove market factor -> cluster ->
    confirm each cluster with Johansen. Returns list of (tickers, johansen_result).
    """
    prices = fetch_universe(tickers, period)
    market_col = "SPY"
    returns = prices.pct_change().dropna()
    residuals = remove_market_factor(returns, market_col)

    candidate_baskets, Z, corr = cluster_candidates(residuals, distance_threshold)
    print(f"Found {len(candidate_baskets)} candidate cluster(s) with >=2 members")

    confirmed = []
    for basket in candidate_baskets:
        if len(basket) < 2 or len(basket) > 8:
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

def fetch_prices(tickers, period):
    return cached_download(tickers, period).dropna()

def johansen_test(price_df, det_order=0, k_ar_diff=1):
    """
    det_order: 0 = no deterministic trend (usual default for prices)
    k_ar_diff: number of lagged differences (start with 1)
    Returns the fitted result; check trace_stat vs crit vals (90/95/99%)
    """
    result = coint_johansen(price_df.values, det_order, k_ar_diff)
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

def zscore_signal(spread, window=30, entry_z=2.0, exit_z=0.5):
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = (spread - mu) / sigma

    signal = pd.Series(0, index=spread.index)
    signal[z > entry_z] = -1   # spread too high -> expect reversion down
    signal[z < -entry_z] = 1   # spread too low -> expect reversion up
    signal[z.abs() < exit_z] = 0
    signal = signal.replace(0, np.nan).ffill().fillna(0)  # hold position until exit
    return z, signal

def rank_confirmed_baskets(confirmed, prices, cost_bps=10):
    """
    For each confirmed basket, refit weights on an estimation window (first
    half) and backtest on the trading window (second half) -- same
    look-ahead-avoidance split as the single-basket case. Ranks by Sharpe.
    """
    results = []
    for basket, _, stat, crit in confirmed:
        basket_prices = prices[basket].dropna()
        split = len(basket_prices) // 2
        estimation_prices = basket_prices.iloc[:split]
        trading_prices = basket_prices.iloc[split:]

        if len(estimation_prices) < 50 or len(trading_prices) < 50:
            continue

        try:
            est_result = coint_johansen(estimation_prices.values, 0, 1)
        except Exception as e:
            print(f"  Skipping {basket} (re-fit failed): {e}")
            continue

        spread_est, weights = build_spread(estimation_prices, est_result)
        trading_spread = pd.Series(
            trading_prices.values @ weights, index=trading_prices.index
        )
        z, signal = zscore_signal(trading_spread)

        spread_returns = trading_prices.pct_change().values @ weights
        spread_returns = pd.Series(spread_returns, index=trading_prices.index)
        position = signal.shift(1).fillna(0)
        strat_returns = position * spread_returns
        trades = position.diff().abs().fillna(0)
        strat_returns = strat_returns - trades * (cost_bps / 10000)
        equity = (1 + strat_returns).cumprod()

        if strat_returns.std() == 0 or strat_returns.std() != strat_returns.std():
            continue  # no variance (e.g. never traded) -> skip

        sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
        total_return = equity.iloc[-1] - 1
        running_max = equity.cummax()
        max_drawdown = ((equity - running_max) / running_max).min()
        n_trades = int(trades.sum())

        results.append({
            "basket": basket,
            "weights": weights,
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "johansen_stat": stat,
            "johansen_crit": crit,
        })

    # drop any NaN Sharpe (e.g. degenerate all-zero signal) before ranking
    results = [r for r in results if r["sharpe"] == r["sharpe"]]
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def print_ranked_results(results, top_n=10):
    print(f"\nTop {min(top_n, len(results))} baskets by Sharpe ratio:\n")
    for i, r in enumerate(results[:top_n]):
        print(f"{i+1}. {r['basket']}")
        print(f"   Sharpe: {r['sharpe']:.2f}  |  Return: {r['total_return']*100:.1f}%  "
              f"|  Max DD: {r['max_drawdown']*100:.1f}%  |  Trades: {r['n_trades']}")



    """
    Simple long/short spread backtest.
    signal: +1 = long the spread (buy low leg, sell high leg per `weights`)
            -1 = short the spread
             0 = flat
    cost_bps: round-trip transaction cost in basis points (1 bps = 0.01%)
              charged whenever the position changes.
    """
    spread_returns = prices.pct_change().values @ weights
    spread_returns = pd.Series(spread_returns, index=prices.index)

    # shift signal by 1 day: you trade on today's close using yesterday's
    # signal, so you're not using today's return to generate today's trade
    position = signal.shift(1).fillna(0)

    strat_returns = position * spread_returns

    # transaction cost applied whenever position changes
    trades = position.diff().abs().fillna(0)
    costs = trades * (cost_bps / 10000)
    strat_returns = strat_returns - costs

    equity = (1 + strat_returns).cumprod()

    sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"\nBacktest summary:")
    print(f"  Total return: {(equity.iloc[-1] - 1) * 100:.1f}%")
    print(f"  Annualized Sharpe: {sharpe:.2f}")
    print(f"  Max drawdown: {max_drawdown * 100:.1f}%")
    print(f"  Number of trades: {int(trades.sum())}")

    return equity, drawdown


if __name__ == "__main__":
    prices = fetch_prices(TICKERS, LOOKBACK)

    # --- Split: estimate cointegration on first half, trade on second half ---
    # this avoids look-ahead bias -- the trading window never influences
    # the weights used to trade it
    split = len(prices) // 2
    estimation_prices = prices.iloc[:split]
    trading_prices = prices.iloc[split:]

    result = johansen_test(estimation_prices)
    spread_est, weights = build_spread(estimation_prices, result)

    print("\nBasket weights (fit on estimation window only):")
    for t, w in zip(TICKERS, weights):
        print(f"  {t}: {w:.3f}")

    # apply the SAME frozen weights to the trading window
    trading_spread = pd.Series(
        trading_prices.values @ weights, index=trading_prices.index, name="spread"
    )
    z, signal = zscore_signal(trading_spread)

    equity, drawdown = backtest(trading_prices, weights, signal)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    trading_prices.plot(ax=axes[0], title="Prices (trading window)")
    trading_spread.plot(ax=axes[1], title="Spread (frozen weights)")
    z.plot(ax=axes[2], title="Z-score / signal")
    axes[2].axhline(2, color="r", ls="--")
    axes[2].axhline(-2, color="r", ls="--")
    equity.plot(ax=axes[3], title="Strategy equity curve (starting at 1.0)")
    plt.tight_layout()
    plt.savefig("coint_backtest.png", dpi=150)
    print("\nSaved plot to coint_backtest.png")
