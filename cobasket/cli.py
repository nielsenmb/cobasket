"""Command-line entry points: cobasket-backtest, cobasket-screen, cobasket-pca-screen."""

import argparse

import matplotlib.pyplot as plt

from cobasket.backtest import backtest_single_basket, print_ranked_results, rank_confirmed_baskets
from cobasket.cointegration import screen_universe
from cobasket.data import fetch_prices, get_sp500_tickers
from cobasket.evidence import BasketWatchlist
from cobasket.pca import cluster_by_loadings, pca_screen_universe
from cobasket.plotting import plot_dendrogram, plot_loadings_2d, plot_scree


def _save_ranked_watchlist(results, path: str, top_n: int, name: str) -> None:
    """Save the highest-ranked screened baskets as a persistent watchlist.

    Parameters
    ----------
    results
        Ranked basket dictionaries returned by :func:`rank_confirmed_baskets`.
    path
        Output JSON path.
    top_n
        Maximum number of ranked baskets to save.
    name
        Human-readable watchlist name.

    Returns
    -------
    None
    """
    baskets = tuple(tuple(item["basket"]) for item in results[:top_n])
    if not baskets:
        print(f"\nNo successfully backtested baskets were available for {path}.")
        return
    output = BasketWatchlist(baskets=baskets, name=name).save(path)
    print(f"\nSaved {len(baskets)} ranked basket(s) to watchlist {output}")


def backtest_cmd():
    """Run the single-basket backtest command-line interface."""
    parser = argparse.ArgumentParser(description="Backtest a single named basket of tickers.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. XOM CVX COP OXY")
    parser.add_argument("--period", default="2y", help="History length, e.g. 2y, 5y (default: 2y)")
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip cost in bps (default: 10)")
    parser.add_argument("--plot-out", default="coint_backtest.png", help="Output plot path")
    args = parser.parse_args()

    prices = fetch_prices(args.tickers, args.period)
    result = backtest_single_basket(prices, args.tickers, cost_bps=args.cost_bps)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    result["trading_prices"].plot(ax=axes[0], title="Prices (trading window)")
    result["trading_spread"].plot(ax=axes[1], title="Spread (frozen weights)")
    result["z"].plot(ax=axes[2], title="Z-score / signal")
    axes[2].axhline(2, color="r", ls="--")
    axes[2].axhline(-2, color="r", ls="--")
    result["equity"].plot(ax=axes[3], title="Strategy equity curve (starting at 1.0)")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    print(f"\nSaved plot to {args.plot_out}")


def screen_cmd():
    """Run the correlation-clustering universe screen."""
    parser = argparse.ArgumentParser(description="Screen a large universe for cointegrated baskets.")
    parser.add_argument("--universe", default="sp500", choices=["sp500"], help="Ticker universe to screen")
    parser.add_argument("--period", default="2y", help="History length (default: 2y)")
    parser.add_argument("--distance-threshold", type=float, default=0.8, help="Hierarchical clustering cut height")
    parser.add_argument("--min-trace-stat-ratio", type=float, default=1.0, help="Multiple of the 95%% critical value required")
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip cost in bps")
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked baskets to print/save")
    parser.add_argument("--watchlist-out", help="Optional JSON path for the top ranked baskets")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh prices and S&P 500 constituents")
    args = parser.parse_args()

    tickers = get_sp500_tickers(force_refresh=args.force_refresh)
    print(f"Screening {len(tickers)} tickers...")
    confirmed, prices, _, _ = screen_universe(
        tickers,
        period=args.period,
        distance_threshold=args.distance_threshold,
        min_trace_stat_ratio=args.min_trace_stat_ratio,
    )

    print(f"\n{len(confirmed)} confirmed cointegrated basket(s):")
    for basket, _, _, _ in confirmed:
        print(f"  {basket}")

    if confirmed:
        print("\nBacktesting confirmed baskets (may take a while)...")
        results = rank_confirmed_baskets(confirmed, prices, cost_bps=args.cost_bps)
        print_ranked_results(results, top_n=args.top_n)
        if args.watchlist_out:
            _save_ranked_watchlist(results, args.watchlist_out, args.top_n, f"Cobasket correlation screen ({args.period})")
    elif args.watchlist_out:
        print(f"\nNo confirmed baskets; watchlist {args.watchlist_out} was not created.")


def pca_screen_cmd():
    """Run the PCA-loading universe screen."""
    parser = argparse.ArgumentParser(description="PCA-based screen for cointegrated baskets, with diagnostic plots.")
    parser.add_argument("--universe", default="sp500", choices=["sp500"], help="Ticker universe to screen")
    parser.add_argument("--period", default="2y", help="History length")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCs to compute")
    parser.add_argument("--n-remove", type=int, default=1, help="Number of top PCs to remove")
    parser.add_argument("--n-components-for-clustering", type=int, default=5, help="PCs used for clustering")
    parser.add_argument("--distance-threshold", type=float, default=1.5, help="Clustering cut height")
    parser.add_argument("--min-trace-stat-ratio", type=float, default=1.0, help="Multiple of the 95%% critical value required")
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip cost in bps")
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked baskets to print/save")
    parser.add_argument("--watchlist-out", help="Optional JSON path for the top ranked baskets")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh prices and S&P 500 constituents")
    parser.add_argument("--plot-dir", default=".", help="Directory to save diagnostic plots")
    args = parser.parse_args()

    tickers = get_sp500_tickers(force_refresh=args.force_refresh)
    print(f"Screening {len(tickers)} tickers via PCA...")
    confirmed, prices, pca, loadings, _ = pca_screen_universe(
        tickers,
        period=args.period,
        n_components=args.n_components,
        n_remove=args.n_remove,
        n_components_for_clustering=args.n_components_for_clustering,
        distance_threshold=args.distance_threshold,
        min_trace_stat_ratio=args.min_trace_stat_ratio,
    )

    print(f"\nSaved {plot_scree(pca, path=f'{args.plot_dir}/pca_scree.png')}")
    candidate_baskets, linkage_matrix = cluster_by_loadings(loadings, args.n_components_for_clustering, args.distance_threshold)
    basket_labels = {ticker: cid for cid, members in enumerate(candidate_baskets) for ticker in members}
    print(f"Saved {plot_loadings_2d(loadings, pc_x='PC2', pc_y='PC3', basket_labels=basket_labels, path=f'{args.plot_dir}/pca_loadings.png')}")
    print(f"Saved {plot_dendrogram(linkage_matrix, loadings.index, path=f'{args.plot_dir}/cluster_dendrogram.png', distance_threshold=args.distance_threshold)}")

    print(f"\n{len(confirmed)} confirmed cointegrated basket(s):")
    for basket, _, _, _ in confirmed:
        print(f"  {basket}")

    if confirmed:
        print("\nBacktesting confirmed baskets (may take a while)...")
        results = rank_confirmed_baskets(confirmed, prices, cost_bps=args.cost_bps)
        print_ranked_results(results, top_n=args.top_n)
        if args.watchlist_out:
            _save_ranked_watchlist(results, args.watchlist_out, args.top_n, f"Cobasket PCA screen ({args.period})")
    elif args.watchlist_out:
        print(f"\nNo confirmed baskets; watchlist {args.watchlist_out} was not created.")
