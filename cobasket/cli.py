"""Command-line entry points: cobasket-backtest, cobasket-screen, cobasket-pca-screen."""

import argparse

import matplotlib.pyplot as plt

from cobasket.backtest import backtest_single_basket, print_ranked_results, rank_confirmed_baskets
from cobasket.cointegration import screen_universe
from cobasket.data import fetch_prices, get_sp500_tickers
from cobasket.pca import cluster_by_loadings, pca_screen_universe
from cobasket.plotting import plot_dendrogram, plot_loadings_2d, plot_scree


def backtest_cmd():
    """Run the single-basket backtest command-line interface.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
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
    """Run the correlation-clustering universe screen.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Screen a large universe for cointegrated baskets.")
    parser.add_argument(
        "--universe", default="sp500", choices=["sp500"],
        help="Ticker universe to screen (default: sp500)",
    )
    parser.add_argument("--period", default="2y", help="History length (default: 2y)")
    parser.add_argument(
        "--distance-threshold", type=float, default=0.8,
        help="Hierarchical clustering cut height, 0-2 (default: 0.8)",
    )
    parser.add_argument(
        "--min-trace-stat-ratio", type=float, default=1.0,
        help="Multiple of the 95%% critical value a basket must clear (default: 1.0)",
    )
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip cost in bps (default: 10)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked baskets to print")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass the disk cache")
    args = parser.parse_args()

    tickers = get_sp500_tickers(force_refresh=args.force_refresh)
    print(f"Screening {len(tickers)} tickers...")

    confirmed, prices, Z, corr = screen_universe(
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


def pca_screen_cmd():
    """Run the PCA-loading universe screen.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="PCA-based screen for cointegrated baskets, with diagnostic plots."
    )
    parser.add_argument(
        "--universe", default="sp500", choices=["sp500"],
        help="Ticker universe to screen (default: sp500)",
    )
    parser.add_argument("--period", default="2y", help="History length (default: 2y)")
    parser.add_argument(
        "--n-components", type=int, default=10,
        help="Number of PCs to compute (default: 10)",
    )
    parser.add_argument(
        "--n-remove", type=int, default=1,
        help="Number of top PCs to remove as 'market' factors before clustering residuals "
             "(default: 1 -- just PC1). Raise to 2-3 if the scree/loadings plots suggest "
             "PC2/PC3 are also broad sector factors rather than idiosyncratic structure.",
    )
    parser.add_argument(
        "--n-components-for-clustering", type=int, default=5,
        help="Number of PCs used as the clustering feature space (default: 5)",
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=1.5,
        help="Hierarchical clustering cut height in loading-space (default: 1.5). "
             "Check cluster_dendrogram.png and re-run with a different value if clusters "
             "look too big or too fragmented.",
    )
    parser.add_argument(
        "--min-trace-stat-ratio", type=float, default=1.0,
        help="Multiple of the 95%% critical value a basket must clear (default: 1.0)",
    )
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip cost in bps (default: 10)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked baskets to print")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass the disk cache")
    parser.add_argument("--plot-dir", default=".", help="Directory to save diagnostic plots")
    args = parser.parse_args()

    tickers = get_sp500_tickers(force_refresh=args.force_refresh)
    print(f"Screening {len(tickers)} tickers via PCA...")

    confirmed, prices, pca, loadings, scores = pca_screen_universe(
        tickers,
        period=args.period,
        n_components=args.n_components,
        n_remove=args.n_remove,
        n_components_for_clustering=args.n_components_for_clustering,
        distance_threshold=args.distance_threshold,
        min_trace_stat_ratio=args.min_trace_stat_ratio,
    )

    # --- diagnostic plots, saved regardless of whether any basket confirms ---
    scree_path = plot_scree(pca, path=f"{args.plot_dir}/pca_scree.png")
    print(f"\nSaved {scree_path}")
    print("  Look at this first: the 'elbow' tells you how many PCs carry real")
    print("  structure vs noise. PC1's bar height ~= how market-dominated returns are.")

    # re-run clustering here just to get cluster ids for coloring the loadings plot
    candidate_baskets, Z = cluster_by_loadings(
        loadings, args.n_components_for_clustering, args.distance_threshold
    )
    basket_labels = {}
    for cid, members in enumerate(candidate_baskets):
        for t in members:
            basket_labels[t] = cid

    loadings_path = plot_loadings_2d(
        loadings, pc_x="PC2", pc_y="PC3", basket_labels=basket_labels,
        path=f"{args.plot_dir}/pca_loadings.png",
    )
    print(f"Saved {loadings_path}")
    print("  Points colored by cluster. Tight, separated blobs = good candidate")
    print("  baskets. A shapeless cloud means loading-space isn't separating well --")
    print("  try adjusting --n-components-for-clustering or --distance-threshold.")

    dendro_path = plot_dendrogram(
        Z, loadings.index, path=f"{args.plot_dir}/cluster_dendrogram.png",
        distance_threshold=args.distance_threshold,
    )
    print(f"Saved {dendro_path}")
    print("  The red dashed line is your --distance-threshold cut. Clusters are")
    print("  branches that merge below that line.")

    print(f"\n{len(confirmed)} confirmed cointegrated basket(s):")
    for basket, _, _, _ in confirmed:
        print(f"  {basket}")

    if confirmed:
        print("\nBacktesting confirmed baskets (may take a while)...")
        results = rank_confirmed_baskets(confirmed, prices, cost_bps=args.cost_bps)
        print_ranked_results(results, top_n=args.top_n)
