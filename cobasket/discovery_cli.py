"""Command-line interface for persistence-aware basket discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

from cobasket.data.universe import get_universe
from cobasket.discovery import discover_baskets
from cobasket.evidence import BasketWatchlist


def _save_watchlist(table, path: Path, top_n: int, universe, *, include_borderline: bool = False) -> int:
    """Save ranked discovery baskets and attach universe metadata.

    Parameters
    ----------
    table
        Discovery result table.
    path
        Watchlist JSON destination.
    top_n
        Maximum number of baskets to save.
    universe
        Resolved universe specification.
    include_borderline
        Include ``borderline`` baskets as well as ``promising`` baskets.

    Returns
    -------
    int
        Number of saved baskets.
    """
    if table.empty:
        selected = table
    elif include_borderline:
        selected = table.loc[table["status"].isin(["promising", "borderline"])].head(top_n)
    else:
        selected = table.loc[table["status"] == "promising"].head(top_n)
    if selected.empty:
        return 0
    baskets = tuple(tuple(item) for item in selected["basket"])
    BasketWatchlist(
        baskets=baskets,
        name=f"Cobasket persistent discovery ({universe.name})",
    ).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["universe_metadata"] = {
        "name": universe.name,
        "market_ticker": universe.market_ticker,
        "quote_currency": universe.quote_currency,
        "analysis_currency": universe.analysis_currency,
        "price_scale": universe.price_scale,
        "single_currency": True,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(baskets)


def main() -> None:
    """Run persistence-aware discovery for a built-in or custom universe."""
    parser = argparse.ArgumentParser(
        description="Discover baskets using current cointegration, historical persistence, weight stability, and backtests."
    )
    parser.add_argument(
        "--universe",
        default="sp500",
        choices=["sp500", "nasdaq100", "ftse100", "eurostoxx50", "custom"],
    )
    parser.add_argument("--tickers-file", help="CSV/text ticker list for --universe custom")
    parser.add_argument("--market-ticker", help="Market proxy for --universe custom")
    parser.add_argument("--currency", help="Single quote currency for --universe custom")
    parser.add_argument("--price-scale", type=float, default=1.0, help="Quote-to-currency scale for custom universe")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--distance-threshold", type=float, default=0.8)
    parser.add_argument("--min-trace-stat-ratio", type=float, default=1.0)
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--min-persistence", type=float, default=0.15, help="Loose persistence floor below which candidates are rejected")
    parser.add_argument("--min-weight-stability", type=float, default=0.60, help="Loose weight-stability floor below which candidates are rejected")
    parser.add_argument("--promising-persistence", type=float, default=0.30)
    parser.add_argument("--promising-evaluations", type=int, default=15)
    parser.add_argument("--promising-weight-stability", type=float, default=0.80)
    parser.add_argument("--include-borderline", action="store_true", help="Also export borderline baskets to the watchlist")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--watchlist-out", default="discovered_watchlist.json")
    parser.add_argument("--table-out", default="discovery_results.csv")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    universe = get_universe(
        args.universe,
        force_refresh=args.force_refresh,
        custom_path=args.tickers_file,
        custom_market_ticker=args.market_ticker,
        custom_currency=args.currency,
        custom_price_scale=args.price_scale,
    )
    print(
        f"Discovering baskets across {len(universe.tickers)} {universe.name} tickers "
        f"({universe.analysis_currency}; market proxy {universe.market_ticker})..."
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
        result = discover_baskets(
            universe.tickers,
            period=args.period,
            market_ticker=universe.market_ticker,
            distance_threshold=args.distance_threshold,
            min_trace_ratio=args.min_trace_stat_ratio,
            cost_bps=args.cost_bps,
            train_window=args.train_window,
            horizon=args.horizon,
            step=args.step,
            min_persistence=args.min_persistence,
            min_weight_stability=args.min_weight_stability,
            promising_persistence=args.promising_persistence,
            promising_evaluations=args.promising_evaluations,
            promising_weight_stability=args.promising_weight_stability,
        )

    table = result.table.copy()
    if table.empty:
        print("No currently cointegrated candidates completed the discovery backtest.")
        return
    display = table.copy()
    display["basket"] = display["basket"].apply(lambda item: ", ".join(item))
    print("\nPersistence-aware discovery results:")
    print(display.head(args.top_n).to_string(index=False))

    table_path = Path(args.table_out).expanduser().resolve()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(table_path, index=False)
    watchlist_path = Path(args.watchlist_out).expanduser().resolve()
    count = _save_watchlist(
        table,
        watchlist_path,
        args.top_n,
        universe,
        include_borderline=args.include_borderline,
    )
    print(f"\nSaved detailed discovery table to {table_path}")
    if count:
        label = "promising/borderline" if args.include_borderline else "promising"
        print(f"Saved {count} {label} basket(s) to {watchlist_path}")
    else:
        print("No promising basket passed the discovery thresholds; watchlist was not created.")


if __name__ == "__main__":
    main()
