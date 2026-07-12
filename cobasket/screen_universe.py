"""
Entry point for basket screening. Run this to search a large universe
for candidate cointegrated baskets.

Usage: python screen_universe.py
"""

import os
import pandas as pd
from coint_basket import screen_universe, rank_confirmed_baskets, print_ranked_results

TICKER_CACHE = "price_cache/sp500_tickers.csv"

def get_sp500_tickers(force_refresh=False):
    """Scrape current S&P 500 constituents from Wikipedia (cached to disk)."""
    if not force_refresh and os.path.exists(TICKER_CACHE):
        return pd.read_csv(TICKER_CACHE)["ticker"].tolist()

    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()

    os.makedirs("price_cache", exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(TICKER_CACHE, index=False)
    return tickers


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Screening {len(tickers)} tickers...")

    # distance_threshold: lower = tighter/fewer clusters, higher = looser/more
    # min_trace_stat_ratio: 1.0 = must clear the 95% critical value exactly;
    #                       raise it (e.g. 1.2) to demand a stronger signal
    confirmed, prices, Z, corr = screen_universe(
        tickers,
        period="2y",
        distance_threshold=0.8,
        min_trace_stat_ratio=1.0,
    )

    print(f"\n{len(confirmed)} confirmed cointegrated basket(s):")
    for basket, result, stat, crit in confirmed:
        print(f"  {basket}")

    if confirmed:
        print("\nBacktesting confirmed baskets (may take a while)...")
        results = rank_confirmed_baskets(confirmed, prices)
        print_ranked_results(results)
