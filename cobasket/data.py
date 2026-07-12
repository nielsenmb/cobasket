"""Price data fetching with on-disk caching."""

import hashlib
import os

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = "price_cache"


def _cache_path(tickers, period, cache_dir):
    """Hash the ticker list + period into a stable filename."""
    key = ",".join(sorted(tickers)) + f"|{period}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"{digest}.parquet")


def cached_download(tickers, period, cache_dir=DEFAULT_CACHE_DIR, force_refresh=False):
    """
    Thin caching wrapper around yf.download. Caches to disk as parquet,
    keyed on the exact ticker set + period requested. Delete the cache
    directory or pass force_refresh=True to bust the cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(tickers, period, cache_dir)

    if not force_refresh and os.path.exists(path):
        print(f"  [cache hit] loading {len(tickers)} tickers from {path}")
        return pd.read_parquet(path)

    print(f"  [cache miss] downloading {len(tickers)} tickers...")
    data = yf.download(tickers, period=period)["Close"]
    data.to_parquet(path)
    return data


def fetch_prices(tickers, period, cache_dir=DEFAULT_CACHE_DIR):
    """Fetch a small, specific basket of tickers."""
    return cached_download(tickers, period, cache_dir).dropna()


def fetch_universe(tickers, period, market_ticker="SPY", cache_dir=DEFAULT_CACHE_DIR):
    """Fetch a universe of tickers plus a market proxy for factor removal."""
    all_tickers = list(tickers) + [market_ticker]
    data = cached_download(all_tickers, period, cache_dir)
    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))  # drop sparse tickers
    data = data.dropna()
    return data


def get_sp500_tickers(cache_dir=DEFAULT_CACHE_DIR, force_refresh=False):
    """Scrape current S&P 500 constituents from Wikipedia (cached to disk)."""
    cache_path = os.path.join(cache_dir, "sp500_tickers.csv")

    if not force_refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path)["ticker"].tolist()

    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()

    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers
