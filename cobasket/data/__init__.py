"""Reliable adjusted-price access with per-ticker caching."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .cleaning import align_prices, clean_prices
from .exceptions import CacheError, CobasketDataError, DownloadError, ValidationError
from .manager import DataManager, PriceMetadata
from .universe import (
    UniverseSpec,
    get_eurostoxx50_tickers,
    get_ftse100_tickers,
    get_nasdaq100_tickers,
    get_sp500_tickers,
    get_universe,
    load_custom_tickers,
)
from .validation import validate_prices

DEFAULT_CACHE_DIR = "price_cache"


def cached_download(
    tickers: Sequence[str],
    period: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch adjusted closing prices while preserving the legacy API."""
    manager = DataManager(cache_dir=cache_dir)
    return manager.prices(tickers, period=period, force_refresh=force_refresh, min_coverage=1e-12)


def fetch_prices(
    tickers: Sequence[str],
    period: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch a small basket on dates shared by every retained ticker."""
    return DataManager(cache_dir=cache_dir).prices(tickers, period=period, min_coverage=1.0)


def fetch_universe(
    tickers: Sequence[str],
    period: str,
    market_ticker: str = "SPY",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch a large universe plus a configurable market proxy.

    Parameters
    ----------
    tickers
        Asset symbols to retrieve.
    period
        Relative history specification accepted by ``yfinance``.
    market_ticker
        Common-market proxy appended to the request.
    cache_dir
        Root directory used for cache files.

    Returns
    -------
    pandas.DataFrame
        Adjusted prices after removing tickers below 90 percent coverage and
        retaining common dates.
    """
    all_tickers = [*tickers, market_ticker]
    return DataManager(cache_dir=cache_dir).prices(all_tickers, period=period, min_coverage=0.9)


__all__ = [
    "CacheError",
    "CobasketDataError",
    "DEFAULT_CACHE_DIR",
    "DataManager",
    "DownloadError",
    "PriceMetadata",
    "UniverseSpec",
    "ValidationError",
    "align_prices",
    "cached_download",
    "clean_prices",
    "fetch_prices",
    "fetch_universe",
    "get_eurostoxx50_tickers",
    "get_ftse100_tickers",
    "get_nasdaq100_tickers",
    "get_sp500_tickers",
    "get_universe",
    "load_custom_tickers",
    "validate_prices",
]
