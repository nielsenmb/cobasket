"""Reliable adjusted-price data access with per-ticker caching."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .cleaning import align_prices, clean_prices
from .exceptions import CacheError, CobasketDataError, DownloadError, ValidationError
from .manager import DataManager, PriceMetadata
from .universe import get_sp500_tickers
from .validation import validate_prices

DEFAULT_CACHE_DIR = "price_cache"


def cached_download(
    tickers: Sequence[str],
    period: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Compatibility wrapper returning unaligned per-ticker adjusted closes."""
    manager = DataManager(cache_dir=cache_dir)
    return manager.prices(
        tickers,
        period=period,
        force_refresh=force_refresh,
        min_coverage=1e-12,
    )


def fetch_prices(
    tickers: Sequence[str],
    period: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch a small basket on its common trading dates."""
    return DataManager(cache_dir=cache_dir).prices(tickers, period=period, min_coverage=1.0)


def fetch_universe(
    tickers: Sequence[str],
    period: str,
    market_ticker: str = "SPY",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch a universe plus market proxy, dropping tickers below 90% coverage."""
    all_tickers = [*tickers, market_ticker]
    return DataManager(cache_dir=cache_dir).prices(
        all_tickers, period=period, min_coverage=0.9
    )


__all__ = [
    "CacheError",
    "CobasketDataError",
    "DEFAULT_CACHE_DIR",
    "DataManager",
    "DownloadError",
    "PriceMetadata",
    "ValidationError",
    "align_prices",
    "cached_download",
    "clean_prices",
    "fetch_prices",
    "fetch_universe",
    "get_sp500_tickers",
    "validate_prices",
]
