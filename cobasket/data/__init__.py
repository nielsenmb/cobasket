"""Reliable adjusted-price access with per-ticker caching."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import pandas as pd

from .cleaning import align_prices, clean_prices
from .exceptions import CacheError, CobasketDataError, DownloadError, ValidationError
from .manager import DataManager, PriceMetadata
from .trading212 import filter_trading212_tickers, load_trading212_instruments
from .universe import (
    UniverseSpec,
    get_eurostoxx50_tickers,
    get_ftse100_tickers,
    get_ftse250_tickers,
    get_ftse350_tickers,
    get_nasdaq100_tickers,
    get_sp400_tickers,
    get_sp500_tickers,
    get_sp600_tickers,
    get_sp1500_tickers,
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
        Adjusted prices after removing unavailable or low-coverage constituents,
        retaining the market proxy, and aligning common dates.

    Raises
    ------
    DownloadError
        If the requested market proxy is unavailable.
    """
    all_tickers = [*tickers, market_ticker]
    manager = DataManager(cache_dir=cache_dir)
    prices = manager.prices(all_tickers, period=period, min_coverage=0.9)
    metadata = manager.last_metadata
    if market_ticker not in prices.columns:
        raise DownloadError(f"market proxy {market_ticker!r} could not be downloaded")
    if metadata is not None:
        failed = [ticker for ticker in metadata.failed_tickers if ticker != market_ticker]
        if failed:
            preview = ", ".join(failed[:10])
            suffix = "" if len(failed) <= 10 else f", ... (+{len(failed) - 10} more)"
            warnings.warn(
                f"Skipped {len(failed)} unavailable or insufficient-coverage universe "
                f"constituent(s): {preview}{suffix}",
                RuntimeWarning,
                stacklevel=2,
            )
    return prices


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
    "filter_trading212_tickers",
    "get_eurostoxx50_tickers",
    "get_ftse100_tickers",
    "get_ftse250_tickers",
    "get_ftse350_tickers",
    "get_nasdaq100_tickers",
    "get_sp400_tickers",
    "get_sp500_tickers",
    "get_sp600_tickers",
    "get_sp1500_tickers",
    "get_universe",
    "load_custom_tickers",
    "load_trading212_instruments",
    "validate_prices",
]
