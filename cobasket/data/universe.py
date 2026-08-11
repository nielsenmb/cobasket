"""Ticker-universe retrieval helpers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
import warnings

import pandas as pd

from .exceptions import DownloadError


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _download_sp500_table() -> pd.DataFrame:
    """Download the current S&P 500 constituent table.

    Returns
    -------
    pandas.DataFrame
        Parsed constituent table containing a ``Symbol`` column.

    Raises
    ------
    DownloadError
        If the page cannot be downloaded or interpreted.
    """
    request = Request(
        SP500_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/145.0 Safari/537.36 Cobasket/0.5"
            )
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
        tables = pd.read_html(StringIO(html))
    except Exception as exc:
        raise DownloadError(f"failed to retrieve S&P 500 constituents: {exc}") from exc

    for table in tables:
        if "Symbol" in table.columns:
            return table
    raise DownloadError("S&P 500 constituent page did not contain a Symbol column")


def get_sp500_tickers(
    cache_dir: str | Path = "price_cache",
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Return current S&P 500 symbols using a local CSV cache.

    Parameters
    ----------
    cache_dir
        Root directory containing the constituent cache.
    force_refresh
        Attempt to refresh the cached constituent table before returning it.

    Returns
    -------
    list of str
        Yahoo-compatible ticker symbols.

    Raises
    ------
    DownloadError
        If no usable cache exists and the online constituent table cannot be
        retrieved or interpreted.

    Notes
    -----
    A failed forced refresh falls back to an existing cache with a warning. This
    keeps screening usable during temporary upstream HTTP failures while making
    the stale-universe fallback explicit.
    """
    cache_path = Path(cache_dir) / "sp500_tickers.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)["ticker"].astype(str).tolist()

    try:
        table = _download_sp500_table()
        tickers = (
            table["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .tolist()
        )
    except DownloadError:
        if cache_path.exists():
            warnings.warn(
                "Could not refresh S&P 500 constituents; using the existing cached universe.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.read_csv(cache_path)["ticker"].astype(str).tolist()
        raise

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers
