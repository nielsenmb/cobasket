"""Ticker-universe retrieval helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import DownloadError


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
        Ignore an existing cache and retrieve the table from Wikipedia.

    Returns
    -------
    list of str
        Yahoo-compatible ticker symbols.

    Raises
    ------
    DownloadError
        If the online constituent table cannot be retrieved or interpreted.
    """
    cache_path = Path(cache_dir) / "sp500_tickers.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)["ticker"].astype(str).tolist()

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = (
            tables[0]["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .tolist()
        )
    except Exception as exc:
        raise DownloadError(f"failed to retrieve S&P 500 constituents: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers
