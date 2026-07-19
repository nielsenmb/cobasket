"""Pure functions for cleaning price tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a consistently indexed, numeric price table.

    The function deliberately does not fill missing observations. Filling prices
    can create artificial zero-return intervals, so alignment policy is left to
    the caller.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce", utc=True)
    cleaned = cleaned.loc[~cleaned.index.isna()]
    cleaned.index = cleaned.index.tz_convert(None)
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")].sort_index()

    cleaned.columns = [str(column).strip().upper() for column in cleaned.columns]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce").astype(float)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned


def align_prices(prices: pd.DataFrame, *, min_coverage: float = 1.0) -> pd.DataFrame:
    """Drop sparse tickers and return rows shared by the remaining tickers.

    ``min_coverage`` is the fraction of dates on which a ticker must have a
    valid price. A value of 0.9 means a ticker may be missing at most 10% of the
    available dates. After sparse columns are removed, rows containing any
    remaining missing value are dropped.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must lie in the interval (0, 1]")
    if prices.empty:
        return prices.copy()

    required = max(1, int(np.ceil(min_coverage * len(prices))))
    aligned = prices.dropna(axis=1, thresh=required)
    return aligned.dropna(axis=0, how="any")
