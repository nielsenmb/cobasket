"""Pure functions for cleaning and aligning price tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw price table to a consistent numeric representation.

    Parameters
    ----------
    prices
        Raw table with dates in the index and tickers in columns.

    Returns
    -------
    pandas.DataFrame
        Sorted, timezone-naive, floating-point price table with duplicate dates
        and columns removed.

    Notes
    -----
    Missing observations are not filled because forward filling can create
    artificial zero-return intervals.

    Raises
    ------
    TypeError
        If ``prices`` is not a pandas DataFrame.
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
    return cleaned.dropna(axis=1, how="all")


def align_prices(prices: pd.DataFrame, *, min_coverage: float = 1.0) -> pd.DataFrame:
    """Remove sparse tickers and retain dates shared by the survivors.

    Parameters
    ----------
    prices
        Clean price table that may contain missing observations.
    min_coverage
        Minimum fraction of dates on which each retained ticker must have a
        finite price.

    Returns
    -------
    pandas.DataFrame
        Complete-case price table after sparse columns are removed.

    Raises
    ------
    ValueError
        If ``min_coverage`` is outside ``(0, 1]``.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must lie in the interval (0, 1]")
    if prices.empty:
        return prices.copy()

    required = max(1, int(np.ceil(min_coverage * len(prices))))
    aligned = prices.dropna(axis=1, thresh=required)
    return aligned.dropna(axis=0, how="any")
