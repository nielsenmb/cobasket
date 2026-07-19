"""Validation of cleaned adjusted-close price tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import ValidationError


def validate_prices(prices: pd.DataFrame, *, allow_missing: bool = False) -> None:
    """Validate a price table, raising :class:`ValidationError` on failure."""
    if not isinstance(prices, pd.DataFrame):
        raise ValidationError("prices must be a pandas DataFrame")
    if prices.empty:
        raise ValidationError("price table is empty")
    if prices.shape[1] == 0:
        raise ValidationError("price table contains no ticker columns")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValidationError("price index must be a pandas DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValidationError("price dates must be monotonically increasing")
    if not prices.index.is_unique:
        raise ValidationError("price dates must be unique")
    if prices.columns.has_duplicates:
        raise ValidationError("ticker names must be unique")
    if any(not pd.api.types.is_float_dtype(dtype) for dtype in prices.dtypes):
        raise ValidationError("all price columns must use floating-point dtypes")

    values = prices.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValidationError("price table contains infinite values")
    if not allow_missing and np.isnan(values).any():
        raise ValidationError("price table contains missing values")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValidationError("price table contains no finite values")
    if (finite <= 0).any():
        raise ValidationError("all finite prices must be strictly positive")
