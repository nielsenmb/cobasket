import numpy as np
import pandas as pd

from cobasket.data import align_prices, clean_prices


def test_clean_prices_sorts_deduplicates_and_normalizes_columns():
    index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    raw = pd.DataFrame({" aapl ": [102, 100, 101]}, index=index)

    result = clean_prices(raw)

    assert result.columns.tolist() == ["AAPL"]
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.loc[pd.Timestamp("2024-01-01"), "AAPL"] == 101.0


def test_align_prices_drops_sparse_tickers_then_incomplete_rows():
    index = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": [1, 2, np.nan, 4], "C": [1, np.nan, np.nan, 4]},
        index=index,
        dtype=float,
    )

    result = align_prices(prices, min_coverage=0.75)

    assert result.columns.tolist() == ["A", "B"]
    assert len(result) == 3
