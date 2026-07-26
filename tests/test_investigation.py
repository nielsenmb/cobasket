"""Tests for basket investigation diagnostics."""

import numpy as np
import pandas as pd
import pytest

from cobasket.investigation import investigate_basket


def synthetic_cointegrated_prices(n: int = 320) -> pd.DataFrame:
    """Create positive price series sharing one stochastic trend."""
    rng = np.random.default_rng(12)
    common = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    stationary = rng.normal(0.0, 0.8, n)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "AAA": common + stationary,
            "BBB": 0.8 * common - stationary + 20.0,
        },
        index=index,
    )


def test_investigate_basket_returns_consistent_diagnostics():
    """Investigation fields should preserve alignment and normalized weights."""
    prices = synthetic_cointegrated_prices()
    result = investigate_basket(prices, window=40)
    assert result.tickers == ("AAA", "BBB")
    assert result.normalized_prices.iloc[0].to_numpy() == pytest.approx([1.0, 1.0])
    assert np.abs(result.weights).sum() == pytest.approx(1.0)
    assert result.spread.index.equals(prices.index)
    assert np.isfinite(result.latest_z_score)
    assert result.trace_ratio > 0.0


def test_investigate_basket_rejects_short_history():
    """The rolling standardization needs more observations than its window."""
    with pytest.raises(ValueError, match="longer"):
        investigate_basket(synthetic_cointegrated_prices(20), window=20)
