"""Regression tests for screening warning cleanup."""

import warnings

import numpy as np
import pandas as pd

from cobasket.cli import _quiet_statsmodels_complex_casts
from cobasket.cointegration import remove_market_factor


def test_remove_market_factor_matches_columnwise_reference() -> None:
    """Vectorized market residuals should match the previous columnwise formula."""
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(size=(200, 4)),
        columns=["SPY", "AAA", "BBB", "CCC"],
    )
    market = returns["SPY"]
    market_variance = float(np.var(market))
    expected = pd.DataFrame(
        {
            column: returns[column] - np.cov(returns[column], market)[0, 1] / market_variance * market
            for column in ["AAA", "BBB", "CCC"]
        }
    )

    actual = remove_market_factor(returns, market_col="SPY")

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)
    assert list(actual.columns) == ["AAA", "BBB", "CCC"]


def test_complex_cast_warning_context_is_targeted() -> None:
    """The CLI context should suppress only the matching statsmodels warning message."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _quiet_statsmodels_complex_casts():
            warnings.warn(
                "Casting complex values to real discards the imaginary part",
                RuntimeWarning,
            )
            warnings.warn("different warning", RuntimeWarning)

    messages = [str(item.message) for item in caught]
    assert "Casting complex values to real discards the imaginary part" not in messages
    assert "different warning" in messages
