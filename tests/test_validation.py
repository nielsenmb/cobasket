import numpy as np
import pandas as pd
import pytest

from cobasket.data import ValidationError, validate_prices


def valid_prices():
    return pd.DataFrame(
        {"AAPL": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2)
    )


def test_valid_prices_pass():
    validate_prices(valid_prices())


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_non_positive_prices_fail(bad_value):
    prices = valid_prices()
    prices.iloc[0, 0] = bad_value
    with pytest.raises(ValidationError, match="strictly positive"):
        validate_prices(prices)


def test_missing_prices_fail_by_default():
    prices = valid_prices()
    prices.iloc[0, 0] = np.nan
    with pytest.raises(ValidationError, match="missing"):
        validate_prices(prices)


def test_duplicate_dates_fail():
    prices = valid_prices()
    prices.index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValidationError, match="unique"):
        validate_prices(prices)
