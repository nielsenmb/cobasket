import cobasket.cli
from cobasket.data import DataManager, fetch_prices, fetch_universe, get_sp500_tickers


def test_public_data_imports_exist():
    assert DataManager is not None
    assert callable(fetch_prices)
    assert callable(fetch_universe)
    assert callable(get_sp500_tickers)
