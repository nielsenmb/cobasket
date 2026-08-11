"""Tests for pluggable discovery universes and quote conventions."""

from __future__ import annotations

import pandas as pd

from cobasket.data import universe as universe_module


def test_ftse100_adds_yahoo_london_suffix(monkeypatch, tmp_path):
    """FTSE symbols should be converted to Yahoo London tickers."""
    monkeypatch.setattr(
        universe_module,
        "_download_tables",
        lambda url, label: [pd.DataFrame({"Ticker": ["AZN", "BARC", "SHEL"]})],
    )
    tickers = universe_module.get_ftse100_tickers(tmp_path, force_refresh=True)
    assert tickers == ["AZN.L", "BARC.L", "SHEL.L"]


def test_ftse100_converts_dotted_share_class(monkeypatch, tmp_path):
    """LSE dotted symbols should use Yahoo's hyphenated class notation."""
    monkeypatch.setattr(
        universe_module,
        "_download_tables",
        lambda url, label: [pd.DataFrame({"Ticker": ["BT.A"]})],
    )
    tickers = universe_module.get_ftse100_tickers(tmp_path, force_refresh=True)
    assert tickers == ["BT-A.L"]


def test_ftse100_upgrades_stale_cached_dotted_symbol(tmp_path):
    """Old cached ``BT.A.L`` entries should be normalized and rewritten."""
    cache = tmp_path / "ftse100_tickers.csv"
    pd.DataFrame({"ticker": ["AZN.L", "BT.A.L"]}).to_csv(cache, index=False)

    tickers = universe_module.get_ftse100_tickers(tmp_path)

    assert tickers == ["AZN.L", "BT-A.L"]
    assert pd.read_csv(cache)["ticker"].tolist() == ["AZN.L", "BT-A.L"]


def test_eurostoxx_preserves_exchange_suffixes(monkeypatch, tmp_path):
    """EURO STOXX symbols already carrying Yahoo exchange suffixes are preserved."""
    monkeypatch.setattr(
        universe_module,
        "_download_tables",
        lambda url, label: [pd.DataFrame({"Ticker": ["ADS.DE", "ADYEN.AS", "AIR.PA"]})],
    )
    tickers = universe_module.get_eurostoxx50_tickers(tmp_path, force_refresh=True)
    assert tickers == ["ADS.DE", "ADYEN.AS", "AIR.PA"]


def test_builtin_universe_currency_conventions(monkeypatch, tmp_path):
    """Built-in markets should expose explicit analysis currencies and price scales."""
    monkeypatch.setattr(universe_module, "get_ftse100_tickers", lambda *args, **kwargs: ["AZN.L"])
    monkeypatch.setattr(universe_module, "get_sp500_tickers", lambda *args, **kwargs: ["AAPL"])
    uk = universe_module.get_universe("ftse100", cache_dir=tmp_path)
    us = universe_module.get_universe("sp500", cache_dir=tmp_path)
    assert (uk.quote_currency, uk.analysis_currency, uk.price_scale) == ("GBp", "GBP", 0.01)
    assert (us.quote_currency, us.analysis_currency, us.price_scale) == ("USD", "USD", 1.0)


def test_custom_universe_requires_currency_and_market_proxy(tmp_path):
    """Custom discovery should not silently guess market or currency conventions."""
    path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": ["AAA", "BBB"]}).to_csv(path, index=False)
    try:
        universe_module.get_universe("custom", custom_path=path)
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("custom universe should require explicit conventions")
