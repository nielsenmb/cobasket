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


def test_ftse250_extracts_epic_from_share_label(monkeypatch, tmp_path):
    """FTSE 250 share labels should expose the EPIC code used by Yahoo."""
    monkeypatch.setattr(
        universe_module,
        "_download_tables",
        lambda url, label: [
            pd.DataFrame({"Share": ["3i Infrastructure (3IN)", "Ao World (AO.)"]})
        ],
    )
    tickers = universe_module.get_ftse250_tickers(tmp_path, force_refresh=True)
    assert tickers == ["3IN.L", "AO-.L"]


def test_ftse350_combines_large_and_mid_cap_constituents(monkeypatch, tmp_path):
    """FTSE 350 should de-duplicate the FTSE 100 and FTSE 250 component lists."""
    monkeypatch.setattr(universe_module, "get_ftse100_tickers", lambda *args, **kwargs: ["AZN.L", "SHEL.L"])
    monkeypatch.setattr(universe_module, "get_ftse250_tickers", lambda *args, **kwargs: ["SHEL.L", "ABDN.L"])
    assert universe_module.get_ftse350_tickers(tmp_path) == ["AZN.L", "SHEL.L", "ABDN.L"]


def test_sp1500_combines_large_mid_and_small_cap_constituents(monkeypatch, tmp_path):
    """S&P Composite 1500 should be the de-duplicated 500/400/600 union."""
    monkeypatch.setattr(universe_module, "get_sp500_tickers", lambda *args, **kwargs: ["AAPL", "MSFT"])
    monkeypatch.setattr(universe_module, "get_sp400_tickers", lambda *args, **kwargs: ["MSFT", "AA"])
    monkeypatch.setattr(universe_module, "get_sp600_tickers", lambda *args, **kwargs: ["AA", "ABM"])
    assert universe_module.get_sp1500_tickers(tmp_path) == ["AAPL", "MSFT", "AA", "ABM"]


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
    monkeypatch.setattr(universe_module, "get_ftse350_tickers", lambda *args, **kwargs: ["AZN.L", "ABDN.L"])
    monkeypatch.setattr(universe_module, "get_sp500_tickers", lambda *args, **kwargs: ["AAPL"])
    monkeypatch.setattr(universe_module, "get_sp1500_tickers", lambda *args, **kwargs: ["AAPL", "AA"])
    uk = universe_module.get_universe("ftse100", cache_dir=tmp_path)
    uk_broad = universe_module.get_universe("ftse350", cache_dir=tmp_path)
    us = universe_module.get_universe("sp500", cache_dir=tmp_path)
    us_broad = universe_module.get_universe("sp1500", cache_dir=tmp_path)
    assert (uk.quote_currency, uk.analysis_currency, uk.price_scale) == ("GBp", "GBP", 0.01)
    assert (uk_broad.market_ticker, uk_broad.analysis_currency) == ("^FTLC", "GBP")
    assert (us.quote_currency, us.analysis_currency, us.price_scale) == ("USD", "USD", 1.0)
    assert (us_broad.market_ticker, us_broad.analysis_currency) == ("^SP1500", "USD")


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
