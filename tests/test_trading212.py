"""Tests for Trading 212 tradeability filtering."""

from __future__ import annotations

import json

from cobasket.data.trading212 import filter_trading212_tickers, load_trading212_instruments


def test_filter_trading212_tickers_matches_symbol_and_currency() -> None:
    """Accessible stock symbols should match Yahoo notation conservatively."""
    instruments = [
        {"ticker": "AAPL_US_EQ", "shortName": "AAPL", "currencyCode": "USD", "type": "STOCK"},
        {"ticker": "SPY_US_EQ", "shortName": "SPY", "currencyCode": "USD", "type": "ETF"},
        {"ticker": "SAP_DE_EQ", "shortName": "SAP", "currencyCode": "EUR", "type": "STOCK"},
    ]
    filtered, mapping = filter_trading212_tickers(
        ["AAPL", "SPY", "SAP.DE"],
        analysis_currency="USD",
        instruments=instruments,
    )
    assert filtered == ("AAPL",)
    assert mapping == {"AAPL": "AAPL_US_EQ"}


def test_filter_trading212_tickers_normalizes_london_share_classes() -> None:
    """Yahoo hyphenated LSE classes should match dotted Trading 212 short names."""
    instruments = [
        {"ticker": "BTA_GB_EQ", "shortName": "BT.A", "currencyCode": "GBP", "type": "STOCK"},
    ]
    filtered, mapping = filter_trading212_tickers(
        ["BT-A.L"],
        analysis_currency="GBP",
        instruments=instruments,
    )
    assert filtered == ("BT-A.L",)
    assert mapping["BT-A.L"] == "BTA_GB_EQ"


def test_load_trading212_instruments_uses_fresh_cache_without_credentials(tmp_path, monkeypatch) -> None:
    """A fresh metadata cache should avoid requiring API credentials or network access."""
    payload = [{"ticker": "AAPL_US_EQ", "shortName": "AAPL", "currencyCode": "USD", "type": "STOCK"}]
    cache = tmp_path / "trading212_live_instruments.json"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("TRADING212_API_KEY", raising=False)
    monkeypatch.delenv("TRADING212_API_SECRET", raising=False)

    loaded = load_trading212_instruments(tmp_path)

    assert loaded == payload
