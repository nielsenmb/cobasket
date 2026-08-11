"""Tests for FX-aware portfolio valuation helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cobasket.fx import latest_fx_quote, normalize_currency, yahoo_fx_ticker
from cobasket.workflow import PortfolioConfig, _watchlist_currency_metadata


class FakeDataManager:
    """Minimal price provider used by FX tests."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.requests: list[tuple[str, ...]] = []

    def prices(self, tickers, **kwargs):
        """Return a one-row price table for the requested FX symbol."""
        symbols = tuple(tickers)
        self.requests.append(symbols)
        return pd.DataFrame({symbols[0]: [self.value]}, index=pd.to_datetime(["2026-08-11"]))


def test_normalize_currency() -> None:
    """Currency codes should be normalized and validated."""
    assert normalize_currency(" gbp ") == "GBP"
    with pytest.raises(ValueError):
        normalize_currency("pounds")


def test_yahoo_fx_ticker() -> None:
    """Direct Yahoo FX symbols should follow source-target ordering."""
    assert yahoo_fx_ticker("USD", "GBP") == "USDGBP=X"
    assert yahoo_fx_ticker("EUR", "USD") == "EURUSD=X"


def test_same_currency_fx_does_not_download() -> None:
    """No data request should be needed when source and base currencies match."""
    manager = FakeDataManager(0.8)
    quote = latest_fx_quote(manager, "GBP", "GBP")
    assert quote.rate == 1.0
    assert quote.ticker is None
    assert manager.requests == []


def test_cross_currency_fx_downloads_direct_pair() -> None:
    """FX conversion should return target-currency units per source unit."""
    manager = FakeDataManager(0.77)
    quote = latest_fx_quote(manager, "USD", "GBP")
    assert quote.ticker == "USDGBP=X"
    assert quote.rate == pytest.approx(0.77)
    assert manager.requests == [("USDGBP=X",)]


def test_watchlist_currency_metadata_reads_london_price_scale(tmp_path) -> None:
    """Discovery metadata should expose GBP and the GBp-to-GBP scale."""
    path = tmp_path / "watchlist.json"
    path.write_text(
        json.dumps(
            {
                "baskets": [["AZN.L", "GSK.L"]],
                "name": "UK",
                "universe_metadata": {
                    "analysis_currency": "GBP",
                    "quote_currency": "GBp",
                    "price_scale": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )
    assert _watchlist_currency_metadata(path) == ("GBP", 0.01)


def test_portfolio_config_base_currency_is_optional_and_validated() -> None:
    """Legacy configs remain valid while explicit base currencies are normalized."""
    legacy = PortfolioConfig(holdings={})
    explicit = PortfolioConfig(holdings={}, base_currency="gbp")
    assert legacy.base_currency is None
    assert explicit.base_currency == "GBP"
    with pytest.raises(ValueError):
        PortfolioConfig(holdings={}, base_currency="sterling")
