"""Regression tests for broad-universe download resilience."""

from types import SimpleNamespace

import pandas as pd
import pytest

import cobasket.data as data_module
from cobasket.data import DownloadError


def test_fetch_universe_warns_and_keeps_available_constituents(monkeypatch, tmp_path):
    """Unavailable constituents should be summarized without aborting discovery."""
    index = pd.date_range("2025-01-01", periods=3)

    class FakeManager:
        def __init__(self, cache_dir):
            self.last_metadata = None

        def prices(self, tickers, period, min_coverage):
            self.last_metadata = SimpleNamespace(failed_tickers=("STALE",))
            return pd.DataFrame({"AAA": [1.0, 1.1, 1.2], "SPY": [2.0, 2.1, 2.2]}, index=index)

    monkeypatch.setattr(data_module, "DataManager", FakeManager)

    with pytest.warns(RuntimeWarning, match="Skipped 1 unavailable"):
        prices = data_module.fetch_universe(["AAA", "STALE"], "1y", cache_dir=tmp_path)

    assert prices.columns.tolist() == ["AAA", "SPY"]


def test_fetch_universe_requires_market_proxy(monkeypatch, tmp_path):
    """A failed market proxy should fail explicitly rather than later in clustering."""
    index = pd.date_range("2025-01-01", periods=3)

    class FakeManager:
        def __init__(self, cache_dir):
            self.last_metadata = None

        def prices(self, tickers, period, min_coverage):
            self.last_metadata = SimpleNamespace(failed_tickers=("SPY",))
            return pd.DataFrame({"AAA": [1.0, 1.1, 1.2]}, index=index)

    monkeypatch.setattr(data_module, "DataManager", FakeManager)

    with pytest.raises(DownloadError, match="market proxy"):
        data_module.fetch_universe(["AAA"], "1y", cache_dir=tmp_path)
