"""Tests for built-in universe constituent-source parsing."""

from __future__ import annotations

import pandas as pd

from cobasket.data import universe


def test_nasdaq100_uses_dedicated_constituent_page(tmp_path, monkeypatch):
    """NASDAQ-100 retrieval should use the page that contains the component table."""
    seen_urls: list[str] = []

    def fake_download_tables(url: str, label: str) -> list[pd.DataFrame]:
        """Return a representative constituent table without network access."""
        seen_urls.append(url)
        assert label == "nasdaq100"
        return [pd.DataFrame({"Ticker": ["AAPL", "GOOG", "BRK.B"]})]

    monkeypatch.setattr(universe, "_download_tables", fake_download_tables)
    tickers = universe.get_nasdaq100_tickers(tmp_path, force_refresh=True)

    assert seen_urls == ["https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"]
    assert tickers == ["AAPL", "GOOG", "BRK-B"]
    assert (tmp_path / "nasdaq100_tickers.csv").exists()
