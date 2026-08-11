from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobasket.data import DataManager, DownloadError


def yahoo_frame(tickers, values=(100.0, 101.0, 102.0)):
    if isinstance(tickers, str):
        tickers = [tickers]
    index = pd.date_range("2024-01-01", periods=len(values))
    columns = pd.MultiIndex.from_product([["Close"], list(tickers)])
    array = np.repeat(np.asarray(values)[:, None], len(tickers), axis=1)
    return pd.DataFrame(array, index=index, columns=columns)


def test_single_and_multiple_tickers_are_combined(tmp_path):
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs["tickers"])
        return yahoo_frame(kwargs["tickers"])

    manager = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    prices = manager.prices(["aapl", "MSFT"], period="1y")

    assert prices.columns.tolist() == ["AAPL", "MSFT"]
    assert calls == [["AAPL", "MSFT"]]
    assert manager.last_metadata.downloaded_tickers == ("AAPL", "MSFT")


def test_cache_is_per_ticker_and_reused(tmp_path):
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs["tickers"])
        return yahoo_frame(kwargs["tickers"])

    first = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    first.prices(["AAPL", "MSFT"], period="1y")

    second = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    second.prices(["AAPL", "NVDA"], period="1y")

    assert calls == [["AAPL", "MSFT"], "NVDA"]
    assert second.last_metadata.cache_hits == ("AAPL",)
    assert second.last_metadata.downloaded_tickers == ("NVDA",)
    assert len(list(Path(tmp_path).glob("prices/*/*.parquet"))) == 3


def test_failed_ticker_does_not_destroy_successful_data(tmp_path):
    def downloader(**kwargs):
        raw = yahoo_frame(kwargs["tickers"])
        if isinstance(kwargs["tickers"], list) and "BAD" in kwargs["tickers"]:
            raw[("Close", "BAD")] = np.nan
        return raw

    manager = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    prices = manager.prices(["AAPL", "BAD"], period="1y")

    assert prices.columns.tolist() == ["AAPL"]
    assert manager.last_metadata.failed_tickers == ("BAD",)


def test_failed_batch_is_retried_per_ticker(tmp_path):
    """One stale symbol should not discard the other symbols in its failed batch."""
    calls = []

    def downloader(**kwargs):
        tickers = kwargs["tickers"]
        calls.append(tickers)
        if isinstance(tickers, list):
            raise RuntimeError("provider rejected batch")
        if tickers == "BAD":
            raise RuntimeError("symbol unavailable")
        return yahoo_frame(tickers)

    manager = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    prices = manager.prices(["AAPL", "BAD", "MSFT"], period="1y")

    assert prices.columns.tolist() == ["AAPL", "MSFT"]
    assert calls == [["AAPL", "BAD", "MSFT"], "AAPL", "BAD", "MSFT"]
    assert manager.last_metadata.downloaded_tickers == ("AAPL", "MSFT")
    assert manager.last_metadata.failed_tickers == ("BAD",)


def test_all_failed_tickers_raise(tmp_path):
    manager = DataManager(
        tmp_path, downloader=lambda **kwargs: pd.DataFrame(), cache_max_age_days=None
    )
    with pytest.raises(DownloadError, match="no usable price data"):
        manager.prices(["BAD"], period="1y")


def test_explicit_date_range_is_forwarded(tmp_path):
    received = {}

    def downloader(**kwargs):
        received.update(kwargs)
        return yahoo_frame(kwargs["tickers"])

    manager = DataManager(tmp_path, downloader=downloader, cache_max_age_days=None)
    manager.prices(["AAPL"], period=None, start="2024-01-01", end="2024-02-01")

    assert received["start"] == "2024-01-01"
    assert received["end"] == "2024-02-01"
    assert "period" not in received


def test_downloads_are_chunked(tmp_path):
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs["tickers"])
        return yahoo_frame(kwargs["tickers"])

    manager = DataManager(
        tmp_path,
        downloader=downloader,
        cache_max_age_days=None,
        download_batch_size=2,
    )
    manager.prices(["A", "B", "C", "D", "E"], period="1y")

    assert calls == [["A", "B"], ["C", "D"], "E"]
