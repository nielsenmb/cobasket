"""Release tests for universe retrieval and screening watchlist export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cobasket.cli import _save_ranked_watchlist
from cobasket.data import universe
from cobasket.data.exceptions import DownloadError
from cobasket.evidence import BasketWatchlist


class _FakeResponse:
    """Minimal context-manager response for mocked ``urlopen`` calls."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_sp500_first_run_uses_explicit_http_request(tmp_path: Path, monkeypatch) -> None:
    """A first-run constituent fetch should populate the local universe cache."""
    html = b"<table><tr><th>Symbol</th></tr><tr><td>BRK.B</td></tr><tr><td>AAPL</td></tr></table>"
    monkeypatch.setattr(universe, "urlopen", lambda request, timeout: _FakeResponse(html))

    tickers = universe.get_sp500_tickers(tmp_path)

    assert tickers == ["BRK-B", "AAPL"]
    assert (tmp_path / "sp500_tickers.csv").exists()


def test_sp500_refresh_falls_back_to_existing_cache(tmp_path: Path, monkeypatch) -> None:
    """A temporary HTTP failure should not discard a previously cached universe."""
    cache = tmp_path / "sp500_tickers.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(cache, index=False)

    def fail(*args, **kwargs):
        raise DownloadError("temporary failure")

    monkeypatch.setattr(universe, "_download_sp500_table", fail)
    with pytest.warns(RuntimeWarning, match="using the existing cached universe"):
        tickers = universe.get_sp500_tickers(tmp_path, force_refresh=True)

    assert tickers == ["AAPL", "MSFT"]


def test_ranked_screen_results_can_be_saved_as_watchlist(tmp_path: Path) -> None:
    """The screening CLI helper should save only the requested highest ranks."""
    results = [
        {"basket": ["AAPL", "MSFT"], "sharpe": 1.2},
        {"basket": ["XOM", "CVX"], "sharpe": 0.8},
    ]
    output = tmp_path / "watchlist.json"

    _save_ranked_watchlist(results, str(output), top_n=1, name="test screen")
    watchlist = BasketWatchlist.load(output)

    assert watchlist.name == "test screen"
    assert watchlist.baskets == (("AAPL", "MSFT"),)
