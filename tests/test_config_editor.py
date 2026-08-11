"""Tests for portfolio and watchlist editor validation helpers."""

import json

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox

from cobasket.evidence import BasketWatchlist
from cobasket.gui.config_editor import (
    ConfigEditorDialog,
    holdings_from_rows,
    parse_basket_text,
)
from cobasket.workflow import PortfolioConfig


def test_parse_basket_text_normalizes_and_deduplicates():
    """Basket text should produce stable upper-case unique symbols."""
    assert parse_basket_text(" aapl, msft, AAPL ") == ("AAPL", "MSFT")


def test_parse_basket_text_requires_two_unique_tickers():
    """A monitored basket must contain at least two distinct assets."""
    with pytest.raises(ValueError, match="at least two"):
        parse_basket_text("AAPL, AAPL")


def test_holdings_from_rows_preserves_zero_quantity_for_reentry():
    """Zero holdings should remain valid rather than being dropped."""
    assert holdings_from_rows([("aapl", 0.0), ("msft", 1.5)]) == {
        "AAPL": 0.0,
        "MSFT": 1.5,
    }


def test_holdings_from_rows_rejects_duplicate_tickers():
    """Duplicate editor rows should not silently overwrite one another."""
    with pytest.raises(ValueError, match="duplicate"):
        holdings_from_rows([("AAPL", 1.0), ("aapl", 2.0)])


def test_holdings_from_rows_rejects_negative_quantities():
    """Long-only holdings cannot contain negative quantities."""
    with pytest.raises(ValueError, match="non-negative"):
        holdings_from_rows([("AAPL", -1.0)])


def test_editor_loads_zero_quantity_holdings_and_baskets(tmp_path):
    """The dialog should retain sold stocks and their monitored basket."""
    app = QApplication.instance() or QApplication([])
    watchlist_path = tmp_path / "watchlist.json"
    BasketWatchlist(baskets=(("AAPL", "MSFT"),)).save(watchlist_path)
    config_path = tmp_path / "portfolio.json"
    PortfolioConfig(
        holdings={"AAPL": 0.0, "MSFT": 2.0},
        cash=500.0,
        watchlist_path=str(watchlist_path),
    ).save(config_path)

    dialog = ConfigEditorDialog(config_path)
    assert dialog.holdings_table.rowCount() == 2
    assert dialog.baskets_table.rowCount() == 1
    assert dialog.cash_spin.value() == 500.0
    dialog.close()
    app.processEvents()


def test_editor_preserves_discovery_and_broker_metadata(tmp_path, monkeypatch):
    """Saving the GUI editor should not discard Trading 212 discovery metadata."""
    app = QApplication.instance() or QApplication([])
    watchlist_path = tmp_path / "watchlist.json"
    BasketWatchlist(baskets=(("AAPL", "MSFT"),)).save(watchlist_path)
    payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
    payload["universe_metadata"] = {"name": "sp1500", "trading212_filtered": True}
    payload["broker_metadata"] = {
        "broker": "trading212",
        "instrument_tickers": {"AAPL": "AAPL_US_EQ", "MSFT": "MSFT_US_EQ"},
    }
    watchlist_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "portfolio.json"
    PortfolioConfig(holdings={}, watchlist_path=str(watchlist_path)).save(config_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    dialog = ConfigEditorDialog(config_path)
    dialog.save_files()

    saved = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert saved["universe_metadata"]["trading212_filtered"] is True
    assert saved["broker_metadata"]["instrument_tickers"]["AAPL"] == "AAPL_US_EQ"
    app.processEvents()
