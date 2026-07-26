"""Tests for portfolio and watchlist editor validation helpers."""

import pytest

from cobasket.gui.config_editor import holdings_from_rows, parse_basket_text


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
