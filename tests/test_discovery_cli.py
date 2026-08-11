"""Tests for discovery CLI configuration bootstrapping."""

from __future__ import annotations

from pathlib import Path

from cobasket.discovery_cli import _ensure_portfolio_config
from cobasket.workflow import PortfolioConfig


def test_discovery_creates_missing_portfolio_config(tmp_path: Path) -> None:
    """Discovery should bootstrap a usable portfolio in a clean directory."""
    watchlist = tmp_path / "discovered_watchlist.json"
    watchlist.write_text("{}\n", encoding="utf-8")
    portfolio = tmp_path / "portfolio.json"

    action = _ensure_portfolio_config(portfolio, watchlist, period="5y", min_trace_ratio=1.0, update_existing=False)

    config = PortfolioConfig.load(portfolio)
    assert action == "created"
    assert config.holdings == {}
    assert config.watchlist_path == "discovered_watchlist.json"
    assert config.period == "5y"


def test_discovery_does_not_overwrite_existing_portfolio_without_flag(tmp_path: Path) -> None:
    """Existing user portfolio settings should be preserved by default."""
    portfolio = tmp_path / "portfolio.json"
    PortfolioConfig(holdings={"AAPL": 2.0}, cash=50.0, watchlist_path="old.json", period="3y").save(portfolio)
    watchlist = tmp_path / "new_watchlist.json"

    action = _ensure_portfolio_config(portfolio, watchlist, period="5y", min_trace_ratio=1.0, update_existing=False)

    config = PortfolioConfig.load(portfolio)
    assert action == "unchanged"
    assert config.holdings == {"AAPL": 2.0}
    assert config.watchlist_path == "old.json"
    assert config.period == "3y"


def test_discovery_updates_only_workflow_fields_when_requested(tmp_path: Path) -> None:
    """Explicit updating should preserve holdings while linking the new watchlist."""
    portfolio = tmp_path / "portfolio.json"
    PortfolioConfig(holdings={"AAPL": 2.0}, cash=50.0, watchlist_path="old.json", period="3y").save(portfolio)
    watchlist = tmp_path / "new_watchlist.json"

    action = _ensure_portfolio_config(portfolio, watchlist, period="5y", min_trace_ratio=1.1, update_existing=True)

    config = PortfolioConfig.load(portfolio)
    assert action == "updated"
    assert config.holdings == {"AAPL": 2.0}
    assert config.cash == 50.0
    assert config.watchlist_path == "new_watchlist.json"
    assert config.period == "5y"
    assert config.min_trace_ratio == 1.1
    assert config.validation_path is None
    assert config.basket_calibration_path is None
