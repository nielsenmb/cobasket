"""Tests for guided workspace-state inspection."""

from __future__ import annotations

import json
import os
import time

from cobasket.gui.workspace_state import inspect_workspace


def _write(path, text="{}\n"):
    """Write a small test artifact and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def _portfolio(tmp_path, **overrides):
    """Write a minimal portfolio configuration for state tests."""
    payload = {
        "holdings": {},
        "cash": 0.0,
        "watchlist_path": "discovered_watchlist.json",
        "validation_path": None,
        "basket_calibration_path": None,
    }
    payload.update(overrides)
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_empty_workspace_starts_with_discovery(tmp_path):
    """A new directory should recommend discovery and nothing downstream."""
    state = inspect_workspace(tmp_path)
    assert state.name == "empty"
    assert state.next_stage == "discover"
    assert state.update_stages == ("discover",)


def test_discovered_workspace_recommends_validation(tmp_path):
    """A watchlist plus portfolio should continue with validation."""
    _portfolio(tmp_path)
    _write(tmp_path / "discovered_watchlist.json")
    state = inspect_workspace(tmp_path)
    assert state.name == "needs_validation"
    assert state.next_stage == "validate"
    assert state.update_stages == ("validate", "calibrate", "report")


def test_existing_workspace_follows_configured_legacy_paths(tmp_path):
    """Older workspaces should be recognized from paths stored in portfolio.json."""
    _portfolio(
        tmp_path,
        watchlist_path="portfolio_watchlist.json",
        validation_path="old_validation.json",
        basket_calibration_path="old_calibration.json",
    )
    _write(tmp_path / "portfolio_watchlist.json")
    _write(tmp_path / "old_validation.json")
    _write(tmp_path / "old_calibration.json")
    _write(tmp_path / "report.json")
    state = inspect_workspace(tmp_path)
    assert state.name == "ready"
    assert state.next_label == "Refresh recommendations"


def test_newer_watchlist_invalidates_validation_and_downstream_state(tmp_path):
    """Changing basket discovery should send the workflow back to validation."""
    _portfolio(tmp_path)
    watchlist = _write(tmp_path / "discovered_watchlist.json")
    _write(tmp_path / "basket_validation.json")
    _write(tmp_path / "basket_calibration.json")
    _write(tmp_path / "report.json")
    future = time.time() + 10.0
    os.utime(watchlist, (future, future))
    state = inspect_workspace(tmp_path)
    assert state.name == "needs_validation"
    assert state.next_stage == "validate"


def test_validation_without_calibration_recommends_calibration(tmp_path):
    """A partially completed workspace should resume at calibration."""
    _portfolio(tmp_path)
    _write(tmp_path / "discovered_watchlist.json")
    _write(tmp_path / "basket_validation.json")
    state = inspect_workspace(tmp_path)
    assert state.name == "needs_calibration"
    assert state.next_stage == "calibrate"
    assert state.update_stages == ("calibrate", "report")


def test_portfolio_edit_marks_existing_report_for_refresh(tmp_path):
    """Changing holdings or base currency should invalidate only the live report."""
    portfolio = _portfolio(tmp_path)
    _write(tmp_path / "discovered_watchlist.json")
    _write(tmp_path / "basket_validation.json")
    _write(tmp_path / "basket_calibration.json")
    _write(tmp_path / "report.json")
    future = time.time() + 10.0
    os.utime(portfolio, (future, future))
    state = inspect_workspace(tmp_path)
    assert state.name == "needs_report"
    assert state.next_stage == "report"
    assert state.update_stages == ("report",)
