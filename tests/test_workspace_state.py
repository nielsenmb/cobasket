"""Tests for guided workspace-state inspection."""

from __future__ import annotations

import json
import os
import time

from cobasket.gui.workspace_state import FreshnessPolicy, inspect_workspace, load_freshness_policy


_SECONDS_PER_DAY = 86400.0


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


def _set_age(path, days, now):
    """Set an artifact modification time to a controlled age.

    Parameters
    ----------
    path
        Artifact whose timestamp should be changed.
    days
        Desired age in days.
    now
        Reference Unix timestamp.
    """
    timestamp = now - float(days) * _SECONDS_PER_DAY
    os.utime(path, (timestamp, timestamp))


def _complete_workspace(tmp_path):
    """Create a complete workspace and return its artifact paths."""
    paths = {
        "portfolio": _portfolio(
            tmp_path,
            validation_path="basket_validation.json",
            basket_calibration_path="basket_calibration.json",
        ),
        "watchlist": _write(tmp_path / "discovered_watchlist.json"),
        "validation": _write(tmp_path / "basket_validation.json"),
        "calibration": _write(tmp_path / "basket_calibration.json"),
        "report": _write(tmp_path / "report.json"),
    }
    return paths


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


def test_old_report_recommends_live_refresh(tmp_path):
    """A report older than its policy interval should be refreshed."""
    now = time.time()
    paths = _complete_workspace(tmp_path)
    _set_age(paths["watchlist"], 30, now)
    _set_age(paths["validation"], 20, now)
    _set_age(paths["calibration"], 10, now)
    _set_age(paths["portfolio"], 5, now)
    _set_age(paths["report"], 4, now)
    policy = FreshnessPolicy(report_days=3, validation_days=90, calibration_days=90, discovery_days=180)
    state = inspect_workspace(tmp_path, now=now, policy=policy)
    assert state.name == "needs_report"
    assert state.next_label == "Refresh recommendations"
    assert "4 days old" in state.summary


def test_old_validation_cascades_through_calibration_and_report(tmp_path):
    """Age-expired validation should require all downstream stages again."""
    now = time.time()
    paths = _complete_workspace(tmp_path)
    _set_age(paths["watchlist"], 120, now)
    _set_age(paths["validation"], 100, now)
    _set_age(paths["calibration"], 80, now)
    _set_age(paths["portfolio"], 2, now)
    _set_age(paths["report"], 1, now)
    policy = FreshnessPolicy(report_days=3, validation_days=90, calibration_days=90, discovery_days=180)
    state = inspect_workspace(tmp_path, now=now, policy=policy)
    assert state.name == "needs_validation"
    assert state.next_label == "Refresh validation"
    assert state.update_stages == ("validate", "calibrate", "report")
    assert "100 days old" in state.summary


def test_custom_calibration_age_can_trigger_before_validation(tmp_path):
    """A custom short calibration interval should be respected independently."""
    now = time.time()
    paths = _complete_workspace(tmp_path)
    _set_age(paths["watchlist"], 100, now)
    _set_age(paths["validation"], 60, now)
    _set_age(paths["calibration"], 40, now)
    _set_age(paths["portfolio"], 2, now)
    _set_age(paths["report"], 1, now)
    policy = FreshnessPolicy(report_days=3, validation_days=180, calibration_days=30, discovery_days=180)
    state = inspect_workspace(tmp_path, now=now, policy=policy)
    assert state.name == "needs_calibration"
    assert state.next_label == "Refresh calibration"
    assert state.update_stages == ("calibrate", "report")


def test_old_discovery_is_advisory_not_automatic(tmp_path):
    """An old watchlist should recommend rediscovery without invalidating valid models."""
    now = time.time()
    paths = _complete_workspace(tmp_path)
    _set_age(paths["watchlist"], 200, now)
    _set_age(paths["validation"], 20, now)
    _set_age(paths["calibration"], 10, now)
    _set_age(paths["portfolio"], 2, now)
    _set_age(paths["report"], 1, now)
    policy = FreshnessPolicy(report_days=3, validation_days=90, calibration_days=90, discovery_days=180)
    state = inspect_workspace(tmp_path, now=now, policy=policy)
    assert state.name == "ready"
    assert "consider re-running" in state.summary
    assert "refresh recommended" in dict(state.stage_statuses)["Discovery"]


def test_workspace_policy_round_trip(tmp_path):
    """Stored workspace freshness settings should override the defaults."""
    policy = FreshnessPolicy(report_days=2, validation_days=60, calibration_days=45, discovery_days=120)
    policy.save(tmp_path / "workspace_freshness.json")
    loaded = load_freshness_policy(tmp_path)
    assert loaded == policy
