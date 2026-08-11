"""Regression tests for the guided GUI workflow command construction."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from cobasket.gui.workflow_dialog import workflow_command


def test_discovery_command_relinks_existing_portfolio() -> None:
    """GUI discovery should update an existing portfolio to the new watchlist."""
    command = workflow_command("discover", universe="sp500", period="5y")
    assert command[:2] == ("-m", "cobasket.discovery_cli")
    assert "--update-portfolio" in command
    assert command[command.index("--universe") + 1] == "sp500"
    assert command[command.index("--period") + 1] == "5y"


def test_guided_workflow_uses_expected_output_files() -> None:
    """Validation, calibration, and report stages should share one workspace schema."""
    validation = workflow_command("validate", universe="sp500", period="5y")
    calibration = workflow_command("calibrate", universe="sp500", period="5y")
    report = workflow_command("report", universe="sp500", period="5y")
    assert "basket_validation.json" in validation
    assert "basket_calibration.json" in calibration
    assert "report.json" in report
    assert "portfolio.json" in validation
    assert "portfolio.json" in calibration
    assert "portfolio.json" in report


def test_unknown_workflow_stage_is_rejected() -> None:
    """Invalid stage names should fail before launching a process."""
    with pytest.raises(ValueError, match="unknown workflow stage"):
        workflow_command("unknown", universe="sp500", period="5y")
