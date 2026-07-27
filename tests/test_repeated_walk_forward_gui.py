"""Headless tests for the repeated walk-forward experiment dashboard."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.repeated_walk_forward_dialog import RepeatedWalkForwardDialog
from cobasket.repeated_walk_forward import RepeatedWalkForwardResult


def _synthetic_result() -> RepeatedWalkForwardResult:
    """Return compact aggregate tables suitable for GUI rendering tests."""
    folds = pd.DataFrame(
        {
            "fold": [1, 2],
            "train_start": pd.to_datetime(["2020-01-01", "2020-07-01"]),
            "train_end": pd.to_datetime(["2020-06-30", "2020-12-31"]),
            "validation_start": pd.to_datetime(["2020-07-01", "2021-01-01"]),
            "validation_end": pd.to_datetime(["2020-09-30", "2021-03-31"]),
            "test_start": pd.to_datetime(["2020-10-01", "2021-04-01"]),
            "test_end": pd.to_datetime(["2020-12-31", "2021-06-30"]),
            "selected_strategy": ["base", "momentum"],
            "test_total_return": [0.05, -0.01],
            "test_annualized_return": [0.20, -0.04],
            "test_sharpe_ratio": [1.1, -0.2],
            "test_maximum_drawdown": [-0.03, -0.08],
            "test_trade_count": [3.0, 4.0],
            "benchmark_total_return": [0.03, 0.02],
            "cash_total_return": [0.0, 0.0],
            "excess_return": [0.02, -0.03],
        }
    )
    frequency = pd.DataFrame(
        {"selected_folds": [1, 1], "selection_fraction": [0.5, 0.5]},
        index=pd.Index(["base", "momentum"], name="strategy"),
    )
    compounded = pd.DataFrame(
        {
            "strategy": [10_000.0, 10_500.0, 10_395.0],
            "equal_weight": [10_000.0, 10_300.0, 10_506.0],
            "cash": [10_000.0, 10_000.0, 10_000.0],
        },
        index=pd.RangeIndex(3, name="fold_boundary"),
    )
    return RepeatedWalkForwardResult(
        folds=(),
        fold_table=folds,
        selection_frequency=frequency,
        compounded_equity=compounded,
        warnings=("Different strategies were selected across folds.",),
    )


def test_repeated_dialog_renders_aggregate_result():
    """Fold tables, plots, warnings, and summary should all populate."""
    app = QApplication.instance() or QApplication([])
    dialog = RepeatedWalkForwardDialog()
    dialog._show_result(_synthetic_result())

    assert dialog.fold_table.rowCount() == 2
    assert dialog.frequency_table.rowCount() == 2
    assert dialog.warning_list.count() == 1
    assert "Folds: 2" in dialog.summary.text()
    assert len(dialog.canvas.figure.axes) == 2

    dialog.close()
    app.processEvents()


def test_repeated_dialog_builds_non_overlapping_default_config():
    """Default fold controls should preserve independent test intervals."""
    app = QApplication.instance() or QApplication([])
    dialog = RepeatedWalkForwardDialog()
    config = dialog._walk_forward_config()

    assert config.step_observations >= config.test_observations
    assert config.allow_overlapping_tests is False

    dialog.close()
    app.processEvents()
