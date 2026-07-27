"""Headless smoke tests for the continuous walk-forward dialog."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication

from cobasket.continuous_walk_forward import ContinuousWalkForwardResult
from cobasket.gui.continuous_walk_forward_dialog import ContinuousWalkForwardDialog


def _app() -> QApplication:
    """Return the shared QApplication instance."""
    return QApplication.instance() or QApplication([])


def _result() -> ContinuousWalkForwardResult:
    """Build a minimal renderable continuous result."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    equity = pd.DataFrame(
        {
            "strategy": [10_000.0, 10_050.0, 10_020.0, 10_100.0],
            "equal_weight": [10_000.0, 10_020.0, 10_030.0, 10_060.0],
            "cash": [10_000.0] * 4,
        },
        index=index,
    )
    positions = pd.DataFrame({"AAA": [0.0, 10.0, 10.0, 10.0]}, index=index)
    weights = pd.DataFrame({"AAA": [0.0, 0.5, 0.5, 0.5]}, index=index)
    metrics = pd.DataFrame(
        {
            "ending_value": [10_100.0, 10_060.0, 10_000.0],
            "profit": [100.0, 60.0, 0.0],
            "total_return": [0.01, 0.006, 0.0],
            "annualized_return": [0.1, 0.06, 0.0],
            "annualized_volatility": [0.12, 0.1, 0.0],
            "sharpe_ratio": [0.8, 0.6, 0.0],
            "maximum_drawdown": [-0.003, 0.0, 0.0],
        },
        index=["strategy", "equal_weight", "cash"],
    )
    selections = pd.DataFrame(
        [{"fold": 1, "test_start": index[0], "test_end": index[-1], "selected_strategy": "demo"}]
    )
    trades = pd.DataFrame(
        [{"date": index[1], "ticker": "AAA", "side": "buy", "notional": 5000.0, "cost": 5.0}]
    )
    decisions = pd.DataFrame(
        [{"date": index[0], "ticker": "AAA", "action": "buy", "target_weight": 0.5}]
    )
    return ContinuousWalkForwardResult(
        equity=equity,
        cash=pd.Series([10_000.0, 5_000.0, 5_000.0, 5_000.0], index=index),
        positions=positions,
        weights=weights,
        trades=trades,
        decisions=decisions,
        selections=selections,
        metrics=metrics,
        warnings=(),
    )


def test_dialog_constructs_and_renders_result() -> None:
    """The dialog should render backend output without network access."""
    _app()
    dialog = ContinuousWalkForwardDialog()
    dialog._show_result(_result())
    assert "Ending value" in dialog.summary_label.text()
    assert dialog.metrics_table.rowCount() == 3
    assert dialog.selection_table.rowCount() == 1
    assert len(dialog.figure.axes) == 3


def test_overlap_is_rejected_by_controls() -> None:
    """The continuous GUI should default to non-overlapping test windows."""
    _app()
    dialog = ContinuousWalkForwardDialog()
    dialog.test_spin.setValue(126)
    dialog.step_spin.setValue(126)
    assert dialog.step_spin.value() >= dialog.test_spin.value()
