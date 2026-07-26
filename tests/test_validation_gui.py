"""Headless rendering test for the historical validation dashboard."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.validation_dialog import ValidationDialog
from cobasket.validation import build_validation_result


def test_validation_dialog_renders_metrics_plots_and_trades():
    """A synthetic validation result should populate the complete dialog."""
    app = QApplication.instance() or QApplication([])
    index = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 120.0, len(index)),
            "BBB": np.linspace(100.0, 108.0, len(index)),
        },
        index=index,
    )
    probabilities = pd.DataFrame(
        {"AAA": [0.75, 0.25], "BBB": [0.25, 0.75]},
        index=index[[10, 30]],
    )
    outcomes = pd.DataFrame(
        {"probability_outperform": [0.75, 0.25], "outperformed": [1, 0]}
    )
    result = build_validation_result(prices, probabilities, outcomes=outcomes)
    dialog = ValidationDialog()
    dialog.set_result(result)
    assert len(dialog.figure.axes) == 4
    assert dialog.trade_table.rowCount() == len(result.backtest.trades)
    assert "—" not in dialog.metric_labels["total_return"].text()
    dialog.close()
    app.processEvents()
