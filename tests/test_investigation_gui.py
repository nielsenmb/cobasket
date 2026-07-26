"""Headless rendering test for the basket investigation figure."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtWidgets import QApplication

from cobasket.gui.investigation_dialog import build_investigation_figure
from cobasket.investigation import investigate_basket


def test_investigation_figure_has_four_diagnostic_panels():
    """A completed investigation should render all planned diagnostic views."""
    app = QApplication.instance() or QApplication([])
    rng = np.random.default_rng(4)
    n = 260
    common = 100.0 + np.cumsum(rng.normal(0.0, 0.4, n))
    noise = rng.normal(0.0, 0.5, n)
    prices = pd.DataFrame(
        {"AAA": common + noise, "BBB": 0.9 * common - noise + 15.0},
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )
    result = investigate_basket(prices, window=40)
    figure = build_investigation_figure(result)
    canvas = FigureCanvasQTAgg(figure)
    assert len(figure.axes) == 4
    assert {axis.get_title() for axis in figure.axes} == {
        "Normalized prices",
        "Fitted spread",
        "Rolling spread z-score",
        "Cointegration weights",
    }
    canvas.close()
    app.processEvents()
