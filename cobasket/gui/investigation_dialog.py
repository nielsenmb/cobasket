"""Interactive PyQt and Matplotlib basket investigation view."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from cobasket.data import DataManager
from cobasket.evidence import BasketWatchlist
from cobasket.investigation import BasketInvestigation, investigate_basket
from cobasket.workflow import PortfolioConfig


class InvestigationWorker(QObject):
    """Download prices and fit one selected basket outside the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, tickers: tuple[str, ...], period: str, window: int) -> None:
        super().__init__()
        self.tickers = tickers
        self.period = period
        self.window = window

    def run(self) -> None:
        """Emit a completed investigation or a readable error message."""
        try:
            prices = DataManager().prices(self.tickers, period=self.period, min_coverage=1.0)
            result = investigate_basket(prices, window=self.window)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


def build_investigation_figure(result: BasketInvestigation) -> Figure:
    """Create the four-panel diagnostic figure for a basket.

    Parameters
    ----------
    result
        Completed basket investigation.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing normalized prices, spread, z-score, and weights.
    """
    figure = Figure(figsize=(10, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    price_ax = figure.add_subplot(grid[0, 0])
    spread_ax = figure.add_subplot(grid[0, 1])
    z_ax = figure.add_subplot(grid[1, 0])
    weight_ax = figure.add_subplot(grid[1, 1])

    result.normalized_prices.plot(ax=price_ax)
    price_ax.set_title("Normalized prices")
    price_ax.set_ylabel("Price / initial price")
    price_ax.legend(loc="best", fontsize="small")

    result.spread.plot(ax=spread_ax)
    spread_ax.axhline(result.spread.mean(), linestyle="--", linewidth=1)
    spread_ax.set_title("Fitted spread")
    spread_ax.set_ylabel("Weighted price combination")

    result.z_score.plot(ax=z_ax)
    for level in (-2.0, 0.0, 2.0):
        z_ax.axhline(level, linestyle="--", linewidth=1)
    z_ax.set_title("Rolling spread z-score")
    z_ax.set_ylabel("Standard deviations")

    result.weights.plot.bar(ax=weight_ax)
    weight_ax.axhline(0.0, linewidth=1)
    weight_ax.set_title("Cointegration weights")
    weight_ax.set_ylabel("L1-normalized weight")
    return figure


class BasketInvestigationDialog(QDialog):
    """Select and inspect baskets associated with one watched ticker."""

    def __init__(self, config_path: str | Path, ticker: str, parent=None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.ticker = ticker.upper()
        self._thread: QThread | None = None
        self._worker: InvestigationWorker | None = None
        self._canvas: FigureCanvasQTAgg | None = None

        config = PortfolioConfig.load(self.config_path)
        watchlist = BasketWatchlist.load(config.watchlist_path)
        self.config = config
        self.baskets = tuple(basket for basket in watchlist.baskets if self.ticker in basket)
        if not self.baskets:
            raise ValueError(f"{self.ticker} is not present in any monitored basket")

        self.setWindowTitle(f"Cobasket investigation — {self.ticker}")
        self.resize(1100, 850)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.basket_combo = QComboBox()
        for basket in self.baskets:
            self.basket_combo.addItem(", ".join(basket), basket)
        self.run_button = QPushButton("Load diagnostics")
        controls.addWidget(QLabel("Basket"))
        controls.addWidget(self.basket_combo, 1)
        controls.addWidget(self.run_button)
        layout.addLayout(controls)

        summary = QFormLayout()
        self.trace_label = QLabel("—")
        self.z_label = QLabel("—")
        summary.addRow("Johansen trace ratio", self.trace_label)
        summary.addRow("Latest spread z-score", self.z_label)
        layout.addLayout(summary)
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout, 1)
        self.status_label = QLabel("Choose a basket and load diagnostics.")
        layout.addWidget(self.status_label)
        self.run_button.clicked.connect(self.run_investigation)

    def run_investigation(self) -> None:
        """Start a background investigation for the selected basket."""
        if self._thread is not None:
            return
        basket = tuple(self.basket_combo.currentData())
        self.run_button.setEnabled(False)
        self.status_label.setText("Downloading prices and fitting basket…")
        thread = QThread(self)
        worker = InvestigationWorker(basket, self.config.period, self.config.z_window)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.set_result)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def set_result(self, result: BasketInvestigation) -> None:
        """Display a completed basket investigation."""
        if self._canvas is not None:
            self.plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(build_investigation_figure(result))
        self.plot_layout.addWidget(self._canvas)
        self.trace_label.setText(f"{result.trace_ratio:.2f}")
        self.z_label.setText(f"{result.latest_z_score:.2f}")
        self.status_label.setText("Diagnostics loaded.")

    def _failed(self, message: str) -> None:
        """Display a failed data download or basket fit."""
        QMessageBox.critical(self, "Investigation failed", message)
        self.status_label.setText("Investigation failed.")

    def _finished(self) -> None:
        """Release background worker references."""
        self.run_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
