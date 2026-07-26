"""PyQt dialog for end-to-end historical basket strategy simulation."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cobasket.data import DataManager
from cobasket.evidence import BasketWatchlist, LongOnlyPolicy
from cobasket.strategy_simulation import BasketStrategyConfig, run_basket_strategy_simulation
from cobasket.workflow import PortfolioConfig


class StrategyWorker(QObject):
    """Download prices and run one basket strategy outside the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, basket: tuple[str, ...], period: str, config: BasketStrategyConfig) -> None:
        super().__init__()
        self.basket = basket
        self.period = period
        self.config = config

    def run(self) -> None:
        """Emit a completed result or a readable error message."""
        try:
            prices = DataManager().prices(self.basket, period=self.period)
            result = run_basket_strategy_simulation(prices, config=self.config)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class StrategySimulationDialog(QDialog):
    """Configure and inspect a leakage-free historical basket simulation."""

    def __init__(self, config_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Basket strategy simulation")
        self.resize(1100, 780)
        portfolio = PortfolioConfig.load(config_path)
        self.watchlist = BasketWatchlist.load(portfolio.watchlist_path)
        self._thread: QThread | None = None
        self._worker: StrategyWorker | None = None
        self._build_ui(portfolio)

    def _build_ui(self, portfolio: PortfolioConfig) -> None:
        """Construct parameter controls, plots, metrics, and trade table."""
        outer = QVBoxLayout(self)
        controls = QFormLayout()
        self.basket_combo = QComboBox()
        for basket in self.watchlist.baskets:
            self.basket_combo.addItem(", ".join(basket), basket)
        self.period_combo = QComboBox()
        self.period_combo.addItems(["2y", "3y", "5y", "10y", "max"])
        self.period_combo.setCurrentText(portfolio.period if portfolio.period in {"2y", "3y", "5y", "10y", "max"} else "5y")
        self.train_spin = QSpinBox(); self.train_spin.setRange(60, 2000); self.train_spin.setValue(252)
        self.z_spin = QSpinBox(); self.z_spin.setRange(10, 500); self.z_spin.setValue(portfolio.z_window)
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 252); self.horizon_spin.setValue(20)
        self.step_spin = QSpinBox(); self.step_spin.setRange(1, 60); self.step_spin.setValue(5)
        self.min_samples_spin = QSpinBox(); self.min_samples_spin.setRange(1, 1000); self.min_samples_spin.setValue(30)
        self.cash_spin = QDoubleSpinBox(); self.cash_spin.setRange(1, 1e9); self.cash_spin.setValue(max(portfolio.cash, 10000.0)); self.cash_spin.setPrefix("$")
        self.buy_spin = QDoubleSpinBox(); self.buy_spin.setRange(0.51, 0.99); self.buy_spin.setSingleStep(0.01); self.buy_spin.setValue(0.60)
        self.strong_spin = QDoubleSpinBox(); self.strong_spin.setRange(0.52, 1.0); self.strong_spin.setSingleStep(0.01); self.strong_spin.setValue(0.70)
        self.reduce_spin = QDoubleSpinBox(); self.reduce_spin.setRange(0.01, 0.49); self.reduce_spin.setSingleStep(0.01); self.reduce_spin.setValue(0.40)
        self.sell_spin = QDoubleSpinBox(); self.sell_spin.setRange(0.0, 0.48); self.sell_spin.setSingleStep(0.01); self.sell_spin.setValue(0.30)
        self.cost_spin = QDoubleSpinBox(); self.cost_spin.setRange(0, 500); self.cost_spin.setValue(10); self.cost_spin.setSuffix(" bps")
        for label, widget in [
            ("Basket", self.basket_combo), ("Price history", self.period_combo),
            ("Training window", self.train_spin), ("Z-score window", self.z_spin),
            ("Outcome horizon", self.horizon_spin), ("Evaluation step", self.step_spin),
            ("Minimum calibration samples", self.min_samples_spin), ("Starting cash", self.cash_spin),
            ("Buy probability", self.buy_spin), ("Strong-buy probability", self.strong_spin),
            ("Reduce probability", self.reduce_spin), ("Sell probability", self.sell_spin),
            ("Transaction cost", self.cost_spin),
        ]:
            controls.addRow(label, widget)
        outer.addLayout(controls)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run simulation")
        self.run_button.clicked.connect(self.run_simulation)
        buttons.addWidget(self.run_button)
        self.status = QLabel("Choose a basket and strategy, then run the simulation.")
        buttons.addWidget(self.status, 1)
        outer.addLayout(buttons)

        self.metrics = QLabel("No result")
        self.metrics.setWordWrap(True)
        outer.addWidget(self.metrics)
        tabs = QTabWidget()
        figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvasQTAgg(figure)
        self.axes = figure.subplots(3, 1, sharex=True)
        tabs.addTab(self.canvas, "Performance")
        self.trades = QTableWidget(0, 6)
        self.trades.setHorizontalHeaderLabels(["Date", "Ticker", "Side", "Quantity", "Price", "Probability"])
        tabs.addTab(self.trades, "Trades")
        outer.addWidget(tabs, 1)

    def _configuration(self) -> BasketStrategyConfig:
        """Build validated backend configuration from the controls."""
        policy = LongOnlyPolicy(
            buy_probability=self.buy_spin.value(), strong_buy_probability=self.strong_spin.value(),
            reduce_probability=self.reduce_spin.value(), sell_probability=self.sell_spin.value(),
            transaction_cost_bps=self.cost_spin.value(),
        )
        return BasketStrategyConfig(
            train_window=self.train_spin.value(), z_window=self.z_spin.value(),
            horizon=self.horizon_spin.value(), step=self.step_spin.value(),
            min_calibration_samples=self.min_samples_spin.value(),
            initial_cash=self.cash_spin.value(), policy=policy,
        )

    def run_simulation(self) -> None:
        """Start one historical simulation in a worker thread."""
        if self._thread is not None:
            return
        try:
            config = self._configuration()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid strategy", str(exc)); return
        basket = tuple(self.basket_combo.currentData())
        self.run_button.setEnabled(False)
        self.status.setText("Downloading prices and running walk-forward fits…")
        thread = QThread(self); worker = StrategyWorker(basket, self.period_combo.currentText(), config)
        worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.finished.connect(self._show_result); worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup)
        self._thread, self._worker = thread, worker
        thread.start()

    def _show_result(self, result) -> None:
        """Render summary statistics, curves, and executed trades."""
        s = result.summary
        self.metrics.setText(
            f"Start ${s['starting_value']:,.2f}  |  End ${s['ending_value']:,.2f}  |  "
            f"Profit ${s['profit']:,.2f}  |  Benchmark ${s['benchmark_profit']:,.2f}  |  "
            f"Excess ${s['excess_profit']:,.2f}  |  Max drawdown {s['maximum_drawdown']:.1%}  |  "
            f"Trades {int(s['trade_count'])}"
        )
        equity = result.backtest.equity
        drawdown = equity / equity.cummax() - 1.0
        invested = result.backtest.weights.sum(axis=1)
        for axis in self.axes: axis.clear()
        self.axes[0].plot(equity.index, equity, label="Strategy"); self.axes[0].plot(result.benchmark_equity.index, result.benchmark_equity, label="Equal weight"); self.axes[0].legend(); self.axes[0].set_ylabel("Value")
        self.axes[1].plot(drawdown.index, drawdown); self.axes[1].set_ylabel("Drawdown")
        self.axes[2].plot(invested.index, invested); self.axes[2].set_ylabel("Invested fraction")
        self.canvas.figure.tight_layout(); self.canvas.draw_idle()
        table = result.backtest.trades
        self.trades.setRowCount(len(table))
        for row, item in enumerate(table.itertuples(index=False)):
            values = [str(item.date)[:10], item.ticker, item.side, f"{item.quantity:.4g}", f"${item.price:,.2f}", f"{item.probability:.1%}"]
            for col, value in enumerate(values): self.trades.setItem(row, col, QTableWidgetItem(value))
        self.status.setText("Simulation complete.")

    def _show_error(self, message: str) -> None:
        """Display a failed simulation without closing the dialog."""
        QMessageBox.critical(self, "Simulation failed", message); self.status.setText("Simulation failed.")

    def _cleanup(self) -> None:
        """Release worker references and re-enable controls."""
        self.run_button.setEnabled(True); self._worker = None
        if self._thread is not None: self._thread.deleteLater()
        self._thread = None
