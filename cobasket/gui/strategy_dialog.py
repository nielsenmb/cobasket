"""PyQt dialog for end-to-end historical basket strategy simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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
)

from cobasket.data import DataManager
from cobasket.evidence import BasketWatchlist, LongOnlyPolicy
from cobasket.robust_strategy import RobustnessGateConfig, run_robustness_aware_strategy
from cobasket.strategy_simulation import BasketStrategyConfig
from cobasket.workflow import PortfolioConfig


class StrategyWorker(QObject):
    """Download prices and run one basket strategy outside the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        basket: tuple[str, ...],
        period: str,
        config: BasketStrategyConfig,
        gate: RobustnessGateConfig,
    ) -> None:
        super().__init__()
        self.basket = basket
        self.period = period
        self.config = config
        self.gate = gate

    def run(self) -> None:
        """Emit a completed result or a readable error message."""
        try:
            prices = DataManager().prices(self.basket, period=self.period)
            result = run_robustness_aware_strategy(
                prices,
                strategy=self.config,
                gate=self.gate,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class StrategySimulationDialog(QDialog):
    """Configure and inspect a leakage-free historical basket simulation."""

    def __init__(self, config_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Basket strategy simulation")
        self.resize(1150, 850)
        config_path = Path(config_path)
        portfolio = PortfolioConfig.load(config_path)
        watchlist_path = Path(portfolio.watchlist_path)
        if not watchlist_path.is_absolute():
            watchlist_path = config_path.parent / watchlist_path
        self.watchlist = BasketWatchlist.load(watchlist_path)
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
        allowed = {"2y", "3y", "5y", "10y", "max"}
        self.period_combo.setCurrentText(portfolio.period if portfolio.period in allowed else "5y")
        self.train_spin = QSpinBox(); self.train_spin.setRange(60, 2000); self.train_spin.setValue(252)
        self.z_spin = QSpinBox(); self.z_spin.setRange(10, 500); self.z_spin.setValue(portfolio.z_window)
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 252); self.horizon_spin.setValue(20)
        self.step_spin = QSpinBox(); self.step_spin.setRange(1, 60); self.step_spin.setValue(5)
        self.min_samples_spin = QSpinBox(); self.min_samples_spin.setRange(1, 1000); self.min_samples_spin.setValue(30)
        self.cash_spin = QDoubleSpinBox(); self.cash_spin.setRange(1, 1e9); self.cash_spin.setValue(max(portfolio.cash, 10000.0)); self.cash_spin.setPrefix("£")
        self.buy_spin = QDoubleSpinBox(); self.buy_spin.setRange(0.51, 0.99); self.buy_spin.setSingleStep(0.01); self.buy_spin.setValue(0.60)
        self.strong_spin = QDoubleSpinBox(); self.strong_spin.setRange(0.52, 1.0); self.strong_spin.setSingleStep(0.01); self.strong_spin.setValue(0.70)
        self.reduce_spin = QDoubleSpinBox(); self.reduce_spin.setRange(0.01, 0.49); self.reduce_spin.setSingleStep(0.01); self.reduce_spin.setValue(0.40)
        self.sell_spin = QDoubleSpinBox(); self.sell_spin.setRange(0.0, 0.48); self.sell_spin.setSingleStep(0.01); self.sell_spin.setValue(0.30)
        self.cost_spin = QDoubleSpinBox(); self.cost_spin.setRange(0, 500); self.cost_spin.setValue(10); self.cost_spin.setSuffix(" bps")
        self.robust_check = QCheckBox("Enable robustness gate"); self.robust_check.setChecked(True)
        self.robust_window = QSpinBox(); self.robust_window.setRange(20, 1000); self.robust_window.setValue(126)
        self.robust_step = QSpinBox(); self.robust_step.setRange(1, 252); self.robust_step.setValue(21)
        self.max_half_life = QDoubleSpinBox(); self.max_half_life.setRange(1, 1000); self.max_half_life.setValue(120)
        self.max_weight_drift = QDoubleSpinBox(); self.max_weight_drift.setRange(0, 2); self.max_weight_drift.setSingleStep(0.05); self.max_weight_drift.setValue(0.50)
        self.stable_fraction = QDoubleSpinBox(); self.stable_fraction.setRange(0, 1); self.stable_fraction.setSingleStep(0.05); self.stable_fraction.setValue(0.60)
        for label, widget in [
            ("Basket", self.basket_combo), ("Price history", self.period_combo),
            ("Training window", self.train_spin), ("Z-score window", self.z_spin),
            ("Outcome horizon", self.horizon_spin), ("Evaluation step", self.step_spin),
            ("Minimum calibration samples", self.min_samples_spin), ("Starting cash", self.cash_spin),
            ("Buy probability", self.buy_spin), ("Strong-buy probability", self.strong_spin),
            ("Reduce probability", self.reduce_spin), ("Sell probability", self.sell_spin),
            ("Transaction cost", self.cost_spin), ("Robustness filtering", self.robust_check),
            ("Robustness window", self.robust_window), ("Robustness step", self.robust_step),
            ("Maximum half-life", self.max_half_life), ("Maximum weight drift", self.max_weight_drift),
            ("Minimum stable fraction", self.stable_fraction),
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
        tabs.addTab(self.trades, "Filtered trades")
        self.stability_table = QTableWidget(0, 6)
        self.stability_table.setHorizontalHeaderLabels(["Date", "Stable", "Trace ratio", "Half-life", "Weight drift", "Reason"])
        tabs.addTab(self.stability_table, "Stability")
        outer.addWidget(tabs, 1)

    def _configuration(self) -> tuple[BasketStrategyConfig, RobustnessGateConfig]:
        """Build validated backend configurations from the controls."""
        policy = LongOnlyPolicy(
            buy_probability=self.buy_spin.value(),
            strong_buy_probability=self.strong_spin.value(),
            reduce_probability=self.reduce_spin.value(),
            sell_probability=self.sell_spin.value(),
            transaction_cost_bps=self.cost_spin.value(),
        )
        strategy = BasketStrategyConfig(
            train_window=self.train_spin.value(), z_window=self.z_spin.value(),
            horizon=self.horizon_spin.value(), step=self.step_spin.value(),
            min_calibration_samples=self.min_samples_spin.value(),
            initial_cash=self.cash_spin.value(), policy=policy,
        )
        gate = RobustnessGateConfig(
            enabled=self.robust_check.isChecked(), window=self.robust_window.value(),
            step=self.robust_step.value(), max_half_life=self.max_half_life.value(),
            max_weight_drift=self.max_weight_drift.value(),
            minimum_stable_fraction=self.stable_fraction.value(),
        )
        return strategy, gate

    def run_simulation(self) -> None:
        """Start one historical simulation in a worker thread."""
        if self._thread is not None:
            return
        if self.basket_combo.count() == 0:
            QMessageBox.warning(self, "No baskets", "The selected watchlist contains no baskets to simulate.")
            return
        try:
            config, gate = self._configuration()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid strategy", str(exc))
            return
        basket = tuple(self.basket_combo.currentData())
        self.run_button.setEnabled(False)
        self.status.setText("Downloading prices and running robustness-aware walk-forward fits…")
        thread = QThread(self)
        worker = StrategyWorker(basket, self.period_combo.currentText(), config, gate)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._show_result)
        worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup)
        self._thread, self._worker = thread, worker
        thread.start()

    def _show_result(self, result) -> None:
        """Render filtered/unfiltered statistics, curves, trades, and stability."""
        filtered = result.filtered
        unfiltered = result.unfiltered
        c = result.comparison
        self.metrics.setText(
            f"Filtered end £{filtered.summary['ending_value']:,.2f}  |  "
            f"Filtered profit £{c['filtered_profit']:,.2f}  |  "
            f"Unfiltered profit £{c['unfiltered_profit']:,.2f}  |  "
            f"Difference £{c['profit_difference']:,.2f}  |  "
            f"Stable evaluations {c['stable_evaluation_fraction']:.1%}  |  "
            f"Filtered drawdown {c['filtered_maximum_drawdown']:.1%}"
        )
        f_equity = filtered.backtest.equity
        u_equity = unfiltered.backtest.equity
        drawdown = f_equity / f_equity.cummax() - 1.0
        invested = filtered.backtest.weights.sum(axis=1)
        for axis in self.axes:
            axis.clear()
        self.axes[0].plot(f_equity.index, f_equity, label="Robustness filtered")
        self.axes[0].plot(u_equity.index, u_equity, label="Unfiltered")
        self.axes[0].plot(filtered.benchmark_equity.index, filtered.benchmark_equity, label="Equal weight")
        self.axes[0].legend(); self.axes[0].set_ylabel("Value")
        self.axes[1].plot(drawdown.index, drawdown); self.axes[1].set_ylabel("Filtered drawdown")
        self.axes[2].plot(invested.index, invested); self.axes[2].set_ylabel("Invested fraction")
        self.canvas.figure.tight_layout(); self.canvas.draw_idle()
        table = filtered.backtest.trades
        self.trades.setRowCount(len(table))
        for row, item in enumerate(table.itertuples(index=False)):
            values = [str(item.date)[:10], item.ticker, item.side, f"{item.quantity:.4g}", f"£{item.price:,.2f}", f"{item.probability:.1%}"]
            for col, value in enumerate(values):
                self.trades.setItem(row, col, QTableWidgetItem(value))
        stability = result.stability.reset_index()
        self.stability_table.setRowCount(len(stability))
        for row, item in enumerate(stability.itertuples(index=False)):
            values = [
                str(item.date)[:10], "Yes" if item.stable else "No",
                "—" if not hasattr(item, "trace_ratio") or pd.isna(item.trace_ratio) else f"{item.trace_ratio:.2f}",
                "—" if not hasattr(item, "half_life") or pd.isna(item.half_life) else f"{item.half_life:.1f}",
                "—" if not hasattr(item, "weight_drift") or pd.isna(item.weight_drift) else f"{item.weight_drift:.2f}",
                str(item.reason),
            ]
            for col, value in enumerate(values):
                self.stability_table.setItem(row, col, QTableWidgetItem(value))
        self.status.setText("Simulation complete.")

    def _show_error(self, message: str) -> None:
        """Display a failed simulation without closing the dialog."""
        QMessageBox.critical(self, "Simulation failed", message)
        self.status.setText("Simulation failed.")

    def _cleanup(self) -> None:
        """Release worker references and re-enable controls."""
        self.run_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
