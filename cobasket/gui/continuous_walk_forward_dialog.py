"""PyQt interface for continuous walk-forward deployment simulations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cobasket.continuous_walk_forward import (
    ContinuousDeploymentConfig,
    run_continuous_walk_forward,
)
from cobasket.price_metrics import PriceMetricConfig, build_price_metrics, merge_metric_tables
from cobasket.repeated_walk_forward import WalkForwardConfig
from cobasket.strategy_experiments import StrategyExperimentConfig
from cobasket.strategy_rules import StrategyRules

from .strategy_experiment_dialog import StrategyEditorDialog, _read_table


class ContinuousWalkForwardWorker(QObject):
    """Load inputs and run a continuous walk-forward simulation."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        price_path: str,
        probability_path: str,
        strategies: tuple[StrategyRules, ...],
        walk_forward: WalkForwardConfig,
        experiment: StrategyExperimentConfig,
        deployment: ContinuousDeploymentConfig,
        metric_config: PriceMetricConfig,
    ) -> None:
        super().__init__()
        self.price_path = price_path
        self.probability_path = probability_path
        self.strategies = strategies
        self.walk_forward = walk_forward
        self.experiment = experiment
        self.deployment = deployment
        self.metric_config = metric_config

    def run(self) -> None:
        """Execute the simulation and emit a result or readable error."""
        try:
            prices = _read_table(self.price_path)
            probability = _read_table(self.probability_path)
            metrics = merge_metric_tables(
                {"probability": probability.reindex(index=prices.index, columns=prices.columns)},
                build_price_metrics(prices, config=self.metric_config),
            )
            result = run_continuous_walk_forward(
                prices,
                metrics,
                self.strategies,
                walk_forward=self.walk_forward,
                experiment=self.experiment,
                deployment=self.deployment,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class ContinuousWalkForwardDialog(QDialog):
    """Configure, run, inspect, and export one continuous deployment test."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Continuous walk-forward deployment")
        self.resize(1400, 900)
        self.strategies: list[StrategyRules] = []
        self.result = None
        self._thread: QThread | None = None
        self._worker: ContinuousWalkForwardWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct controls, plots, tables, and warnings."""
        outer = QVBoxLayout(self)
        splitter = QSplitter()
        outer.addWidget(splitter, 1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        paths = QFormLayout()
        from PyQt6.QtWidgets import QLineEdit

        self.price_edit = QLineEdit()
        self.probability_edit = QLineEdit()
        paths.addRow("Historical prices", self._path_row(self.price_edit))
        paths.addRow("Probability table", self._path_row(self.probability_edit))
        controls_layout.addLayout(paths)

        controls_layout.addWidget(QLabel("Candidate strategies"))
        self.strategy_list = QListWidget()
        controls_layout.addWidget(self.strategy_list, 1)
        strategy_buttons = QHBoxLayout()
        for label, callback in [
            ("New", self.new_strategy),
            ("Edit", self.edit_strategy),
            ("Load", self.load_strategy),
            ("Remove", self.remove_strategy),
        ]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            strategy_buttons.addWidget(button)
        controls_layout.addLayout(strategy_buttons)

        fold_form = QFormLayout()
        self.train_spin = self._integer_spin(20, 10000, 504)
        self.validation_spin = self._integer_spin(10, 5000, 126)
        self.test_spin = self._integer_spin(10, 5000, 126)
        self.step_spin = self._integer_spin(10, 5000, 126)
        self.expanding_check = QCheckBox("Use all earlier training observations")
        fold_form.addRow("Training observations", self.train_spin)
        fold_form.addRow("Validation observations", self.validation_spin)
        fold_form.addRow("Test observations", self.test_spin)
        fold_form.addRow("Fold step", self.step_spin)
        fold_form.addRow("Expanding training", self.expanding_check)
        controls_layout.addLayout(fold_form)

        experiment_form = QFormLayout()
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(
            ["sharpe_ratio", "total_return", "annualized_return", "maximum_drawdown", "annualized_volatility"]
        )
        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setRange(1.0, 1e9)
        self.cash_spin.setPrefix("$")
        self.cash_spin.setValue(10_000.0)
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems(["retain", "liquidate"])
        experiment_form.addRow("Selection metric", self.selection_combo)
        experiment_form.addRow("Starting cash", self.cash_spin)
        experiment_form.addRow("Strategy-change policy", self.boundary_combo)
        controls_layout.addLayout(experiment_form)

        metric_form = QFormLayout()
        self.momentum_window = self._integer_spin(2, 1000, 60)
        self.trend_window = self._integer_spin(2, 1000, 100)
        self.volatility_window = self._integer_spin(2, 1000, 20)
        metric_form.addRow("Momentum window", self.momentum_window)
        metric_form.addRow("Trend window", self.trend_window)
        metric_form.addRow("Volatility window", self.volatility_window)
        controls_layout.addLayout(metric_form)

        self.run_button = QPushButton("Run continuous deployment")
        self.run_button.clicked.connect(self.run_simulation)
        controls_layout.addWidget(self.run_button)
        self.status = QLabel("Choose historical inputs and add candidate strategies.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        splitter.addWidget(controls)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        self.summary_label = QLabel("No simulation result")
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(self.summary_label)
        self.tabs = QTabWidget()
        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.metrics_table = self._table()
        self.selection_table = self._table()
        self.trade_table = self._table()
        self.decision_table = self._table()
        self.warning_list = QListWidget()
        self.tabs.addTab(self.canvas, "Equity and exposure")
        self.tabs.addTab(self.metrics_table, "Metrics")
        self.tabs.addTab(self.selection_table, "Strategy timeline")
        self.tabs.addTab(self.trade_table, "Trades")
        self.tabs.addTab(self.decision_table, "Decisions")
        self.tabs.addTab(self.warning_list, "Warnings")
        results_layout.addWidget(self.tabs, 1)
        export_button = QPushButton("Export continuous simulation…")
        export_button.clicked.connect(self.export_result)
        results_layout.addWidget(export_button)
        splitter.addWidget(results)
        splitter.setSizes([430, 970])

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        """Return a configured integer spin box."""
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _path_row(self, line_edit) -> QWidget:
        """Return a line edit and table-selection button."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(line_edit))
        layout.addWidget(button)
        return widget

    def _browse(self, target) -> None:
        """Choose a CSV or Parquet table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose historical table", "", "Tables (*.csv *.parquet *.pq)"
        )
        if path:
            target.setText(path)

    @staticmethod
    def _table() -> QTableWidget:
        """Return a read-only sortable table."""
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def new_strategy(self) -> None:
        """Create and append a candidate strategy."""
        dialog = StrategyEditorDialog(parent=self)
        if dialog.exec() and dialog.strategy is not None:
            self._append_strategy(dialog.strategy)

    def edit_strategy(self) -> None:
        """Edit the selected candidate strategy."""
        row = self.strategy_list.currentRow()
        if row < 0:
            return
        dialog = StrategyEditorDialog(self.strategies[row], self)
        if dialog.exec() and dialog.strategy is not None:
            if any(item.name == dialog.strategy.name and index != row for index, item in enumerate(self.strategies)):
                QMessageBox.warning(self, "Duplicate name", "Strategy names must be unique.")
                return
            self.strategies[row] = dialog.strategy
            self.strategy_list.item(row).setText(dialog.strategy.name)

    def load_strategy(self) -> None:
        """Load a strategy JSON file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load strategy", "", "JSON (*.json)")
        if path:
            try:
                self._append_strategy(StrategyRules.load(path))
            except Exception as exc:
                QMessageBox.critical(self, "Cannot load strategy", str(exc))

    def _append_strategy(self, strategy: StrategyRules) -> None:
        """Append a uniquely named candidate strategy."""
        if any(item.name == strategy.name for item in self.strategies):
            raise ValueError(f"duplicate strategy name: {strategy.name}")
        self.strategies.append(strategy)
        self.strategy_list.addItem(strategy.name)

    def remove_strategy(self) -> None:
        """Remove the selected candidate strategy."""
        row = self.strategy_list.currentRow()
        if row >= 0:
            self.strategy_list.takeItem(row)
            self.strategies.pop(row)

    def run_simulation(self) -> None:
        """Validate controls and start continuous deployment in a worker thread."""
        if self._thread is not None:
            return
        if not self.price_edit.text().strip() or not self.probability_edit.text().strip():
            QMessageBox.warning(self, "Missing input", "Choose both price and probability tables.")
            return
        if not self.strategies:
            QMessageBox.warning(self, "Missing strategies", "Add at least one candidate strategy.")
            return
        if self.step_spin.value() < self.test_spin.value():
            QMessageBox.warning(
                self,
                "Overlapping tests",
                "Continuous deployment requires the fold step to be at least the test length.",
            )
            return

        cash = self.cash_spin.value()
        worker = ContinuousWalkForwardWorker(
            self.price_edit.text().strip(),
            self.probability_edit.text().strip(),
            tuple(self.strategies),
            WalkForwardConfig(
                train_observations=self.train_spin.value(),
                validation_observations=self.validation_spin.value(),
                test_observations=self.test_spin.value(),
                step_observations=self.step_spin.value(),
                expanding_train=self.expanding_check.isChecked(),
            ),
            StrategyExperimentConfig(
                selection_metric=self.selection_combo.currentText(),
                initial_cash=cash,
            ),
            ContinuousDeploymentConfig(
                initial_cash=cash,
                boundary_policy=self.boundary_combo.currentText(),
            ),
            PriceMetricConfig(
                momentum_window=self.momentum_window.value(),
                trend_window=self.trend_window.value(),
                volatility_window=self.volatility_window.value(),
            ),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._show_result)
        worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup)
        self._worker, self._thread = worker, thread
        self.run_button.setEnabled(False)
        self.status.setText("Selecting strategies and simulating one continuous account…")
        thread.start()

    def _show_result(self, result) -> None:
        """Render continuous portfolio results."""
        self.result = result
        strategy_metrics = result.metrics.loc["strategy"]
        self.summary_label.setText(
            f"Ending value ${strategy_metrics['ending_value']:,.2f}; "
            f"profit ${strategy_metrics['profit']:,.2f}; "
            f"total return {strategy_metrics['total_return']:.1%}; "
            f"maximum drawdown {strategy_metrics['maximum_drawdown']:.1%}."
        )
        self._fill_table(self.metrics_table, result.metrics)
        self._fill_table(self.selection_table, result.selections)
        self._fill_table(self.trade_table, result.trades)
        self._fill_table(self.decision_table, result.decisions)
        self.warning_list.clear()
        self.warning_list.addItems(result.warnings or ("No simulation warnings.",))
        self._plot_result(result)
        self.status.setText("Continuous deployment simulation complete.")

    def _plot_result(self, result) -> None:
        """Plot equity, drawdown, and invested fraction."""
        self.figure.clear()
        equity_axis = self.figure.add_subplot(311)
        result.equity.plot(ax=equity_axis)
        equity_axis.set_title("Continuous portfolio value")
        equity_axis.set_ylabel("Value ($)")
        equity_axis.grid(True, alpha=0.3)

        drawdown_axis = self.figure.add_subplot(312, sharex=equity_axis)
        drawdown = result.equity["strategy"] / result.equity["strategy"].cummax() - 1.0
        drawdown.plot(ax=drawdown_axis)
        drawdown_axis.set_title("Strategy drawdown")
        drawdown_axis.set_ylabel("Drawdown")
        drawdown_axis.grid(True, alpha=0.3)

        exposure_axis = self.figure.add_subplot(313, sharex=equity_axis)
        result.weights.sum(axis=1).plot(ax=exposure_axis)
        exposure_axis.set_title("Invested fraction")
        exposure_axis.set_ylabel("Portfolio fraction")
        exposure_axis.set_ylim(-0.02, 1.02)
        exposure_axis.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _fill_table(widget: QTableWidget, table: pd.DataFrame) -> None:
        """Populate a table widget from a DataFrame."""
        display = table.reset_index()
        widget.setSortingEnabled(False)
        widget.setRowCount(len(display))
        widget.setColumnCount(len(display.columns))
        widget.setHorizontalHeaderLabels([str(item) for item in display.columns])
        for row, item in display.iterrows():
            for column, value in enumerate(item):
                if isinstance(value, float):
                    text = f"{value:.5g}"
                elif isinstance(value, pd.Timestamp):
                    text = value.date().isoformat()
                else:
                    text = str(value)
                widget.setItem(row, column, QTableWidgetItem(text))
        widget.resizeColumnsToContents()
        widget.setSortingEnabled(True)

    def export_result(self) -> None:
        """Export all continuous deployment outputs."""
        if self.result is None:
            QMessageBox.information(self, "No result", "Run a simulation first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export continuous simulation")
        if directory:
            self.result.save(Path(directory))
            self.status.setText(f"Simulation exported to {directory}")

    def _show_error(self, message: str) -> None:
        """Display a simulation failure."""
        QMessageBox.critical(self, "Simulation failed", message)
        self.status.setText("Continuous deployment simulation failed.")

    def _cleanup(self) -> None:
        """Release worker references and re-enable execution."""
        self.run_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
