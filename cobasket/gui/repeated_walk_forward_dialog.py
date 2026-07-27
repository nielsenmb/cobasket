"""PyQt dashboard for repeated walk-forward strategy experiments."""

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
    QHeaderView,
    QLabel,
    QLineEdit,
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

from cobasket.price_metrics import PriceMetricConfig, build_price_metrics, merge_metric_tables
from cobasket.repeated_walk_forward import (
    RepeatedWalkForwardResult,
    WalkForwardConfig,
    run_repeated_walk_forward,
)
from cobasket.strategy_experiments import StrategyExperimentConfig
from cobasket.strategy_rules import StrategyRules

from .strategy_experiment_dialog import StrategyEditorDialog, _read_table


class RepeatedExperimentWorker(QObject):
    """Load historical inputs and run repeated walk-forward experiments."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        price_path: str,
        probability_path: str,
        strategies: tuple[StrategyRules, ...],
        walk_forward: WalkForwardConfig,
        experiment: StrategyExperimentConfig,
        metric_config: PriceMetricConfig,
    ) -> None:
        super().__init__()
        self.price_path = price_path
        self.probability_path = probability_path
        self.strategies = strategies
        self.walk_forward = walk_forward
        self.experiment = experiment
        self.metric_config = metric_config

    def run(self) -> None:
        """Emit a completed result or a readable failure message."""
        try:
            prices = _read_table(self.price_path)
            probability = _read_table(self.probability_path)
            metrics = merge_metric_tables(
                {"probability": probability.reindex(index=prices.index, columns=prices.columns)},
                build_price_metrics(prices, config=self.metric_config),
            )
            result = run_repeated_walk_forward(
                prices,
                metrics,
                self.strategies,
                walk_forward=self.walk_forward,
                experiment=self.experiment,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class RepeatedWalkForwardDialog(QDialog):
    """Configure and inspect repeated chronological strategy experiments."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Repeated walk-forward experiments")
        self.resize(1320, 860)
        self.strategies: list[StrategyRules] = []
        self.result: RepeatedWalkForwardResult | None = None
        self._thread: QThread | None = None
        self._worker: RepeatedExperimentWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct controls, plots, tables, warnings, and export actions."""
        outer = QVBoxLayout(self)
        splitter = QSplitter()
        outer.addWidget(splitter, 1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        paths = QFormLayout()
        self.price_edit = QLineEdit()
        self.probability_edit = QLineEdit()
        paths.addRow("Historical prices", self._path_row(self.price_edit))
        paths.addRow("Probability table", self._path_row(self.probability_edit))
        controls_layout.addLayout(paths)

        controls_layout.addWidget(QLabel("Candidate strategies"))
        self.strategy_list = QListWidget()
        controls_layout.addWidget(self.strategy_list, 1)
        buttons = QHBoxLayout()
        for label, callback in [
            ("New", self.new_strategy),
            ("Edit", self.edit_strategy),
            ("Load", self.load_strategy),
            ("Remove", self.remove_strategy),
        ]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        controls_layout.addLayout(buttons)

        fold_form = QFormLayout()
        self.train_spin = QSpinBox(); self.train_spin.setRange(20, 5000); self.train_spin.setValue(504)
        self.validation_spin = QSpinBox(); self.validation_spin.setRange(10, 2000); self.validation_spin.setValue(126)
        self.test_spin = QSpinBox(); self.test_spin.setRange(10, 2000); self.test_spin.setValue(126)
        self.step_spin = QSpinBox(); self.step_spin.setRange(1, 2000); self.step_spin.setValue(126)
        self.expanding_check = QCheckBox("Use expanding training window")
        self.overlap_check = QCheckBox("Allow overlapping test intervals")
        fold_form.addRow("Training observations", self.train_spin)
        fold_form.addRow("Validation observations", self.validation_spin)
        fold_form.addRow("Test observations", self.test_spin)
        fold_form.addRow("Fold step", self.step_spin)
        fold_form.addRow("Training mode", self.expanding_check)
        fold_form.addRow("Overlap", self.overlap_check)
        controls_layout.addLayout(fold_form)

        experiment_form = QFormLayout()
        self.selection_combo = QComboBox()
        self.selection_combo.addItems([
            "sharpe_ratio", "total_return", "annualized_return",
            "maximum_drawdown", "annualized_volatility",
        ])
        self.cash_spin = QDoubleSpinBox(); self.cash_spin.setRange(1.0, 1e9); self.cash_spin.setValue(10_000.0); self.cash_spin.setPrefix("$")
        self.momentum_window = QSpinBox(); self.momentum_window.setRange(2, 1000); self.momentum_window.setValue(60)
        self.trend_window = QSpinBox(); self.trend_window.setRange(2, 1000); self.trend_window.setValue(100)
        self.volatility_window = QSpinBox(); self.volatility_window.setRange(2, 1000); self.volatility_window.setValue(20)
        experiment_form.addRow("Selection metric", self.selection_combo)
        experiment_form.addRow("Starting cash", self.cash_spin)
        experiment_form.addRow("Momentum window", self.momentum_window)
        experiment_form.addRow("Trend window", self.trend_window)
        experiment_form.addRow("Volatility window", self.volatility_window)
        controls_layout.addLayout(experiment_form)

        self.run_button = QPushButton("Run repeated experiment")
        self.run_button.clicked.connect(self.run_experiment)
        controls_layout.addWidget(self.run_button)
        self.status = QLabel("Choose inputs and add candidate strategies.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        splitter.addWidget(controls)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        self.summary = QLabel("No repeated experiment result")
        self.summary.setWordWrap(True)
        results_layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvasQTAgg(figure)
        self.axes = figure.subplots(2, 1)
        self.tabs.addTab(self.canvas, "Overview")
        self.fold_table = self._result_table()
        self.frequency_table = self._result_table()
        self.warning_list = QListWidget()
        self.tabs.addTab(self.fold_table, "Folds")
        self.tabs.addTab(self.frequency_table, "Selections")
        self.tabs.addTab(self.warning_list, "Warnings")
        results_layout.addWidget(self.tabs, 1)
        export_button = QPushButton("Export repeated experiment…")
        export_button.clicked.connect(self.export_result)
        results_layout.addWidget(export_button)
        splitter.addWidget(results)
        splitter.setSizes([440, 880])

    def _path_row(self, target: QLineEdit) -> QWidget:
        """Return a file entry field with a browse button."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(target, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(target))
        layout.addWidget(button)
        return widget

    def _browse(self, target: QLineEdit) -> None:
        """Choose a CSV or Parquet historical table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose historical table", "", "Tables (*.csv *.parquet *.pq)"
        )
        if path:
            target.setText(path)

    def _result_table(self) -> QTableWidget:
        """Create a sortable, read-only table."""
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        return table

    def new_strategy(self) -> None:
        """Create a new candidate strategy."""
        dialog = StrategyEditorDialog(parent=self)
        if dialog.exec() and dialog.strategy is not None:
            self._append_strategy(dialog.strategy)

    def edit_strategy(self) -> None:
        """Edit the selected strategy in place."""
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
        """Load one strategy from JSON."""
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
        """Remove the selected candidate."""
        row = self.strategy_list.currentRow()
        if row >= 0:
            self.strategy_list.takeItem(row)
            self.strategies.pop(row)

    def _walk_forward_config(self) -> WalkForwardConfig:
        """Build validated fold settings from the controls."""
        return WalkForwardConfig(
            train_observations=self.train_spin.value(),
            validation_observations=self.validation_spin.value(),
            test_observations=self.test_spin.value(),
            step_observations=self.step_spin.value(),
            expanding_train=self.expanding_check.isChecked(),
            allow_overlapping_tests=self.overlap_check.isChecked(),
        )

    def run_experiment(self) -> None:
        """Start repeated evaluation in a worker thread."""
        if self._thread is not None:
            return
        if not self.price_edit.text().strip() or not self.probability_edit.text().strip():
            QMessageBox.warning(self, "Missing input", "Choose both price and probability tables.")
            return
        if not self.strategies:
            QMessageBox.warning(self, "Missing strategies", "Add at least one candidate strategy.")
            return
        try:
            walk_forward = self._walk_forward_config()
            experiment = StrategyExperimentConfig(
                selection_metric=self.selection_combo.currentText(),
                initial_cash=self.cash_spin.value(),
            )
            metric_config = PriceMetricConfig(
                momentum_window=self.momentum_window.value(),
                trend_window=self.trend_window.value(),
                volatility_window=self.volatility_window.value(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Invalid experiment", str(exc))
            return
        worker = RepeatedExperimentWorker(
            self.price_edit.text().strip(),
            self.probability_edit.text().strip(),
            tuple(self.strategies),
            walk_forward,
            experiment,
            metric_config,
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
        self.status.setText("Running repeated validation selection and untouched test folds…")
        thread.start()

    def _show_result(self, result: RepeatedWalkForwardResult) -> None:
        """Render fold summaries, selection frequencies, curves, and warnings."""
        self.result = result
        folds = result.fold_table
        beat_fraction = float((folds["excess_return"] > 0.0).mean())
        selected_count = int((result.selection_frequency["selected_folds"] > 0).sum())
        ending = result.compounded_equity.iloc[-1]
        self.summary.setText(
            f"Folds: {len(folds)} | Strategies selected: {selected_count} | "
            f"Beat equal weight: {beat_fraction:.1%} | "
            f"Compounded strategy: ${ending['strategy']:,.2f} | "
            f"Equal weight: ${ending['equal_weight']:,.2f}"
        )
        self._fill_table(self.fold_table, folds)
        self._fill_table(self.frequency_table, result.selection_frequency)
        self.warning_list.clear()
        self.warning_list.addItems(result.warnings or ("No repeated-experiment warnings.",))

        for axis in self.axes:
            axis.clear()
        curves = result.compounded_equity
        for column in curves.columns:
            self.axes[0].plot(curves.index, curves[column], marker="o", label=column)
        self.axes[0].set_ylabel("Compounded value")
        self.axes[0].set_xlabel("Fold boundary")
        self.axes[0].legend()
        frequency = result.selection_frequency
        self.axes[1].bar(frequency.index.astype(str), frequency["selection_fraction"])
        self.axes[1].set_ylabel("Selection fraction")
        self.axes[1].tick_params(axis="x", rotation=30)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()
        self.status.setText("Repeated walk-forward experiment complete.")

    def _fill_table(self, widget: QTableWidget, table: pd.DataFrame) -> None:
        """Populate one read-only table from a DataFrame."""
        display = table.reset_index()
        widget.setSortingEnabled(False)
        widget.setRowCount(len(display))
        widget.setColumnCount(len(display.columns))
        widget.setHorizontalHeaderLabels([str(item) for item in display.columns])
        for row, item in display.iterrows():
            for column, value in enumerate(item):
                if isinstance(value, pd.Timestamp):
                    text = value.date().isoformat()
                elif isinstance(value, (int, float)):
                    text = f"{value:.4g}"
                else:
                    text = str(value)
                widget.setItem(row, column, QTableWidgetItem(text))
        widget.resizeColumnsToContents()
        widget.setSortingEnabled(True)

    def export_result(self) -> None:
        """Export aggregate tables and per-fold selected strategies."""
        if self.result is None:
            QMessageBox.information(self, "No result", "Run a repeated experiment first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export repeated experiment")
        if directory:
            self.result.save(directory)
            self.status.setText(f"Repeated experiment exported to {directory}")

    def _show_error(self, message: str) -> None:
        """Display a failed repeated experiment."""
        QMessageBox.critical(self, "Repeated experiment failed", message)
        self.status.setText("Repeated experiment failed.")

    def _cleanup(self) -> None:
        """Release worker references and re-enable execution."""
        self.run_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
