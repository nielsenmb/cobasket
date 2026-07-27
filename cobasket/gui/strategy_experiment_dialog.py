"""PyQt strategy-rule editor and controlled experiment dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
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
from cobasket.strategy_experiments import (
    ExperimentSplit,
    StrategyExperimentConfig,
    run_strategy_experiment,
)
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


OPERATORS = (">", ">=", "<", "<=", "==", "!=")
ACTIONS = ("sell", "reduce", "buy", "strong_buy", "hold")
MATCH_MODES = ("all", "any")


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a dated CSV or Parquet table.

    Parameters
    ----------
    path
        Input table path. The first CSV column is interpreted as the date index.

    Returns
    -------
    pandas.DataFrame
        Chronological table with a ``DatetimeIndex``.
    """
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(source)
    else:
        table = pd.read_csv(source, index_col=0)
    table.index = pd.to_datetime(table.index)
    return table.sort_index()


def _condition_text(conditions: Iterable[MetricCondition]) -> str:
    """Serialize conditions into compact editable text."""
    return "; ".join(
        f"{item.metric} {item.operator} {item.threshold}" for item in conditions
    )


def _parse_threshold(text: str) -> float | bool:
    """Parse a numeric or boolean rule threshold."""
    cleaned = text.strip()
    if cleaned.lower() == "true":
        return True
    if cleaned.lower() == "false":
        return False
    return float(cleaned)


def parse_conditions(text: str) -> tuple[MetricCondition, ...]:
    """Parse semicolon-separated conditions from the strategy editor.

    Each condition uses ``metric operator threshold`` syntax, for example
    ``probability >= 0.6; stable == True``.
    """
    conditions: list[MetricCondition] = []
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 3 or parts[1] not in OPERATORS:
            raise ValueError(
                f"invalid condition {raw!r}; use 'metric operator threshold'"
            )
        conditions.append(MetricCondition(parts[0], parts[1], _parse_threshold(parts[2])))
    if not conditions:
        raise ValueError("each rule requires at least one condition")
    return tuple(conditions)


class StrategyEditorDialog(QDialog):
    """Edit one ordered declarative strategy."""

    def __init__(self, strategy: StrategyRules | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Strategy editor")
        self.resize(920, 520)
        self.strategy: StrategyRules | None = strategy
        self._build_ui()
        if strategy is not None:
            self.set_strategy(strategy)

    def _build_ui(self) -> None:
        """Construct strategy metadata, ordered rules, and actions."""
        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit("New strategy")
        self.cost_spin = QDoubleSpinBox()
        self.cost_spin.setRange(0.0, 500.0)
        self.cost_spin.setSuffix(" bps")
        self.cost_spin.setValue(10.0)
        self.minimum_trade_spin = QDoubleSpinBox()
        self.minimum_trade_spin.setRange(0.0, 1_000_000.0)
        self.minimum_trade_spin.setPrefix("£")
        self.minimum_trade_spin.setValue(1.0)
        form.addRow("Strategy name", self.name_edit)
        form.addRow("Transaction cost", self.cost_spin)
        form.addRow("Minimum trade value", self.minimum_trade_spin)
        outer.addLayout(form)

        self.rules = QTableWidget(0, 5)
        self.rules.setHorizontalHeaderLabels(
            ["Action", "Match", "Conditions", "Target weight", "Priority"]
        )
        self.rules.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rules.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.rules, 1)

        row_buttons = QHBoxLayout()
        for label, callback in [
            ("Add rule", self.add_rule),
            ("Remove", self.remove_rule),
            ("Move up", lambda: self.move_rule(-1)),
            ("Move down", lambda: self.move_rule(1)),
        ]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            row_buttons.addWidget(button)
        row_buttons.addStretch(1)
        load_button = QPushButton("Load JSON…")
        load_button.clicked.connect(self.load_strategy)
        save_button = QPushButton("Save JSON…")
        save_button.clicked.connect(self.save_strategy)
        row_buttons.addWidget(load_button)
        row_buttons.addWidget(save_button)
        outer.addLayout(row_buttons)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        accept = QPushButton("Use strategy")
        accept.clicked.connect(self.accept_strategy)
        buttons.addWidget(cancel)
        buttons.addWidget(accept)
        outer.addLayout(buttons)

        if self.rules.rowCount() == 0:
            self.add_rule()

    def add_rule(self) -> None:
        """Append a default editable rule."""
        row = self.rules.rowCount()
        self.rules.insertRow(row)
        self.rules.setItem(row, 0, QTableWidgetItem("buy"))
        self.rules.setItem(row, 1, QTableWidgetItem("all"))
        self.rules.setItem(row, 2, QTableWidgetItem("probability >= 0.60"))
        self.rules.setItem(row, 3, QTableWidgetItem("0.10"))
        self.rules.setItem(row, 4, QTableWidgetItem(str(row + 1)))

    def remove_rule(self) -> None:
        """Remove the currently selected rule."""
        row = self.rules.currentRow()
        if row >= 0:
            self.rules.removeRow(row)
            self._refresh_priorities()

    def move_rule(self, offset: int) -> None:
        """Move the selected rule while preserving all cell contents."""
        row = self.rules.currentRow()
        destination = row + offset
        if row < 0 or destination < 0 or destination >= self.rules.rowCount():
            return
        values = [self.rules.item(row, column).text() for column in range(4)]
        self.rules.removeRow(row)
        self.rules.insertRow(destination)
        for column, value in enumerate(values):
            self.rules.setItem(destination, column, QTableWidgetItem(value))
        self.rules.selectRow(destination)
        self._refresh_priorities()

    def _refresh_priorities(self) -> None:
        """Synchronize the visible priority numbers with row order."""
        for row in range(self.rules.rowCount()):
            self.rules.setItem(row, 4, QTableWidgetItem(str(row + 1)))

    def strategy_from_widgets(self) -> StrategyRules:
        """Build and validate a backend strategy from the table."""
        rules: list[StrategyRule] = []
        for row in range(self.rules.rowCount()):
            action = self.rules.item(row, 0).text().strip()
            match = self.rules.item(row, 1).text().strip().lower()
            if action not in ACTIONS:
                raise ValueError(f"unsupported action in row {row + 1}: {action}")
            if match not in MATCH_MODES:
                raise ValueError(f"match must be all or any in row {row + 1}")
            target_text = self.rules.item(row, 3).text().strip()
            target = None if target_text.lower() in {"", "hold", "none"} else float(target_text)
            rules.append(
                StrategyRule(
                    action=action,
                    conditions=parse_conditions(self.rules.item(row, 2).text()),
                    target_weight=target,
                    match=match,
                )
            )
        return StrategyRules(
            name=self.name_edit.text().strip(),
            rules=tuple(rules),
            transaction_cost_bps=self.cost_spin.value(),
            minimum_trade_value=self.minimum_trade_spin.value(),
        )

    def set_strategy(self, strategy: StrategyRules) -> None:
        """Populate the editor from a backend strategy."""
        self.name_edit.setText(strategy.name)
        self.cost_spin.setValue(strategy.transaction_cost_bps)
        self.minimum_trade_spin.setValue(strategy.minimum_trade_value)
        self.rules.setRowCount(0)
        for priority, rule in enumerate(strategy.rules, start=1):
            row = self.rules.rowCount()
            self.rules.insertRow(row)
            values = [
                rule.action,
                rule.match,
                _condition_text(rule.conditions),
                "hold" if rule.target_weight is None else str(rule.target_weight),
                str(priority),
            ]
            for column, value in enumerate(values):
                self.rules.setItem(row, column, QTableWidgetItem(value))

    def load_strategy(self) -> None:
        """Load a strategy JSON file into the editor."""
        path, _ = QFileDialog.getOpenFileName(self, "Load strategy", "", "JSON (*.json)")
        if path:
            try:
                self.set_strategy(StrategyRules.load(path))
            except Exception as exc:
                QMessageBox.critical(self, "Cannot load strategy", str(exc))

    def save_strategy(self) -> None:
        """Save the current validated strategy as JSON."""
        try:
            strategy = self.strategy_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid strategy", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save strategy", "strategy.json", "JSON (*.json)")
        if path:
            strategy.save(path)

    def accept_strategy(self) -> None:
        """Validate and retain the edited strategy before closing."""
        try:
            self.strategy = self.strategy_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid strategy", str(exc))
            return
        self.accept()


class ExperimentWorker(QObject):
    """Load historical inputs and run one controlled strategy experiment."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        price_path: str,
        probability_path: str,
        strategies: tuple[StrategyRules, ...],
        train_fraction: float,
        validation_fraction: float,
        selection_metric: str,
        initial_cash: float,
        metric_config: PriceMetricConfig,
    ) -> None:
        super().__init__()
        self.price_path = price_path
        self.probability_path = probability_path
        self.strategies = strategies
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.selection_metric = selection_metric
        self.initial_cash = initial_cash
        self.metric_config = metric_config

    def run(self) -> None:
        """Run the experiment and emit either a result or readable error."""
        try:
            prices = _read_table(self.price_path)
            probability = _read_table(self.probability_path)
            metrics = merge_metric_tables(
                {"probability": probability.reindex(index=prices.index, columns=prices.columns)},
                build_price_metrics(prices, config=self.metric_config),
            )
            split = ExperimentSplit.from_fractions(
                prices.index,
                train_fraction=self.train_fraction,
                validation_fraction=self.validation_fraction,
            )
            result = run_strategy_experiment(
                prices,
                metrics,
                self.strategies,
                split,
                config=StrategyExperimentConfig(
                    selection_metric=self.selection_metric,
                    initial_cash=self.initial_cash,
                ),
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class StrategyExperimentDialog(QDialog):
    """Build candidate strategies and run a controlled historical experiment."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Strategy experiments")
        self.resize(1250, 820)
        self.strategies: list[StrategyRules] = []
        self._thread: QThread | None = None
        self._worker: ExperimentWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct input, strategy, split, and result panels."""
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

        experiment = QFormLayout()
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.20, 0.80)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.50)
        self.validation_spin = QDoubleSpinBox()
        self.validation_spin.setRange(0.10, 0.60)
        self.validation_spin.setSingleStep(0.05)
        self.validation_spin.setValue(0.25)
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(
            ["sharpe_ratio", "total_return", "annualized_return", "maximum_drawdown", "annualized_volatility"]
        )
        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setRange(1.0, 1e9)
        self.cash_spin.setPrefix("£")
        self.cash_spin.setValue(10_000.0)
        experiment.addRow("Training fraction", self.train_spin)
        experiment.addRow("Validation fraction", self.validation_spin)
        experiment.addRow("Selection metric", self.selection_combo)
        experiment.addRow("Starting cash", self.cash_spin)
        controls_layout.addLayout(experiment)

        metric_form = QFormLayout()
        self.momentum_window = QSpinBox(); self.momentum_window.setRange(2, 1000); self.momentum_window.setValue(60)
        self.trend_window = QSpinBox(); self.trend_window.setRange(2, 1000); self.trend_window.setValue(100)
        self.volatility_window = QSpinBox(); self.volatility_window.setRange(2, 1000); self.volatility_window.setValue(20)
        metric_form.addRow("Momentum window", self.momentum_window)
        metric_form.addRow("Trend window", self.trend_window)
        metric_form.addRow("Volatility window", self.volatility_window)
        controls_layout.addLayout(metric_form)

        self.run_button = QPushButton("Run controlled experiment")
        self.run_button.clicked.connect(self.run_experiment)
        controls_layout.addWidget(self.run_button)
        self.status = QLabel("Add candidate strategies and choose historical input tables.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        splitter.addWidget(controls)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        self.selected_label = QLabel("No experiment result")
        self.selected_label.setWordWrap(True)
        results_layout.addWidget(self.selected_label)
        self.tabs = QTabWidget()
        self.train_table = self._result_table()
        self.validation_table = self._result_table()
        self.test_table = self._result_table()
        self.warning_list = QListWidget()
        self.tabs.addTab(self.train_table, "Training")
        self.tabs.addTab(self.validation_table, "Validation")
        self.tabs.addTab(self.test_table, "Test")
        self.tabs.addTab(self.warning_list, "Warnings")
        results_layout.addWidget(self.tabs, 1)
        export_button = QPushButton("Export experiment…")
        export_button.clicked.connect(self.export_result)
        results_layout.addWidget(export_button)
        splitter.addWidget(results)
        splitter.setSizes([430, 820])
        self.result = None

    def _path_row(self, line_edit: QLineEdit) -> QWidget:
        """Return a line edit and browse button as one form widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(line_edit))
        layout.addWidget(button)
        return widget

    def _browse(self, target: QLineEdit) -> None:
        """Choose a CSV or Parquet table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose historical table", "", "Tables (*.csv *.parquet *.pq)"
        )
        if path:
            target.setText(path)

    def _result_table(self) -> QTableWidget:
        """Create a sortable result table."""
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def new_strategy(self) -> None:
        """Create and add a new candidate strategy."""
        dialog = StrategyEditorDialog(parent=self)
        if dialog.exec() and dialog.strategy is not None:
            self._append_strategy(dialog.strategy)

    def edit_strategy(self) -> None:
        """Edit the selected candidate in place."""
        row = self.strategy_list.currentRow()
        if row < 0:
            return
        dialog = StrategyEditorDialog(self.strategies[row], self)
        if dialog.exec() and dialog.strategy is not None:
            if any(
                item.name == dialog.strategy.name and index != row
                for index, item in enumerate(self.strategies)
            ):
                QMessageBox.warning(self, "Duplicate name", "Strategy names must be unique.")
                return
            self.strategies[row] = dialog.strategy
            self.strategy_list.item(row).setText(dialog.strategy.name)

    def load_strategy(self) -> None:
        """Load a strategy JSON file into the candidate list."""
        path, _ = QFileDialog.getOpenFileName(self, "Load strategy", "", "JSON (*.json)")
        if path:
            try:
                self._append_strategy(StrategyRules.load(path))
            except Exception as exc:
                QMessageBox.critical(self, "Cannot load strategy", str(exc))

    def _append_strategy(self, strategy: StrategyRules) -> None:
        """Append a uniquely named candidate."""
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

    def run_experiment(self) -> None:
        """Start the controlled experiment in a worker thread."""
        if self._thread is not None:
            return
        if not self.price_edit.text().strip() or not self.probability_edit.text().strip():
            QMessageBox.warning(self, "Missing input", "Choose both price and probability tables.")
            return
        if not self.strategies:
            QMessageBox.warning(self, "Missing strategies", "Add at least one candidate strategy.")
            return
        if self.train_spin.value() + self.validation_spin.value() >= 1.0:
            QMessageBox.warning(self, "Invalid split", "Training and validation must leave a test interval.")
            return
        metric_config = PriceMetricConfig(
            momentum_window=self.momentum_window.value(),
            trend_window=self.trend_window.value(),
            volatility_window=self.volatility_window.value(),
        )
        worker = ExperimentWorker(
            self.price_edit.text().strip(),
            self.probability_edit.text().strip(),
            tuple(self.strategies),
            self.train_spin.value(),
            self.validation_spin.value(),
            self.selection_combo.currentText(),
            self.cash_spin.value(),
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
        self.status.setText("Running training, validation selection, and final test evaluation…")
        thread.start()

    def _show_result(self, result) -> None:
        """Render experiment tables and warnings."""
        self.result = result
        self.selected_label.setText(
            f"Selected on validation: {result.selected_strategy.name}. "
            "The test tab contains this strategy and the two benchmarks only."
        )
        self._fill_table(self.train_table, result.train_table)
        self._fill_table(self.validation_table, result.validation_table)
        self._fill_table(self.test_table, result.test_table)
        self.warning_list.clear()
        self.warning_list.addItems(result.warnings or ("No experiment warnings.",))
        self.status.setText("Experiment complete.")

    def _fill_table(self, widget: QTableWidget, table: pd.DataFrame) -> None:
        """Populate one result table from a DataFrame."""
        display = table.reset_index()
        widget.setSortingEnabled(False)
        widget.setRowCount(len(display))
        widget.setColumnCount(len(display.columns))
        widget.setHorizontalHeaderLabels([str(item) for item in display.columns])
        for row, item in display.iterrows():
            for column, value in enumerate(item):
                text = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
                widget.setItem(row, column, QTableWidgetItem(text))
        widget.resizeColumnsToContents()
        widget.setSortingEnabled(True)

    def export_result(self) -> None:
        """Export the completed experiment directory."""
        if self.result is None:
            QMessageBox.information(self, "No result", "Run an experiment first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export experiment")
        if directory:
            self.result.save(directory)
            self.status.setText(f"Experiment exported to {directory}")

    def _show_error(self, message: str) -> None:
        """Display an experiment failure."""
        QMessageBox.critical(self, "Experiment failed", message)
        self.status.setText("Experiment failed.")

    def _cleanup(self) -> None:
        """Release worker references and re-enable execution."""
        self.run_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
