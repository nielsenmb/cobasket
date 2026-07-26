"""PyQt historical policy-backtest and calibration dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cobasket.validation import ValidationResult, build_validation_result


def read_indexed_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table with a datetime index.

    Parameters
    ----------
    path
        Input ``.csv`` or ``.parquet`` file.

    Returns
    -------
    pandas.DataFrame
        Numeric table sorted by its parsed datetime index.
    """
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


class ValidationDialog(QDialog):
    """Display historical policy performance and probability calibration."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cobasket historical validation")
        self.resize(1200, 820)
        self.result: ValidationResult | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct file controls, metric labels, plots, and trade table."""
        outer = QVBoxLayout(self)
        inputs = QGroupBox("Historical inputs")
        form = QFormLayout(inputs)
        self.prices_edit = QLineEdit()
        self.probabilities_edit = QLineEdit()
        self.outcomes_edit = QLineEdit()
        form.addRow("Prices", self._path_row(self.prices_edit))
        form.addRow("Probabilities", self._path_row(self.probabilities_edit))
        form.addRow("Walk-forward outcomes (optional)", self._path_row(self.outcomes_edit))
        run_button = QPushButton("Run validation")
        run_button.clicked.connect(self.run_validation)
        form.addRow(run_button)
        outer.addWidget(inputs)

        metrics = QGroupBox("Summary metrics")
        metric_layout = QHBoxLayout(metrics)
        self.metric_labels: dict[str, QLabel] = {}
        for key, title in (
            ("total_return", "Total return"),
            ("annualized_return", "Annualized return"),
            ("sharpe_ratio", "Sharpe ratio"),
            ("maximum_drawdown", "Maximum drawdown"),
            ("trade_count", "Trades"),
            ("brier_score", "Brier score"),
            ("expected_calibration_error", "Calibration error"),
        ):
            label = QLabel(f"{title}: —")
            self.metric_labels[key] = label
            metric_layout.addWidget(label)
        outer.addWidget(metrics)

        tabs = QTabWidget()
        self.figure = Figure(figsize=(11, 7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        tabs.addTab(self.canvas, "Performance and calibration")
        self.trade_table = QTableWidget()
        tabs.addTab(self.trade_table, "Trade history")
        outer.addWidget(tabs, 1)
        self.status_label = QLabel("Choose historical price and probability files.")
        outer.addWidget(self.status_label)

    def _path_row(self, edit: QLineEdit) -> QWidget:
        """Create a line edit with a file browser button."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self._browse(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget

    def _browse(self, edit: QLineEdit) -> None:
        """Select a CSV or Parquet validation input file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose validation input",
            str(Path.cwd()),
            "Data files (*.csv *.parquet);;All files (*)",
        )
        if path:
            edit.setText(path)

    def run_validation(self) -> None:
        """Load selected inputs, run validation, and refresh the dashboard."""
        try:
            prices = read_indexed_table(self.prices_edit.text().strip())
            probabilities = read_indexed_table(self.probabilities_edit.text().strip())
            outcomes_path = self.outcomes_edit.text().strip()
            outcomes = pd.read_csv(outcomes_path) if outcomes_path else None
            result = build_validation_result(prices, probabilities, outcomes=outcomes)
        except Exception as exc:
            QMessageBox.critical(self, "Validation failed", f"{type(exc).__name__}: {exc}")
            return
        self.set_result(result)
        self.status_label.setText("Historical validation complete.")

    def set_result(self, result: ValidationResult) -> None:
        """Populate plots, summary metrics, and trades from a result."""
        self.result = result
        metrics = result.backtest.metrics
        percent_keys = {"total_return", "annualized_return", "maximum_drawdown"}
        for key, label in self.metric_labels.items():
            if key == "brier_score" or key == "expected_calibration_error":
                value = getattr(result.calibration, key, None) if result.calibration else None
            else:
                value = metrics.get(key)
            if value is None:
                text = "—"
            elif key in percent_keys:
                text = f"{float(value):.1%}"
            elif key == "trade_count":
                text = f"{int(value)}"
            else:
                text = f"{float(value):.3f}"
            title = label.text().split(":", 1)[0]
            label.setText(f"{title}: {text}")
        self._draw_result(result)
        self._populate_trades(result.backtest.trades)

    def _draw_result(self, result: ValidationResult) -> None:
        """Render equity, drawdown, exposure, and reliability panels."""
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        equity_ax, drawdown_ax, exposure_ax, reliability_ax = axes.flat
        equity_ax.plot(result.backtest.equity.index, result.backtest.equity, label="Policy")
        equity_ax.plot(result.benchmark_equity.index, result.benchmark_equity, label="Equal weight")
        equity_ax.set_title("Portfolio equity")
        equity_ax.legend()
        drawdown_ax.plot(result.drawdown.index, result.drawdown)
        drawdown_ax.set_title("Drawdown from prior peak")
        drawdown_ax.axhline(0.0, linewidth=0.8)
        exposure_ax.plot(result.invested_fraction.index, result.invested_fraction)
        exposure_ax.set_ylim(0.0, 1.05)
        exposure_ax.set_title("Invested fraction")
        reliability_ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        if result.calibration is not None:
            table = result.calibration.reliability
            reliability_ax.scatter(
                table["mean_probability"],
                table["observed_frequency"],
                s=20 + 4 * table["sample_count"],
                label="Observed",
            )
        reliability_ax.set_xlim(0.0, 1.0)
        reliability_ax.set_ylim(0.0, 1.0)
        reliability_ax.set_title("Reliability diagram")
        reliability_ax.set_xlabel("Predicted probability")
        reliability_ax.set_ylabel("Observed frequency")
        reliability_ax.legend()
        self.canvas.draw_idle()

    def _populate_trades(self, trades: pd.DataFrame) -> None:
        """Fill the trade ledger table."""
        self.trade_table.clear()
        self.trade_table.setColumnCount(len(trades.columns))
        self.trade_table.setHorizontalHeaderLabels([str(column) for column in trades.columns])
        self.trade_table.setRowCount(len(trades))
        for row, values in enumerate(trades.itertuples(index=False, name=None)):
            for column, value in enumerate(values):
                self.trade_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.trade_table.resizeColumnsToContents()
