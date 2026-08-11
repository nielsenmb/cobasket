"""PyQt dashboard for Cobasket portfolio reports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cobasket.workflow import PortfolioAnalyzer, PortfolioConfig, PortfolioReport

from .models import portfolio_report_from_dict, probability_label


def classify_cobasket_json(path: str | Path) -> str:
    """Classify a Cobasket JSON file as a configuration or report.

    Parameters
    ----------
    path
        Existing JSON file to inspect.

    Returns
    -------
    str
        Either ``"portfolio"`` or ``"report"``.

    Raises
    ------
    ValueError
        If the file does not match either supported schema.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "holdings" in payload and "watchlist_path" in payload:
        return "portfolio"
    if isinstance(payload, dict) and "generated_at_utc" in payload and "tickers" in payload:
        return "report"
    raise ValueError("JSON is neither a Cobasket portfolio configuration nor a saved report")


class AnalysisWorker(QObject):
    """Run a fresh portfolio analysis outside the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config_path: str, force_refresh: bool = False) -> None:
        super().__init__()
        self.config_path = config_path
        self.force_refresh = force_refresh

    def run(self) -> None:
        """Load the configuration and emit a completed portfolio report."""
        try:
            config = PortfolioConfig.load(self.config_path)
            report = PortfolioAnalyzer().run(config, force_refresh=self.force_refresh)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(report)


class CobasketDashboard(QMainWindow):
    """Display and refresh current Cobasket portfolio recommendations."""

    columns = (
        "Ticker",
        "Held",
        "Price",
        "Market value",
        "Probability",
        "Recommendation",
        "Warnings",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cobasket dashboard")
        self.resize(1180, 800)
        self.report: PortfolioReport | None = None
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the dashboard widgets and signal connections."""
        central = QWidget(self)
        outer = QVBoxLayout(central)

        sources = QGroupBox("Sources")
        source_layout = QVBoxLayout(sources)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Portfolio configuration"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("portfolio.json")
        config_browse = QPushButton("Browse…")
        self.refresh_button = QPushButton("Run analysis")
        config_row.addWidget(self.path_edit, 1)
        config_row.addWidget(config_browse)
        config_row.addWidget(self.refresh_button)
        source_layout.addLayout(config_row)

        report_row = QHBoxLayout()
        report_row.addWidget(QLabel("Saved report"))
        self.report_path_edit = QLineEdit()
        self.report_path_edit.setPlaceholderText("report.json")
        report_browse = QPushButton("Browse…")
        load_button = QPushButton("Load report")
        report_row.addWidget(self.report_path_edit, 1)
        report_row.addWidget(report_browse)
        report_row.addWidget(load_button)
        source_layout.addLayout(report_row)
        outer.addWidget(sources)

        summary = QGroupBox("Portfolio summary")
        summary_layout = QHBoxLayout(summary)
        self.total_value_label = QLabel("Total value: —")
        self.cash_label = QLabel("Cash: —")
        self.invested_label = QLabel("Invested: —")
        self.price_date_label = QLabel("Latest prices: —")
        for widget in (self.total_value_label, self.cash_label, self.invested_label, self.price_date_label):
            summary_layout.addWidget(widget)
        summary_layout.addStretch(1)
        outer.addWidget(summary)

        splitter = QSplitter()
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        details_box = QGroupBox("Selected ticker")
        details_form = QFormLayout(details_box)
        self.detail_ticker = QLabel("—")
        self.detail_probability = QLabel("—")
        self.detail_recommendation = QLabel("—")
        self.detail_baskets = QLabel("—")
        self.detail_baskets.setWordWrap(True)
        details_form.addRow("Ticker", self.detail_ticker)
        details_form.addRow("Probability", self.detail_probability)
        details_form.addRow("Recommendation", self.detail_recommendation)
        details_form.addRow("Basket memberships", self.detail_baskets)
        detail_layout.addWidget(details_box)

        self.explanation = QTextEdit()
        self.explanation.setReadOnly(True)
        self.explanation.setPlaceholderText("Select a ticker to see the explanation.")
        detail_layout.addWidget(self.explanation, 1)

        warnings_box = QGroupBox("Report warnings")
        warnings_layout = QVBoxLayout(warnings_box)
        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        warnings_layout.addWidget(self.warnings_text)
        detail_layout.addWidget(warnings_box, 1)
        splitter.addWidget(detail_widget)
        splitter.setSizes([760, 420])
        outer.addWidget(splitter, 1)

        self.status_label = QLabel("Choose a portfolio configuration to analyse or a saved report to display.")
        outer.addWidget(self.status_label)
        self.setCentralWidget(central)

        config_browse.clicked.connect(self._browse_config)
        report_browse.clicked.connect(self._browse_report)
        load_button.clicked.connect(self.load_report)
        self.refresh_button.clicked.connect(self.run_analysis)
        self.table.itemSelectionChanged.connect(self._show_selected_ticker)

    def _browse_config(self) -> None:
        """Choose a portfolio configuration JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose portfolio configuration", str(Path.cwd()), "JSON files (*.json);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _browse_report(self) -> None:
        """Choose a saved portfolio report JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose saved report", str(Path.cwd()), "JSON files (*.json);;All files (*)"
        )
        if path:
            self.report_path_edit.setText(path)

    def load_report(self) -> None:
        """Load and display an existing serialized portfolio report."""
        path = Path(self.report_path_edit.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "Missing report", "Choose an existing saved report JSON file.")
            return
        try:
            kind = classify_cobasket_json(path)
            if kind != "report":
                raise ValueError("Selected file is a portfolio configuration, not a saved report")
            payload = json.loads(path.read_text(encoding="utf-8"))
            report = portfolio_report_from_dict(payload)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid report", f"Could not load report:\n{exc}")
            return
        self.set_report(report)
        self.status_label.setText(f"Loaded report from {path}")

    def run_analysis(self) -> None:
        """Run a fresh analysis from a portfolio configuration JSON file."""
        path = Path(self.path_edit.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "Missing configuration", "Choose an existing portfolio configuration JSON file.")
            return
        try:
            kind = classify_cobasket_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        if kind != "portfolio":
            QMessageBox.warning(
                self,
                "Portfolio configuration required",
                "Run analysis requires portfolio.json. A saved report can only be loaded in the Saved report row.",
            )
            return
        if self._thread is not None:
            return

        self.refresh_button.setEnabled(False)
        self.status_label.setText("Running analysis and updating prices…")
        thread = QThread(self)
        worker = AnalysisWorker(str(path))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._analysis_finished)
        worker.failed.connect(self._analysis_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._analysis_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _analysis_finished(self, report: PortfolioReport) -> None:
        """Display a successfully generated report."""
        self.set_report(report)
        self.status_label.setText("Analysis complete.")

    def _analysis_failed(self, message: str) -> None:
        """Show a backend analysis failure without terminating the GUI."""
        QMessageBox.critical(self, "Analysis failed", message)
        self.status_label.setText("Analysis failed.")

    def _analysis_thread_finished(self) -> None:
        """Release worker references after a background analysis completes."""
        self.refresh_button.setEnabled(True)
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None

    def set_report(self, report: PortfolioReport) -> None:
        """Populate all dashboard widgets from a portfolio report."""
        self.report = report
        self.total_value_label.setText(f"Total value: ${report.total_value:,.2f}")
        self.cash_label.setText(f"Cash: ${report.cash:,.2f}")
        self.invested_label.setText(f"Invested: ${report.invested_value:,.2f}")
        self.price_date_label.setText(f"Latest prices: {report.latest_price_date[:10]}")

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(report.tickers))
        for source_row, item in enumerate(report.tickers):
            values = (
                item.ticker,
                f"{item.held_quantity:g}",
                f"${item.current_price:,.2f}",
                f"${item.market_value:,.2f}",
                probability_label(item),
                item.recommendation,
                str(len(item.warnings)),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, source_row)
                self.table.setItem(source_row, column, cell)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

        warning_text = "\n".join(f"• {item}" for item in report.warnings) if report.warnings else "No report-level warnings."
        self.warnings_text.setPlainText(warning_text)
        if report.tickers:
            self.table.selectRow(0)
        else:
            self._clear_details()

    def _show_selected_ticker(self) -> None:
        """Update the detail panel for the selected recommendation row."""
        if self.report is None:
            return
        selected = self.table.selectedItems()
        if not selected:
            return
        source_row = selected[0].data(Qt.ItemDataRole.UserRole)
        if source_row is None:
            source_row = selected[0].row()
        item = self.report.tickers[int(source_row)]
        self.detail_ticker.setText(item.ticker)
        self.detail_probability.setText(probability_label(item))
        self.detail_recommendation.setText(item.recommendation)
        baskets = [", ".join(basket) for basket in item.basket_memberships]
        self.detail_baskets.setText("; ".join(baskets) if baskets else "None")
        text = item.explanation
        if item.warnings:
            text += "\n\nWarnings:\n" + "\n".join(f"• {warning}" for warning in item.warnings)
        self.explanation.setPlainText(text)

    def _clear_details(self) -> None:
        """Reset the ticker detail widgets."""
        self.detail_ticker.setText("—")
        self.detail_probability.setText("—")
        self.detail_recommendation.setText("—")
        self.detail_baskets.setText("—")
        self.explanation.clear()


def main() -> None:
    """Launch the Cobasket PyQt dashboard."""
    app = QApplication(sys.argv)
    window = CobasketDashboard()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
