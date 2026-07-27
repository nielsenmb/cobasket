"""Dashboard extension with editable state and diagnostic analysis tools."""

from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from cobasket.history import RecommendationHistoryStore
from cobasket.workflow import PortfolioReport

from .config_editor import ConfigEditorDialog
from .dashboard import CobasketDashboard
from .history_dialog import RecommendationHistoryDialog
from .investigation_dialog import BasketInvestigationDialog
from .strategy_dialog import StrategySimulationDialog
from .strategy_experiment_dialog import StrategyExperimentDialog
from .validation_dialog import ValidationDialog


class EditableCobasketDashboard(CobasketDashboard):
    """Cobasket dashboard with editing, investigation, and validation actions."""

    def __init__(self) -> None:
        super().__init__()
        portfolio_menu = self.menuBar().addMenu("Portfolio")
        edit_action = portfolio_menu.addAction("Edit portfolio and watchlist…")
        edit_action.triggered.connect(self.edit_configuration)

        investigate_action = portfolio_menu.addAction("Investigate selected ticker…")
        investigate_action.triggered.connect(self.investigate_selected_ticker)
        self.table.cellDoubleClicked.connect(lambda *_: self.investigate_selected_ticker())

        history_action = portfolio_menu.addAction("Recommendation history…")
        history_action.triggered.connect(self.open_recommendation_history)

        simulation_action = portfolio_menu.addAction("Simulate basket strategy…")
        simulation_action.triggered.connect(self.open_strategy_simulation)

        experiment_action = portfolio_menu.addAction("Strategy experiments…")
        experiment_action.triggered.connect(self.open_strategy_experiments)

        validation_action = portfolio_menu.addAction("Historical validation…")
        validation_action.triggered.connect(self.open_validation)
        self._history_dialog: RecommendationHistoryDialog | None = None
        self._investigation_dialog: BasketInvestigationDialog | None = None
        self._strategy_dialog: StrategySimulationDialog | None = None
        self._experiment_dialog: StrategyExperimentDialog | None = None
        self._validation_dialog: ValidationDialog | None = None

    def _configuration_path(self) -> Path | None:
        """Return the selected configuration path when it exists."""
        path = Path(self.path_edit.text().strip())
        return path if path.exists() else None

    def _history_path(self) -> Path:
        """Return the history database beside the selected configuration file."""
        config_path = self._configuration_path()
        directory = config_path.parent if config_path is not None else Path.cwd()
        return directory / "cobasket_history.sqlite"

    def _selected_ticker(self) -> str | None:
        """Return the ticker represented by the selected dashboard row."""
        selected = self.table.selectedItems()
        if not selected or self.report is None:
            return None
        source_row = selected[0].data(256)
        if source_row is None:
            source_row = selected[0].row()
        return self.report.tickers[int(source_row)].ticker

    def _analysis_finished(self, report: PortfolioReport) -> None:
        """Display and persist a successfully generated live report."""
        super()._analysis_finished(report)
        try:
            RecommendationHistoryStore(self._history_path()).record_report(report)
        except Exception as exc:
            self.status_label.setText(
                f"Analysis complete, but history storage failed: {type(exc).__name__}: {exc}"
            )
            return
        self.status_label.setText("Analysis complete and recommendation history saved.")

    def edit_configuration(self) -> None:
        """Open the current portfolio configuration in the editor."""
        path = self._configuration_path()
        if path is None:
            QMessageBox.warning(
                self,
                "Missing configuration",
                "Choose an existing portfolio configuration JSON file first.",
            )
            return
        dialog = ConfigEditorDialog(path, self)
        if dialog.exec():
            self.status_label.setText(
                "Portfolio configuration saved. Run analysis to refresh recommendations."
            )

    def investigate_selected_ticker(self) -> None:
        """Open diagnostic plots for the currently selected ticker and its baskets."""
        config_path = self._configuration_path()
        if config_path is None:
            QMessageBox.warning(
                self,
                "Missing configuration",
                "Choose an existing portfolio configuration JSON file first.",
            )
            return
        ticker = self._selected_ticker()
        if ticker is None:
            QMessageBox.information(self, "No ticker selected", "Select a ticker row first.")
            return
        try:
            dialog = BasketInvestigationDialog(config_path, ticker, self)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot investigate ticker", str(exc))
            return
        self._investigation_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_investigation_dialog", None))
        dialog.show()

    def open_recommendation_history(self) -> None:
        """Open the stored model, action, and outcome history timeline."""
        dialog = RecommendationHistoryDialog(
            self._history_path(),
            ticker=self._selected_ticker(),
            parent=self,
        )
        self._history_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_history_dialog", None))
        dialog.show()

    def open_strategy_simulation(self) -> None:
        """Open the end-to-end historical basket strategy simulator."""
        config_path = self._configuration_path()
        if config_path is None:
            QMessageBox.warning(
                self,
                "Missing configuration",
                "Choose an existing portfolio configuration JSON file first.",
            )
            return
        try:
            dialog = StrategySimulationDialog(config_path, self)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open simulator", str(exc))
            return
        self._strategy_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_strategy_dialog", None))
        dialog.show()

    def open_strategy_experiments(self) -> None:
        """Open the declarative rule editor and controlled experiment view."""
        dialog = StrategyExperimentDialog(self)
        self._experiment_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_experiment_dialog", None))
        dialog.show()

    def open_validation(self) -> None:
        """Open the historical policy and calibration validation dashboard."""
        dialog = ValidationDialog(self)
        self._validation_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_validation_dialog", None))
        dialog.show()


def main() -> None:
    """Launch the editable Cobasket dashboard."""
    app = QApplication(sys.argv)
    window = EditableCobasketDashboard()
    window.show()
    raise SystemExit(app.exec())
