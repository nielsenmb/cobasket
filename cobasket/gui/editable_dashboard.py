"""Dashboard extension with editable state and diagnostic analysis tools."""

from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from cobasket.history import RecommendationHistoryStore
from cobasket.workflow import PortfolioReport

from .config_editor import ConfigEditorDialog
from .continuous_walk_forward_dialog import ContinuousWalkForwardDialog
from .dashboard import CobasketDashboard, classify_cobasket_json
from .history_dialog import RecommendationHistoryDialog
from .investigation_dialog import BasketInvestigationDialog
from .repeated_walk_forward_dialog import RepeatedWalkForwardDialog
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

        repeated_action = portfolio_menu.addAction("Repeated walk-forward experiments…")
        repeated_action.triggered.connect(self.open_repeated_walk_forward)

        continuous_action = portfolio_menu.addAction("Continuous walk-forward deployment…")
        continuous_action.triggered.connect(self.open_continuous_walk_forward)

        validation_action = portfolio_menu.addAction("Historical validation…")
        validation_action.triggered.connect(self.open_validation)
        self._history_dialog: RecommendationHistoryDialog | None = None
        self._investigation_dialog: BasketInvestigationDialog | None = None
        self._strategy_dialog: StrategySimulationDialog | None = None
        self._experiment_dialog: StrategyExperimentDialog | None = None
        self._repeated_dialog: RepeatedWalkForwardDialog | None = None
        self._continuous_dialog: ContinuousWalkForwardDialog | None = None
        self._validation_dialog: ValidationDialog | None = None

    def _configuration_path(self) -> Path | None:
        """Return the selected portfolio configuration path when valid."""
        text = self.path_edit.text().strip()
        if not text:
            return None
        path = Path(text)
        if not path.exists():
            return None
        try:
            return path if classify_cobasket_json(path) == "portfolio" else None
        except Exception:
            return None

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

    def _require_configuration(self, action: str) -> Path | None:
        """Return a valid configuration path or display an actionable warning."""
        path = self._configuration_path()
        if path is not None:
            return path
        QMessageBox.warning(
            self,
            "Portfolio configuration required",
            f"{action} requires a portfolio configuration JSON file. Choose portfolio.json in the Portfolio configuration row first.",
        )
        return None

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
        path = self._require_configuration("Editing the portfolio and watchlist")
        if path is None:
            return
        dialog = ConfigEditorDialog(path, self)
        if dialog.exec():
            self.status_label.setText(
                "Portfolio configuration saved. Run analysis to refresh recommendations."
            )

    def investigate_selected_ticker(self) -> None:
        """Open diagnostic plots for the currently selected ticker and its baskets."""
        config_path = self._require_configuration("Ticker investigation")
        if config_path is None:
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
        try:
            dialog = RecommendationHistoryDialog(
                self._history_path(), ticker=self._selected_ticker(), parent=self
            )
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open recommendation history", str(exc))
            return
        self._history_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_history_dialog", None))
        dialog.show()

    def open_strategy_simulation(self) -> None:
        """Open the end-to-end historical basket strategy simulator."""
        config_path = self._require_configuration("Basket strategy simulation")
        if config_path is None:
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

    def open_repeated_walk_forward(self) -> None:
        """Open repeated strategy selection across chronological market regimes."""
        dialog = RepeatedWalkForwardDialog(self)
        self._repeated_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_repeated_dialog", None))
        dialog.show()

    def open_continuous_walk_forward(self) -> None:
        """Open continuous strategy reselection with one persistent account."""
        dialog = ContinuousWalkForwardDialog(self)
        self._continuous_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_continuous_dialog", None))
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
