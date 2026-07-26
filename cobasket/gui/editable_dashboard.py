"""Dashboard extension with editable state and diagnostic analysis tools."""

from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from .config_editor import ConfigEditorDialog
from .dashboard import CobasketDashboard
from .investigation_dialog import BasketInvestigationDialog
from .strategy_dialog import StrategySimulationDialog
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

        simulation_action = portfolio_menu.addAction("Simulate basket strategy…")
        simulation_action.triggered.connect(self.open_strategy_simulation)

        validation_action = portfolio_menu.addAction("Historical validation…")
        validation_action.triggered.connect(self.open_validation)
        self._investigation_dialog: BasketInvestigationDialog | None = None
        self._strategy_dialog: StrategySimulationDialog | None = None
        self._validation_dialog: ValidationDialog | None = None

    def edit_configuration(self) -> None:
        """Open the current portfolio configuration in the editor."""
        path = Path(self.path_edit.text().strip())
        if not path.exists():
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
        config_path = Path(self.path_edit.text().strip())
        if not config_path.exists():
            QMessageBox.warning(
                self,
                "Missing configuration",
                "Choose an existing portfolio configuration JSON file first.",
            )
            return
        selected = self.table.selectedItems()
        if not selected or self.report is None:
            QMessageBox.information(self, "No ticker selected", "Select a ticker row first.")
            return
        source_row = selected[0].data(256)
        if source_row is None:
            source_row = selected[0].row()
        ticker = self.report.tickers[int(source_row)].ticker
        try:
            dialog = BasketInvestigationDialog(config_path, ticker, self)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot investigate ticker", str(exc))
            return
        self._investigation_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_investigation_dialog", None))
        dialog.show()

    def open_strategy_simulation(self) -> None:
        """Open the end-to-end historical basket strategy simulator."""
        config_path = Path(self.path_edit.text().strip())
        if not config_path.exists():
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
