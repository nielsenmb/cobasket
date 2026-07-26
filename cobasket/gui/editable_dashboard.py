"""Dashboard extension with editable state and basket investigation tools."""

from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from .config_editor import ConfigEditorDialog
from .dashboard import CobasketDashboard
from .investigation_dialog import BasketInvestigationDialog


class EditableCobasketDashboard(CobasketDashboard):
    """Cobasket dashboard with editing and diagnostic investigation actions."""

    def __init__(self) -> None:
        super().__init__()
        portfolio_menu = self.menuBar().addMenu("Portfolio")
        edit_action = portfolio_menu.addAction("Edit portfolio and watchlist…")
        edit_action.triggered.connect(self.edit_configuration)

        investigate_action = portfolio_menu.addAction("Investigate selected ticker…")
        investigate_action.triggered.connect(self.investigate_selected_ticker)
        self.table.cellDoubleClicked.connect(lambda *_: self.investigate_selected_ticker())
        self._investigation_dialog: BasketInvestigationDialog | None = None

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


def main() -> None:
    """Launch the editable Cobasket dashboard."""
    app = QApplication(sys.argv)
    window = EditableCobasketDashboard()
    window.show()
    raise SystemExit(app.exec())
