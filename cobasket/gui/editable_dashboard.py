"""Dashboard extension with editable portfolio and watchlist state."""

from __future__ import annotations

from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from .config_editor import ConfigEditorDialog
from .dashboard import CobasketDashboard


class EditableCobasketDashboard(CobasketDashboard):
    """Cobasket dashboard with portfolio and watchlist editing actions."""

    def __init__(self) -> None:
        super().__init__()
        portfolio_menu = self.menuBar().addMenu("Portfolio")
        edit_action = portfolio_menu.addAction("Edit portfolio and watchlist…")
        edit_action.triggered.connect(self.edit_configuration)

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


def main() -> None:
    """Launch the editable Cobasket dashboard."""
    app = QApplication(sys.argv)
    window = EditableCobasketDashboard()
    window.show()
    raise SystemExit(app.exec())
