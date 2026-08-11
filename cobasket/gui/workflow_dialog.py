"""Guided GUI orchestration for the Cobasket discovery-to-report workflow."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .freshness_dialog import FreshnessDialog
from .workspace_state import WorkspaceState, inspect_workspace


_STAGE_LABELS = {
    "discover": "Discover baskets",
    "validate": "Validate baskets",
    "calibrate": "Calibrate probabilities",
    "report": "Generate live report",
}


def workflow_command(
    stage: str,
    *,
    universe: str,
    period: str,
    trading212_only: bool = False,
) -> tuple[str, ...]:
    """Build one Cobasket workflow command for execution in a workspace.

    Parameters
    ----------
    stage
        Workflow stage: ``discover``, ``validate``, ``calibrate``, or ``report``.
    universe
        Built-in discovery universe.
    period
        Historical price period passed to discovery.
    trading212_only
        Add the account-accessible Trading 212 stock filter to discovery.

    Returns
    -------
    tuple of str
        Python module invocation arguments excluding the Python executable.

    Raises
    ------
    ValueError
        If ``stage`` is unknown.
    """
    commands = {
        "discover": (
            "-m",
            "cobasket.discovery_cli",
            "--universe",
            universe,
            "--period",
            period,
            "--watchlist-out",
            "discovered_watchlist.json",
            "--table-out",
            "discovery_results.csv",
            "--portfolio",
            "portfolio.json",
            "--update-portfolio",
        ),
        "validate": (
            "-m",
            "cobasket.validation_cli",
            "--portfolio",
            "portfolio.json",
            "--output",
            "basket_validation.json",
            "--update-portfolio",
        ),
        "calibrate": (
            "-m",
            "cobasket.basket_calibration_cli",
            "--portfolio",
            "portfolio.json",
            "--validation",
            "basket_validation.json",
            "--output",
            "basket_calibration.json",
            "--update-portfolio",
        ),
        "report": (
            "-m",
            "cobasket.report_cli",
            "--portfolio",
            "portfolio.json",
            "--output",
            "report.json",
        ),
    }
    try:
        command = commands[stage]
    except KeyError as exc:
        raise ValueError(f"unknown workflow stage: {stage}") from exc
    if stage == "discover" and trading212_only:
        command = (*command, "--trading212-only")
    return command


class WorkflowDialog(QDialog):
    """Guide the user through the normal Cobasket workspace workflow."""

    report_ready = pyqtSignal(str)
    portfolio_ready = pyqtSignal(str)

    def __init__(self, workspace: str | Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cobasket workspace")
        self.resize(900, 700)
        self._process: QProcess | None = None
        self._queue: deque[str] = deque()
        self._active_stage: str | None = None
        self._state: WorkspaceState | None = None
        self._build_ui()
        initial = Path(workspace) if workspace is not None else Path.cwd()
        self.workspace_edit.setText(str(initial.resolve()))
        self._refresh_status()

    def _build_ui(self) -> None:
        """Construct workspace controls, status, and process output."""
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Choose a workspace directory. Cobasket will inspect what is already there and recommend "
            "the next step. New workspaces start with discovery; complete workspaces normally only "
            "need recommendation refreshes."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        workspace_row = QHBoxLayout()
        self.workspace_edit = QLineEdit()
        browse_button = QPushButton("Browse…")
        workspace_row.addWidget(self.workspace_edit, 1)
        workspace_row.addWidget(browse_button)
        form.addRow("Workspace", workspace_row)

        self.universe_combo = QComboBox()
        self.universe_combo.addItems(
            (
                "sp1500",
                "sp500",
                "sp400",
                "sp600",
                "nasdaq100",
                "ftse350",
                "ftse100",
                "ftse250",
                "eurostoxx50",
            )
        )
        self.universe_combo.setCurrentText("sp500")
        form.addRow("Discovery universe", self.universe_combo)
        self.period_combo = QComboBox()
        self.period_combo.addItems(("5y", "2y", "10y"))
        form.addRow("Discovery period", self.period_combo)
        self.trading212_checkbox = QCheckBox("Only stocks available in my Trading 212 account")
        self.trading212_checkbox.setToolTip(
            "Uses the Trading 212 accessible-instruments API. Set TRADING212_API_KEY and "
            "TRADING212_API_SECRET in the environment before launching Cobasket."
        )
        form.addRow("Broker filter", self.trading212_checkbox)
        outer.addLayout(form)

        state_box = QGroupBox("Workspace status")
        state_layout = QVBoxLayout(state_box)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        state_layout.addWidget(self.status_label)
        self.stage_status_label = QLabel()
        self.stage_status_label.setWordWrap(True)
        state_layout.addWidget(self.stage_status_label)
        outer.addWidget(state_box)

        primary_row = QHBoxLayout()
        self.next_button = QPushButton("Next step")
        self.update_button = QPushButton("Update required stages to report")
        primary_row.addWidget(self.next_button, 1)
        primary_row.addWidget(self.update_button)
        outer.addLayout(primary_row)

        secondary_row = QHBoxLayout()
        self.open_portfolio_button = QPushButton("Edit holdings / portfolio…")
        self.rediscover_button = QPushButton("Re-run discovery…")
        self.load_report_button = QPushButton("Show current report")
        self.freshness_button = QPushButton("Freshness settings…")
        close_button = QPushButton("Close")
        secondary_row.addWidget(self.open_portfolio_button)
        secondary_row.addWidget(self.rediscover_button)
        secondary_row.addWidget(self.load_report_button)
        secondary_row.addWidget(self.freshness_button)
        secondary_row.addStretch(1)
        secondary_row.addWidget(close_button)
        outer.addLayout(secondary_row)

        output_box = QGroupBox("Details")
        output_layout = QVBoxLayout(output_box)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Output from the current workflow step will appear here.")
        output_layout.addWidget(self.output)
        outer.addWidget(output_box, 1)

        browse_button.clicked.connect(self._browse_workspace)
        self.workspace_edit.editingFinished.connect(self._refresh_status)
        self.next_button.clicked.connect(self._run_recommended_step)
        self.update_button.clicked.connect(self._run_required_updates)
        self.rediscover_button.clicked.connect(self._confirm_rediscovery)
        self.open_portfolio_button.clicked.connect(self._edit_portfolio)
        self.load_report_button.clicked.connect(self._emit_report)
        self.freshness_button.clicked.connect(self._edit_freshness)
        close_button.clicked.connect(self.reject)

    def workspace(self) -> Path:
        """Return the selected workflow directory."""
        return Path(self.workspace_edit.text().strip()).expanduser().resolve()

    def _browse_workspace(self) -> None:
        """Choose a workflow directory."""
        path = QFileDialog.getExistingDirectory(self, "Choose Cobasket workspace", str(self.workspace()))
        if path:
            self.workspace_edit.setText(path)
            self._refresh_status()

    def _run_recommended_step(self) -> None:
        """Run the single next stage recommended for the current workspace."""
        state = inspect_workspace(self.workspace())
        self._start_sequence((state.next_stage,))

    def _run_required_updates(self) -> None:
        """Run all currently required downstream stages through a live report."""
        state = inspect_workspace(self.workspace())
        self._start_sequence(state.update_stages)

    def _confirm_rediscovery(self) -> None:
        """Confirm and start a fresh basket discovery for the workspace."""
        workspace = self.workspace()
        if (workspace / "portfolio.json").exists():
            answer = QMessageBox.question(
                self,
                "Re-run discovery",
                "Discovery will replace the workspace watchlist and invalidate downstream validation "
                "and calibration for the old baskets. Holdings and cash are preserved. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._start_sequence(("discover",))

    def _edit_freshness(self) -> None:
        """Edit persistent age-based refresh recommendations for the workspace."""
        dialog = FreshnessDialog(self.workspace(), self)
        if dialog.exec():
            self._refresh_status()

    def _start_sequence(self, stages: tuple[str, ...]) -> None:
        """Queue one or more workflow stages."""
        if self._process is not None or not stages:
            return
        workspace = self.workspace()
        workspace.mkdir(parents=True, exist_ok=True)
        if stages[0] != "discover" and not (workspace / "portfolio.json").exists():
            QMessageBox.warning(
                self,
                "Portfolio required",
                "This workspace has no portfolio.json. Run discovery first.",
            )
            return
        self._queue = deque(stages)
        self.output.clear()
        self._set_running(True)
        self._run_next()

    def _run_next(self) -> None:
        """Launch the next queued CLI stage."""
        if not self._queue:
            self._active_stage = None
            self._set_running(False)
            self._refresh_status()
            portfolio = self.workspace() / "portfolio.json"
            if portfolio.exists():
                self.portfolio_ready.emit(str(portfolio))
            report = self.workspace() / "report.json"
            if report.exists():
                self.report_ready.emit(str(report))
            return

        stage = self._queue.popleft()
        if stage != "discover" and not (self.workspace() / "portfolio.json").exists():
            self._queue.clear()
            self._active_stage = None
            self._set_running(False)
            self._refresh_status()
            QMessageBox.warning(
                self,
                "No discovered portfolio",
                "The previous stage did not produce portfolio.json, usually because no promising basket passed discovery.",
            )
            return
        self._active_stage = stage
        self.output.append(f"\n=== {_STAGE_LABELS[stage]} ===\n")
        command = workflow_command(
            stage,
            universe=self.universe_combo.currentText(),
            period=self.period_combo.currentText(),
            trading212_only=self.trading212_checkbox.isChecked(),
        )
        process = QProcess(self)
        process.setWorkingDirectory(str(self.workspace()))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        process.start(sys.executable, list(command))

    def _read_output(self) -> None:
        """Append available process output to the console."""
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        if text:
            cursor = self.output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.output.setTextCursor(cursor)
            self.output.insertPlainText(text)
            self.output.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _status) -> None:
        """Advance the workflow after a process exits."""
        self._read_output()
        stage = self._active_stage or "workflow"
        self._process = None
        if exit_code != 0:
            self._queue.clear()
            self._set_running(False)
            self._refresh_status()
            QMessageBox.critical(
                self,
                "Workflow stage failed",
                f"{_STAGE_LABELS.get(stage, stage)} exited with code {exit_code}. See Details for the error output.",
            )
            return
        self._refresh_status()
        self._run_next()

    def _process_error(self, _error) -> None:
        """Report a process-start/runtime error without crashing the GUI."""
        if self._process is None:
            return
        self.output.append(f"\nProcess error: {self._process.errorString()}")

    def _set_running(self, running: bool) -> None:
        """Enable or disable controls while a stage is executing."""
        for button in (
            self.next_button,
            self.update_button,
            self.rediscover_button,
            self.open_portfolio_button,
            self.load_report_button,
            self.freshness_button,
        ):
            button.setEnabled(not running)
        self.workspace_edit.setEnabled(not running)
        self.universe_combo.setEnabled(not running)
        self.period_combo.setEnabled(not running)
        self.trading212_checkbox.setEnabled(not running)
        if not running:
            self._refresh_status()

    def _refresh_status(self) -> None:
        """Inspect the workspace and update the recommended action."""
        state = inspect_workspace(self.workspace())
        self._state = state
        prefix = f"Running {_STAGE_LABELS[self._active_stage]}…\n" if self._process and self._active_stage else ""
        self.status_label.setText(prefix + state.summary)
        stage_lines = [f"{index}. {name}: {status}" for index, (name, status) in enumerate(state.stage_statuses, 1)]
        self.stage_status_label.setText("\n".join(stage_lines))
        self.next_button.setText(state.next_label)

        has_portfolio = (self.workspace() / "portfolio.json").exists()
        has_report = (self.workspace() / "report.json").exists()
        idle = self._process is None
        multiple_updates = len(state.update_stages) > 1

        self.next_button.setEnabled(idle)
        self.update_button.setVisible(multiple_updates)
        self.update_button.setEnabled(multiple_updates and idle)
        self.rediscover_button.setVisible(state.name != "empty")
        self.rediscover_button.setEnabled(state.name != "empty" and idle)
        self.open_portfolio_button.setVisible(has_portfolio)
        self.open_portfolio_button.setEnabled(has_portfolio and idle)
        self.load_report_button.setVisible(has_report)
        self.load_report_button.setEnabled(has_report and idle)
        self.freshness_button.setEnabled(idle)

    def _edit_portfolio(self) -> None:
        """Open the generated portfolio in the existing configuration editor."""
        portfolio = self.workspace() / "portfolio.json"
        if not portfolio.exists():
            return
        from .config_editor import ConfigEditorDialog

        dialog = ConfigEditorDialog(portfolio, self)
        if dialog.exec():
            self.portfolio_ready.emit(str(portfolio))
            self._refresh_status()

    def _emit_report(self) -> None:
        """Emit the generated report path for loading by the parent dashboard."""
        report = self.workspace() / "report.json"
        if report.exists():
            self.report_ready.emit(str(report))

    def closeEvent(self, event) -> None:
        """Prevent accidental closure while a workflow process is running."""
        if self._process is not None:
            QMessageBox.information(self, "Workflow running", "Finish the current workflow before closing.")
            event.ignore()
            return
        super().closeEvent(event)
