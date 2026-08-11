"""Guided GUI orchestration for the Cobasket discovery-to-report workflow."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


_STAGE_LABELS = {
    "discover": "Discover baskets",
    "validate": "Validate baskets",
    "calibrate": "Calibrate probabilities",
    "report": "Generate live report",
}


def workflow_command(stage: str, *, universe: str, period: str) -> tuple[str, ...]:
    """Build one Cobasket workflow command for execution in a workspace.

    Parameters
    ----------
    stage
        Workflow stage: ``discover``, ``validate``, ``calibrate``, or ``report``.
    universe
        Built-in discovery universe.
    period
        Historical price period passed to discovery.

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
        return commands[stage]
    except KeyError as exc:
        raise ValueError(f"unknown workflow stage: {stage}") from exc


class WorkflowDialog(QDialog):
    """Run discovery, validation, calibration, and reporting from one window."""

    report_ready = pyqtSignal(str)
    portfolio_ready = pyqtSignal(str)

    def __init__(self, workspace: str | Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cobasket guided workflow")
        self.resize(900, 700)
        self._process: QProcess | None = None
        self._queue: deque[str] = deque()
        self._active_stage: str | None = None
        self._build_ui()
        initial = Path(workspace) if workspace is not None else Path.cwd()
        self.workspace_edit.setText(str(initial.resolve()))
        self._refresh_status()

    def _build_ui(self) -> None:
        """Construct workflow controls and output console."""
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Use this window for the normal Cobasket path: discover persistent baskets, "
            "validate them, fit basket-specific probability calibration, then generate a live report."
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
        self.universe_combo.addItems(("sp500", "nasdaq100", "ftse100", "eurostoxx50"))
        form.addRow("Discovery universe", self.universe_combo)
        self.period_combo = QComboBox()
        self.period_combo.addItems(("5y", "2y", "10y"))
        form.addRow("Historical period", self.period_combo)
        outer.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        stage_row = QHBoxLayout()
        self.discover_button = QPushButton("1. Discover")
        self.validate_button = QPushButton("2. Validate")
        self.calibrate_button = QPushButton("3. Calibrate")
        self.report_button = QPushButton("4. Report")
        self.run_all_button = QPushButton("Run all")
        for button in (
            self.discover_button,
            self.validate_button,
            self.calibrate_button,
            self.report_button,
            self.run_all_button,
        ):
            stage_row.addWidget(button)
        outer.addLayout(stage_row)

        edit_row = QHBoxLayout()
        self.open_portfolio_button = QPushButton("Edit portfolio/holdings…")
        self.load_report_button = QPushButton("Load generated report")
        close_button = QPushButton("Close")
        edit_row.addWidget(self.open_portfolio_button)
        edit_row.addWidget(self.load_report_button)
        edit_row.addStretch(1)
        edit_row.addWidget(close_button)
        outer.addLayout(edit_row)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Workflow output will appear here.")
        outer.addWidget(self.output, 1)

        browse_button.clicked.connect(self._browse_workspace)
        self.discover_button.clicked.connect(lambda: self._start_sequence(("discover",)))
        self.validate_button.clicked.connect(lambda: self._start_sequence(("validate",)))
        self.calibrate_button.clicked.connect(lambda: self._start_sequence(("calibrate",)))
        self.report_button.clicked.connect(lambda: self._start_sequence(("report",)))
        self.run_all_button.clicked.connect(
            lambda: self._start_sequence(("discover", "validate", "calibrate", "report"))
        )
        self.open_portfolio_button.clicked.connect(self._edit_portfolio)
        self.load_report_button.clicked.connect(self._emit_report)
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

    def _start_sequence(self, stages: tuple[str, ...]) -> None:
        """Queue one or more workflow stages."""
        if self._process is not None:
            return
        workspace = self.workspace()
        workspace.mkdir(parents=True, exist_ok=True)
        if stages[0] != "discover" and not (workspace / "portfolio.json").exists():
            QMessageBox.warning(
                self,
                "Portfolio required",
                "portfolio.json is missing. Run discovery first or choose a workspace containing a Cobasket portfolio.",
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
        self.output.append(f"\n=== {_STAGE_LABELS[stage]} ===")
        command = workflow_command(
            stage,
            universe=self.universe_combo.currentText(),
            period=self.period_combo.currentText(),
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
            self.output.moveCursor(self.output.textCursor().MoveOperation.End)
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
                f"{_STAGE_LABELS.get(stage, stage)} exited with code {exit_code}. See the output log for details.",
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
            self.discover_button,
            self.validate_button,
            self.calibrate_button,
            self.report_button,
            self.run_all_button,
        ):
            button.setEnabled(not running)
        self.workspace_edit.setEnabled(not running)
        self.universe_combo.setEnabled(not running)
        self.period_combo.setEnabled(not running)

    def _refresh_status(self) -> None:
        """Summarize which workflow artifacts exist in the workspace."""
        workspace = self.workspace()
        files = {
            "portfolio": workspace / "portfolio.json",
            "watchlist": workspace / "discovered_watchlist.json",
            "validation": workspace / "basket_validation.json",
            "calibration": workspace / "basket_calibration.json",
            "report": workspace / "report.json",
        }
        available = [name for name, path in files.items() if path.exists()]
        if self._process is not None and self._active_stage is not None:
            prefix = f"Running {_STAGE_LABELS[self._active_stage]}… "
        else:
            prefix = ""
        suffix = ", ".join(available) if available else "no workflow files yet"
        self.status_label.setText(prefix + "Workspace contains: " + suffix + ".")
        self.open_portfolio_button.setEnabled(files["portfolio"].exists() and self._process is None)
        self.load_report_button.setEnabled(files["report"].exists() and self._process is None)

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
