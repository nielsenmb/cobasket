"""GUI controls for workspace freshness recommendations."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .workspace_state import FreshnessPolicy, freshness_policy_path, load_freshness_policy


class FreshnessDialog(QDialog):
    """Edit persistent refresh intervals for one Cobasket workspace."""

    def __init__(self, workspace: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.workspace = Path(workspace).expanduser().resolve()
        self.setWindowTitle("Workspace freshness settings")
        self._build_ui()
        self._load_policy()

    def _build_ui(self) -> None:
        """Construct refresh-interval controls."""
        outer = QVBoxLayout(self)
        explanation = QLabel(
            "These intervals control when Cobasket recommends refreshing existing artifacts. "
            "They do not change the statistical models themselves."
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        form = QFormLayout()
        self.report_spin = self._days_spinbox()
        self.validation_spin = self._days_spinbox()
        self.calibration_spin = self._days_spinbox()
        self.discovery_spin = self._days_spinbox()
        form.addRow("Live report refresh", self.report_spin)
        form.addRow("Validation refresh", self.validation_spin)
        form.addRow("Calibration refresh", self.calibration_spin)
        form.addRow("Discovery reminder", self.discovery_spin)
        outer.addLayout(form)

        defaults_button = QPushButton("Restore defaults")
        defaults_button.clicked.connect(self._set_defaults)
        outer.addWidget(defaults_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_policy)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _days_spinbox() -> QDoubleSpinBox:
        """Create a spin box for a positive refresh interval.

        Returns
        -------
        QDoubleSpinBox
            Configured interval control expressed in days.
        """
        spin = QDoubleSpinBox()
        spin.setRange(1.0, 3650.0)
        spin.setDecimals(0)
        spin.setSuffix(" days")
        return spin

    def _load_policy(self) -> None:
        """Populate controls from the stored workspace policy."""
        self._set_policy(load_freshness_policy(self.workspace))

    def _set_policy(self, policy: FreshnessPolicy) -> None:
        """Populate controls from a policy object.

        Parameters
        ----------
        policy
            Freshness intervals to display.
        """
        self.report_spin.setValue(policy.report_days)
        self.validation_spin.setValue(policy.validation_days)
        self.calibration_spin.setValue(policy.calibration_days)
        self.discovery_spin.setValue(policy.discovery_days)

    def _set_defaults(self) -> None:
        """Restore the default refresh intervals in the editor."""
        self._set_policy(FreshnessPolicy())

    def _save_policy(self) -> None:
        """Persist the edited policy and close the dialog."""
        policy = FreshnessPolicy(
            report_days=self.report_spin.value(),
            validation_days=self.validation_spin.value(),
            calibration_days=self.calibration_spin.value(),
            discovery_days=self.discovery_spin.value(),
        )
        policy.save(freshness_policy_path(self.workspace))
        self.accept()
