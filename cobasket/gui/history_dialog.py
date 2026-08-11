"""PyQt recommendation-history and user-decision timeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cobasket.history import RecommendationHistoryStore


def _format_optional_number(value, formatter) -> str:
    """Format a nullable numeric database value safely.

    Parameters
    ----------
    value
        Database value that may be ``None`` or ``NaN``.
    formatter
        Callable accepting a finite floating-point value.

    Returns
    -------
    str
        Formatted value or an em dash when missing.
    """
    return "—" if pd.isna(value) else formatter(float(value))


class RecommendationHistoryDialog(QDialog):
    """Display model recommendations, transitions, actions, and outcomes."""

    def __init__(self, database_path: str | Path, ticker: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.store = RecommendationHistoryStore(database_path)
        self.setWindowTitle("Recommendation history")
        self.resize(1050, 760)
        self._build_ui()
        self._load_tickers(ticker)

    def _build_ui(self) -> None:
        """Construct timeline, action, and outcome widgets."""
        outer = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Ticker"))
        self.ticker_combo = QComboBox()
        controls.addWidget(self.ticker_combo)
        refresh = QPushButton("Refresh")
        controls.addWidget(refresh)
        controls.addStretch(1)
        outer.addLayout(controls)

        self.summary = QLabel("No stored recommendation history.")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)

        tabs = QTabWidget()
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        timeline_layout.addWidget(self.canvas, 2)
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            ["Generated", "Recommendation", "Probability", "Evidence", "Held", "Price", "Transition"]
        )
        timeline_layout.addWidget(self.history_table, 1)
        tabs.addTab(timeline_widget, "Model history")

        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        form = QFormLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems(["no action", "buy", "add", "reduce", "sell"])
        self.quantity_spin = QDoubleSpinBox()
        self.quantity_spin.setRange(0.0, 1e9)
        self.quantity_spin.setDecimals(6)
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0.0, 1e9)
        self.price_spin.setDecimals(4)
        self.note_edit = QLineEdit()
        form.addRow("Action", self.action_combo)
        form.addRow("Quantity", self.quantity_spin)
        form.addRow("Price", self.price_spin)
        form.addRow("Note", self.note_edit)
        action_layout.addLayout(form)
        record = QPushButton("Record action")
        action_layout.addWidget(record)
        self.action_table = QTableWidget(0, 6)
        self.action_table.setHorizontalHeaderLabels(["Time", "Action", "Quantity", "Price", "Note", "ID"])
        action_layout.addWidget(self.action_table)
        tabs.addTab(action_widget, "My decisions")

        outcome_widget = QWidget()
        outcome_layout = QVBoxLayout(outcome_widget)
        self.outcome_table = QTableWidget(0, 5)
        self.outcome_table.setHorizontalHeaderLabels(
            ["Recommendation date", "Recommendation", "Horizon", "Outcome date", "Return"]
        )
        outcome_layout.addWidget(self.outcome_table)
        self.outcome_help = QTextEdit()
        self.outcome_help.setReadOnly(True)
        self.outcome_help.setMaximumHeight(90)
        self.outcome_help.setPlainText(
            "Outcomes are added when historical prices are supplied to the history store. "
            "They measure the ticker's forward return after each stored recommendation."
        )
        outcome_layout.addWidget(self.outcome_help)
        tabs.addTab(outcome_widget, "Later outcomes")
        outer.addWidget(tabs)

        refresh.clicked.connect(self.refresh)
        self.ticker_combo.currentTextChanged.connect(lambda _: self.refresh())
        record.clicked.connect(self.record_action)

    def _load_tickers(self, selected: str | None) -> None:
        """Populate the ticker selector and preserve a requested selection."""
        tickers = self.store.tickers()
        self.ticker_combo.blockSignals(True)
        self.ticker_combo.clear()
        self.ticker_combo.addItems(tickers)
        if selected and selected.upper() in tickers:
            self.ticker_combo.setCurrentText(selected.upper())
        self.ticker_combo.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        """Reload all panels for the selected ticker."""
        ticker = self.ticker_combo.currentText().strip()
        if not ticker:
            self.figure.clear()
            self.canvas.draw_idle()
            self.history_table.setRowCount(0)
            self.action_table.setRowCount(0)
            self.outcome_table.setRowCount(0)
            self.summary.setText("No stored recommendation history.")
            return

        history = self.store.ticker_history(ticker)
        transitions = self.store.transitions(ticker)
        transition_by_time = {item.changed_at: f"{item.previous} → {item.current}" for item in transitions}
        self.history_table.setRowCount(len(history))
        for row_index, (_, row) in enumerate(history.iterrows()):
            timestamp = row["generated_at_utc"]
            transition = transition_by_time.get(timestamp.isoformat(), "")
            values = (
                timestamp.strftime("%Y-%m-%d %H:%M"),
                str(row["recommendation"]),
                _format_optional_number(row["probability_outperform"], lambda x: f"{100 * x:.1f}%"),
                f"{float(row['evidence_score']):.3f}",
                f"{float(row['held_quantity']):g}",
                f"${float(row['current_price']):,.2f}",
                transition,
            )
            for column, value in enumerate(values):
                self.history_table.setItem(row_index, column, QTableWidgetItem(value))
        self.history_table.resizeColumnsToContents()

        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if not history.empty:
            x = history["generated_at_utc"]
            probability = pd.to_numeric(history["probability_outperform"], errors="coerce")
            evidence = pd.to_numeric(history["evidence_score"], errors="coerce")
            if probability.notna().any():
                axis.plot(x, probability, marker="o", label="Outperformance probability")
                lower = pd.to_numeric(history["probability_lower"], errors="coerce")
                upper = pd.to_numeric(history["probability_upper"], errors="coerce")
                valid_interval = lower.notna() & upper.notna()
                if valid_interval.any():
                    axis.fill_between(
                        x[valid_interval], lower[valid_interval], upper[valid_interval], alpha=0.2,
                        label="Credible interval",
                    )
                axis.axhline(0.5, linewidth=1, linestyle=":")
                axis.set_ylabel("Probability")
                axis.set_ylim(0.0, 1.0)
            else:
                axis.text(0.5, 0.92, "No calibrated probabilities stored", ha="center", transform=axis.transAxes)
                axis.set_yticks([])

            evidence_axis = axis.twinx()
            evidence_axis.plot(x, evidence, linestyle="--", marker="o", label="Evidence score")
            evidence_axis.set_ylabel("Evidence score")
            evidence_axis.set_ylim(-1.05, 1.05)
            axis.set_title(f"{ticker} recommendation history")
            axis.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        actions = self.store.actions(ticker)
        self.action_table.setRowCount(len(actions))
        for row_index, (_, row) in enumerate(actions.iterrows()):
            values = (
                row["action_at_utc"].strftime("%Y-%m-%d %H:%M"),
                str(row["action"]),
                "" if pd.isna(row["quantity"]) else f"{float(row['quantity']):g}",
                "" if pd.isna(row["price"]) else f"${float(row['price']):,.2f}",
                "" if pd.isna(row["note"]) else str(row["note"]),
                str(int(row["action_id"])),
            )
            for column, value in enumerate(values):
                self.action_table.setItem(row_index, column, QTableWidgetItem(value))
        self.action_table.resizeColumnsToContents()

        outcomes = self.store.outcome_history(ticker)
        self.outcome_table.setRowCount(len(outcomes))
        for row_index, (_, row) in enumerate(outcomes.iterrows()):
            values = (
                str(row["generated_at_utc"])[:19],
                str(row["recommendation"]),
                f"{int(row['horizon'])} days",
                str(row["outcome_date"])[:10],
                f"{100 * float(row['forward_return']):+.2f}%",
            )
            for column, value in enumerate(values):
                self.outcome_table.setItem(row_index, column, QTableWidgetItem(value))
        self.outcome_table.resizeColumnsToContents()

        current = history.iloc[-1] if not history.empty else None
        if current is None:
            self.summary.setText(f"No stored reports for {ticker}.")
        else:
            self.summary.setText(
                f"{ticker}: {current['recommendation']} in the latest snapshot. "
                f"{len(history)} report snapshot(s), {len(transitions)} recommendation change(s), "
                f"and {len(actions)} recorded user action(s)."
            )

    def record_action(self) -> None:
        """Store the action form as a new user decision."""
        ticker = self.ticker_combo.currentText().strip()
        if not ticker:
            return
        quantity = self.quantity_spin.value()
        price = self.price_spin.value()
        try:
            self.store.record_action(
                ticker,
                datetime.now(timezone.utc).isoformat(),
                self.action_combo.currentText(),
                quantity=quantity if quantity > 0 else None,
                price=price if price > 0 else None,
                note=self.note_edit.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not record action", str(exc))
            return
        self.note_edit.clear()
        self.refresh()
