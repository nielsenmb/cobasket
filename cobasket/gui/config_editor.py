"""Editable portfolio and watchlist configuration for the Cobasket GUI."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cobasket.evidence import BasketWatchlist
from cobasket.workflow import PortfolioConfig


def parse_basket_text(text: str) -> tuple[str, ...]:
    """Parse a comma-separated basket specification.

    Parameters
    ----------
    text
        Comma-separated ticker symbols.

    Returns
    -------
    tuple of str
        Unique upper-case ticker symbols in their original order.

    Raises
    ------
    ValueError
        If fewer than two unique ticker symbols are supplied.
    """
    tickers = tuple(dict.fromkeys(part.strip().upper() for part in text.split(",") if part.strip()))
    if len(tickers) < 2:
        raise ValueError("each basket must contain at least two unique tickers")
    return tickers


def holdings_from_rows(rows: list[tuple[str, float]]) -> dict[str, float]:
    """Normalize editable holding rows into a holdings mapping.

    Parameters
    ----------
    rows
        Pairs containing ticker symbols and non-negative quantities.

    Returns
    -------
    dict
        Upper-case ticker symbols mapped to quantities.

    Raises
    ------
    ValueError
        If a ticker is empty, duplicated, or has a negative quantity.
    """
    holdings: dict[str, float] = {}
    for ticker, quantity in rows:
        symbol = str(ticker).strip().upper()
        amount = float(quantity)
        if not symbol:
            raise ValueError("holding ticker symbols must not be empty")
        if symbol in holdings:
            raise ValueError(f"duplicate holding ticker: {symbol}")
        if amount < 0.0:
            raise ValueError("holding quantities must be non-negative")
        holdings[symbol] = amount
    return holdings


class ConfigEditorDialog(QDialog):
    """Edit a portfolio configuration and persistent basket watchlist."""

    def __init__(self, config_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path)
        self._watchlist_metadata: dict[str, object] = {}
        self.setWindowTitle("Edit Cobasket portfolio")
        self.resize(760, 660)
        self._build_ui()
        self.load_files()

    def _build_ui(self) -> None:
        """Construct editor widgets and connect their actions."""
        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_portfolio_tab(), "Portfolio")
        tabs.addTab(self._build_watchlist_tab(), "Watchlist")
        outer.addWidget(tabs, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        reload_button = QPushButton("Reload")
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        buttons.addWidget(reload_button)
        buttons.addWidget(save_button)
        buttons.addWidget(close_button)
        outer.addLayout(buttons)
        reload_button.clicked.connect(self.load_files)
        save_button.clicked.connect(self.save_files)
        close_button.clicked.connect(self.reject)

    def _build_portfolio_tab(self) -> QWidget:
        """Create controls for holdings, cash, currency, and analysis settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setRange(0.0, 1e12)
        self.cash_spin.setDecimals(2)
        self.base_currency_edit = QLineEdit()
        self.base_currency_edit.setPlaceholderText("native market currency")
        self.base_currency_edit.setMaxLength(3)
        self.period_edit = QLineEdit("3y")
        self.z_window_spin = QSpinBox()
        self.z_window_spin.setRange(2, 5000)
        self.trace_spin = QDoubleSpinBox()
        self.trace_spin.setRange(0.01, 100.0)
        self.trace_spin.setDecimals(3)
        self.stale_spin = QDoubleSpinBox()
        self.stale_spin.setRange(0.0, 3650.0)
        self.stale_spin.setDecimals(1)
        self.calibration_edit = QLineEdit()
        self.validation_edit = QLineEdit()
        self.basket_calibration_edit = QLineEdit()

        form.addRow("Base currency", self.base_currency_edit)
        form.addRow("Cash (base currency)", self.cash_spin)
        form.addRow("Price history period", self.period_edit)
        form.addRow("Z-score window", self.z_window_spin)
        form.addRow("Minimum trace ratio", self.trace_spin)
        form.addRow("Stale-price limit (days)", self.stale_spin)
        form.addRow("Pooled calibration JSON", self.calibration_edit)
        form.addRow("Basket validation JSON", self.validation_edit)
        form.addRow("Basket-specific calibration JSON", self.basket_calibration_edit)
        layout.addLayout(form)

        layout.addWidget(QLabel("Holdings (quantity zero keeps a ticker eligible for re-entry)"))
        self.holdings_table = QTableWidget(0, 2)
        self.holdings_table.setHorizontalHeaderLabels(("Ticker", "Quantity"))
        self.holdings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.holdings_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.holdings_table, 1)

        actions = QHBoxLayout()
        add_button = QPushButton("Add holding")
        remove_button = QPushButton("Remove row")
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        add_button.clicked.connect(self._add_holding_row)
        remove_button.clicked.connect(lambda: self._remove_selected_rows(self.holdings_table))
        return widget

    def _build_watchlist_tab(self) -> QWidget:
        """Create controls for persistent monitored baskets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.watchlist_name_edit = QLineEdit("Cobasket watchlist")
        self.watchlist_path_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        path_row = QHBoxLayout()
        path_row.addWidget(self.watchlist_path_edit, 1)
        path_row.addWidget(browse_button)
        form.addRow("Watchlist name", self.watchlist_name_edit)
        form.addRow("Watchlist JSON", path_row)
        layout.addLayout(form)

        layout.addWidget(QLabel("One comma-separated basket per row"))
        self.baskets_table = QTableWidget(0, 1)
        self.baskets_table.setHorizontalHeaderLabels(("Tickers",))
        self.baskets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.baskets_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.baskets_table, 1)

        actions = QHBoxLayout()
        add_button = QPushButton("Add basket")
        remove_button = QPushButton("Remove basket")
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        add_button.clicked.connect(self._add_basket_row)
        remove_button.clicked.connect(lambda: self._remove_selected_rows(self.baskets_table))
        browse_button.clicked.connect(self._browse_watchlist)
        return widget

    def _add_holding_row(self, ticker: str = "", quantity: float = 0.0) -> None:
        """Append an editable holding row."""
        row = self.holdings_table.rowCount()
        self.holdings_table.insertRow(row)
        self.holdings_table.setItem(row, 0, QTableWidgetItem(ticker))
        self.holdings_table.setItem(row, 1, QTableWidgetItem(f"{quantity:g}"))

    def _add_basket_row(self, basket: tuple[str, ...] | None = None) -> None:
        """Append an editable watchlist-basket row."""
        row = self.baskets_table.rowCount()
        self.baskets_table.insertRow(row)
        self.baskets_table.setItem(row, 0, QTableWidgetItem(", ".join(basket or ())))

    @staticmethod
    def _remove_selected_rows(table: QTableWidget) -> None:
        """Remove selected rows from an editor table."""
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def _browse_watchlist(self) -> None:
        """Select a watchlist JSON path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose watchlist JSON",
            self.watchlist_path_edit.text() or str(self.config_path.parent / "portfolio_watchlist.json"),
            "JSON files (*.json)",
        )
        if path:
            self.watchlist_path_edit.setText(path)

    def load_files(self) -> None:
        """Load the portfolio configuration and associated watchlist."""
        try:
            config = PortfolioConfig.load(self.config_path)
            watchlist_path = Path(config.watchlist_path)
            if not watchlist_path.is_absolute():
                watchlist_path = self.config_path.parent / watchlist_path
            watchlist = BasketWatchlist.load(watchlist_path)
            payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
            self._watchlist_metadata = dict(payload.get("universe_metadata") or {})
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", f"Could not load configuration:\n{exc}")
            return

        self.cash_spin.setValue(config.cash)
        self.base_currency_edit.setText(config.base_currency or "")
        self.period_edit.setText(config.period)
        self.z_window_spin.setValue(config.z_window)
        self.trace_spin.setValue(config.min_trace_ratio)
        self.stale_spin.setValue(config.max_price_age_days)
        self.calibration_edit.setText(config.calibration_path or "")
        self.validation_edit.setText(config.validation_path or "")
        self.basket_calibration_edit.setText(config.basket_calibration_path or "")
        self.watchlist_path_edit.setText(str(watchlist_path))
        self.watchlist_name_edit.setText(watchlist.name)

        self.holdings_table.setRowCount(0)
        for ticker, quantity in sorted(config.holdings.items()):
            self._add_holding_row(ticker, quantity)
        self.baskets_table.setRowCount(0)
        for basket in watchlist.baskets:
            self._add_basket_row(basket)

    def _collect_holdings(self) -> dict[str, float]:
        """Read and validate holding rows from the table."""
        rows: list[tuple[str, float]] = []
        for row in range(self.holdings_table.rowCount()):
            ticker_item = self.holdings_table.item(row, 0)
            quantity_item = self.holdings_table.item(row, 1)
            ticker = ticker_item.text() if ticker_item is not None else ""
            quantity_text = quantity_item.text() if quantity_item is not None else "0"
            rows.append((ticker, float(quantity_text)))
        return holdings_from_rows(rows)

    def _collect_baskets(self) -> tuple[tuple[str, ...], ...]:
        """Read and validate basket rows from the table."""
        baskets = []
        for row in range(self.baskets_table.rowCount()):
            item = self.baskets_table.item(row, 0)
            baskets.append(parse_basket_text(item.text() if item is not None else ""))
        if not baskets:
            raise ValueError("watchlist must contain at least one basket")
        return tuple(baskets)

    def save_files(self) -> None:
        """Validate and save both the watchlist and portfolio configuration."""
        try:
            holdings = self._collect_holdings()
            baskets = self._collect_baskets()
            watchlist_text = self.watchlist_path_edit.text().strip()
            if not watchlist_text:
                raise ValueError("watchlist path must not be empty")
            watchlist_path = Path(watchlist_text)
            watchlist = BasketWatchlist(
                baskets=baskets,
                name=self.watchlist_name_edit.text().strip() or "Cobasket watchlist",
            )
            config = PortfolioConfig(
                holdings=holdings,
                cash=self.cash_spin.value(),
                watchlist_path=str(watchlist_path),
                base_currency=self.base_currency_edit.text().strip() or None,
                calibration_path=self.calibration_edit.text().strip() or None,
                validation_path=self.validation_edit.text().strip() or None,
                basket_calibration_path=self.basket_calibration_edit.text().strip() or None,
                period=self.period_edit.text().strip(),
                z_window=self.z_window_spin.value(),
                min_trace_ratio=self.trace_spin.value(),
                max_price_age_days=self.stale_spin.value(),
            )
            watchlist.save(watchlist_path)
            if self._watchlist_metadata:
                payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
                payload["universe_metadata"] = self._watchlist_metadata
                watchlist_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            config.save(self.config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        QMessageBox.information(self, "Saved", "Portfolio and watchlist files were saved.")
        self.accept()
