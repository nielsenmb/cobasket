"""Persistent recommendation, action, and outcome history for Cobasket."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cobasket.workflow import PortfolioReport


@dataclass(frozen=True)
class RecommendationTransition:
    """One change in the recommendation assigned to a ticker.

    Parameters
    ----------
    ticker
        Asset symbol.
    changed_at
        UTC timestamp of the newer recommendation.
    previous
        Recommendation in the preceding stored report.
    current
        Recommendation in the newer report.
    """

    ticker: str
    changed_at: str
    previous: str
    current: str


class RecommendationHistoryStore:
    """Store report snapshots, user actions, and later outcomes in SQLite.

    Parameters
    ----------
    path
        SQLite database path. Parent directories are created automatically.
    """

    def __init__(self, path: str | Path = "cobasket_history.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with named-column row access."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        """Create the database schema when it does not already exist."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_id INTEGER PRIMARY KEY,
                    generated_at_utc TEXT NOT NULL UNIQUE,
                    latest_price_date TEXT NOT NULL,
                    cash REAL NOT NULL,
                    invested_value REAL NOT NULL,
                    total_value REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id INTEGER PRIMARY KEY,
                    report_id INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
                    ticker TEXT NOT NULL,
                    held_quantity REAL NOT NULL,
                    current_price REAL NOT NULL,
                    market_value REAL NOT NULL,
                    evidence_score REAL NOT NULL,
                    evidence_confidence REAL NOT NULL,
                    probability_outperform REAL,
                    probability_lower REAL,
                    probability_upper REAL,
                    sample_count INTEGER,
                    recommendation TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    UNIQUE(report_id, ticker)
                );

                CREATE TABLE IF NOT EXISTS user_actions (
                    action_id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    action_at_utc TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL,
                    price REAL,
                    note TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    recommendation_id INTEGER NOT NULL
                        REFERENCES recommendations(recommendation_id) ON DELETE CASCADE,
                    horizon INTEGER NOT NULL,
                    outcome_date TEXT NOT NULL,
                    forward_return REAL NOT NULL,
                    PRIMARY KEY(recommendation_id, horizon)
                );

                CREATE INDEX IF NOT EXISTS idx_recommendations_ticker
                    ON recommendations(ticker);
                CREATE INDEX IF NOT EXISTS idx_actions_ticker
                    ON user_actions(ticker);
                """
            )

    def record_report(self, report: PortfolioReport) -> int:
        """Insert one portfolio report and all of its ticker recommendations.

        Re-recording a report with the same generation timestamp is idempotent.

        Parameters
        ----------
        report
            Completed live portfolio report.

        Returns
        -------
        int
            Database identifier of the stored report.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO reports (
                    generated_at_utc, latest_price_date, cash, invested_value, total_value
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.generated_at_utc,
                    report.latest_price_date,
                    report.cash,
                    report.invested_value,
                    report.total_value,
                ),
            )
            row = connection.execute(
                "SELECT report_id FROM reports WHERE generated_at_utc = ?",
                (report.generated_at_utc,),
            ).fetchone()
            assert row is not None
            report_id = int(row["report_id"])
            for item in report.tickers:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO recommendations (
                        report_id, ticker, held_quantity, current_price, market_value,
                        evidence_score, evidence_confidence, probability_outperform,
                        probability_lower, probability_upper, sample_count,
                        recommendation, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report_id,
                        item.ticker,
                        item.held_quantity,
                        item.current_price,
                        item.market_value,
                        item.evidence_score,
                        item.evidence_confidence,
                        item.probability_outperform,
                        item.probability_lower,
                        item.probability_upper,
                        item.calibration_sample_count,
                        item.recommendation,
                        item.explanation,
                    ),
                )
            return report_id

    def ticker_history(self, ticker: str) -> pd.DataFrame:
        """Return chronological recommendation history for one ticker."""
        symbol = str(ticker).strip().upper()
        query = """
            SELECT r.generated_at_utc, r.latest_price_date,
                   q.recommendation_id, q.ticker, q.held_quantity,
                   q.current_price, q.market_value, q.evidence_score,
                   q.evidence_confidence, q.probability_outperform,
                   q.probability_lower, q.probability_upper, q.sample_count,
                   q.recommendation, q.explanation
            FROM recommendations AS q
            JOIN reports AS r ON r.report_id = q.report_id
            WHERE q.ticker = ?
            ORDER BY r.generated_at_utc
        """
        with self._connect() as connection:
            table = pd.read_sql_query(query, connection, params=(symbol,))
        if not table.empty:
            table["generated_at_utc"] = pd.to_datetime(
                table["generated_at_utc"], utc=True
            )
            table["latest_price_date"] = pd.to_datetime(table["latest_price_date"])
        return table

    def transitions(self, ticker: str) -> tuple[RecommendationTransition, ...]:
        """Return recommendation changes for one ticker in chronological order."""
        history = self.ticker_history(ticker)
        if history.empty:
            return ()
        transitions: list[RecommendationTransition] = []
        previous = str(history.iloc[0]["recommendation"])
        for _, row in history.iloc[1:].iterrows():
            current = str(row["recommendation"])
            if current != previous:
                transitions.append(
                    RecommendationTransition(
                        ticker=str(row["ticker"]),
                        changed_at=pd.Timestamp(row["generated_at_utc"]).isoformat(),
                        previous=previous,
                        current=current,
                    )
                )
            previous = current
        return tuple(transitions)

    def record_action(
        self,
        ticker: str,
        action_at_utc: str,
        action: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        note: str = "",
    ) -> int:
        """Record a user decision or executed portfolio action.

        Parameters
        ----------
        ticker
            Asset symbol.
        action_at_utc
            ISO-formatted action timestamp.
        action
            User-facing action label, such as ``buy``, ``sell``, or ``no action``.
        quantity, price
            Optional executed quantity and price.
        note
            Optional free-text rationale.

        Returns
        -------
        int
            Identifier of the inserted action.
        """
        symbol = str(ticker).strip().upper()
        label = str(action).strip()
        if not symbol or not label:
            raise ValueError("ticker and action must not be empty")
        if quantity is not None and quantity < 0:
            raise ValueError("quantity must be non-negative")
        if price is not None and price <= 0:
            raise ValueError("price must be positive")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO user_actions (
                    ticker, action_at_utc, action, quantity, price, note
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (symbol, action_at_utc, label, quantity, price, str(note)),
            )
            return int(cursor.lastrowid)

    def actions(self, ticker: str) -> pd.DataFrame:
        """Return chronological user actions for one ticker."""
        symbol = str(ticker).strip().upper()
        with self._connect() as connection:
            table = pd.read_sql_query(
                """
                SELECT action_id, ticker, action_at_utc, action, quantity, price, note
                FROM user_actions WHERE ticker = ? ORDER BY action_at_utc
                """,
                connection,
                params=(symbol,),
            )
        if not table.empty:
            table["action_at_utc"] = pd.to_datetime(table["action_at_utc"], utc=True)
        return table

    def update_outcomes(
        self,
        prices: pd.DataFrame,
        *,
        horizons: Sequence[int] = (5, 20, 60),
    ) -> int:
        """Attach forward returns to stored recommendations when data are available.

        Parameters
        ----------
        prices
            Positive historical prices indexed by date.
        horizons
            Forward trading-observation horizons to store.

        Returns
        -------
        int
            Number of newly inserted outcome rows.
        """
        clean = prices.astype(float).sort_index()
        if clean.empty or (clean <= 0).any().any():
            raise ValueError("prices must contain positive observations")
        horizon_values = tuple(int(value) for value in horizons)
        if not horizon_values or any(value < 1 for value in horizon_values):
            raise ValueError("horizons must contain positive integers")

        inserted = 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT q.recommendation_id, q.ticker, r.latest_price_date
                FROM recommendations AS q
                JOIN reports AS r ON r.report_id = q.report_id
                """
            ).fetchall()
            for row in rows:
                ticker = str(row["ticker"])
                if ticker not in clean.columns:
                    continue
                dates = clean.index
                base_date = pd.Timestamp(row["latest_price_date"])
                position = int(dates.searchsorted(base_date, side="left"))
                if position >= len(dates):
                    continue
                start_price = float(clean.iloc[position][ticker])
                for horizon in horizon_values:
                    target = position + horizon
                    if target >= len(dates):
                        continue
                    end_price = float(clean.iloc[target][ticker])
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO outcomes (
                            recommendation_id, horizon, outcome_date, forward_return
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(row["recommendation_id"]),
                            horizon,
                            pd.Timestamp(dates[target]).isoformat(),
                            end_price / start_price - 1.0,
                        ),
                    )
                    inserted += int(cursor.rowcount > 0)
        return inserted

    def outcome_history(self, ticker: str) -> pd.DataFrame:
        """Return stored recommendation outcomes for one ticker."""
        symbol = str(ticker).strip().upper()
        query = """
            SELECT r.generated_at_utc, q.ticker, q.recommendation,
                   o.horizon, o.outcome_date, o.forward_return
            FROM outcomes AS o
            JOIN recommendations AS q
                ON q.recommendation_id = o.recommendation_id
            JOIN reports AS r ON r.report_id = q.report_id
            WHERE q.ticker = ?
            ORDER BY r.generated_at_utc, o.horizon
        """
        with self._connect() as connection:
            return pd.read_sql_query(query, connection, params=(symbol,))

    def tickers(self) -> tuple[str, ...]:
        """Return all ticker symbols represented in the history database."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT ticker FROM recommendations ORDER BY ticker"
            ).fetchall()
        return tuple(str(row["ticker"]) for row in rows)
