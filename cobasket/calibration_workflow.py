"""Watchlist-level probability calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cobasket.config_paths import resolve_portfolio_config_paths
from cobasket.data import DataManager
from cobasket.evidence import (
    BasketWatchlist,
    ProbabilityCalibration,
    fit_probability_calibration,
    walk_forward_evidence,
)
from cobasket.workflow import PortfolioConfig


@dataclass(frozen=True)
class WatchlistCalibrationResult:
    """Container for a fitted watchlist-level probability calibration.

    Parameters
    ----------
    calibration
        Fitted mapping from raw evidence score to outperformance probability.
    records
        Pooled walk-forward evidence/outcome records used for the fit.
    basket_summary
        Per-basket counts describing successful and skipped calibration records.
    """

    calibration: ProbabilityCalibration
    records: pd.DataFrame
    basket_summary: pd.DataFrame


def calibrate_watchlist(
    config_path: str | Path,
    *,
    train_window: int = 252,
    z_window: int | None = None,
    horizon: int = 20,
    step: int = 5,
    score_edges: Sequence[float] = (-1.0, -0.60, -0.25, 0.25, 0.60, 1.0),
    min_trace_ratio: float | None = None,
    force_refresh: bool = False,
    data_manager: DataManager | None = None,
) -> WatchlistCalibrationResult:
    """Fit one empirical probability calibration across a monitored watchlist.

    Each basket is evaluated independently through time. At every evaluation
    date the cointegration relation is fitted using only earlier prices, and
    the later outcome is measured over ``horizon`` trading observations. The
    resulting asset-level records are pooled only after those leakage-free
    evaluations have been generated.

    Parameters
    ----------
    config_path
        Portfolio configuration JSON containing the watchlist and data period.
    train_window
        Trailing observations used to fit each historical basket relation.
    z_window
        Trailing spread-standardization window. Defaults to the portfolio value.
    horizon
        Forward outcome horizon in trading observations.
    step
        Number of observations between historical evaluations.
    score_edges
        Evidence-score bin edges used by the beta-binomial calibration.
    min_trace_ratio
        Historical Johansen threshold. Defaults to the portfolio value.
    force_refresh
        Whether to bypass reusable market-data cache files.
    data_manager
        Optional data manager, primarily for tests or alternative data sources.

    Returns
    -------
    WatchlistCalibrationResult
        Fitted calibration, pooled records, and per-basket diagnostics.

    Raises
    ------
    ValueError
        If the watchlist yields no usable walk-forward outcomes.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = resolve_portfolio_config_paths(PortfolioConfig.load(config_path), config_path)
    watchlist = BasketWatchlist.load(config.watchlist_path)
    manager = data_manager or DataManager()
    z_window = config.z_window if z_window is None else int(z_window)
    min_trace_ratio = config.min_trace_ratio if min_trace_ratio is None else float(min_trace_ratio)

    prices = manager.prices(
        watchlist.tickers,
        period=config.period,
        force_refresh=force_refresh,
        min_coverage=1.0,
    )
    record_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for basket in watchlist.baskets:
        basket_prices = prices.loc[:, list(basket)].dropna()
        try:
            records = walk_forward_evidence(
                basket_prices,
                train_window=train_window,
                z_window=z_window,
                horizon=horizon,
                step=step,
                min_trace_ratio=min_trace_ratio,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            summaries.append(
                {"basket": ", ".join(basket), "records": 0, "evaluations": 0, "status": str(exc)}
            )
            continue
        if records.empty:
            summaries.append(
                {"basket": ", ".join(basket), "records": 0, "evaluations": 0, "status": "no accepted historical fits"}
            )
            continue
        records = records.copy()
        records["basket"] = ", ".join(basket)
        record_tables.append(records)
        summaries.append(
            {
                "basket": ", ".join(basket),
                "records": len(records),
                "evaluations": records["evaluation_date"].nunique(),
                "status": "ok",
            }
        )

    if not record_tables:
        raise ValueError("watchlist produced no usable walk-forward calibration outcomes")

    pooled = pd.concat(record_tables, ignore_index=True)
    calibration = fit_probability_calibration(pooled, score_edges=score_edges, horizon=horizon)
    return WatchlistCalibrationResult(
        calibration=calibration,
        records=pooled,
        basket_summary=pd.DataFrame(summaries),
    )
