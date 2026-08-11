"""Persistence-aware discovery and ranking of candidate stock baskets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from cobasket.backtest import rank_confirmed_baskets
from cobasket.cointegration import screen_universe
from cobasket.evidence.calibration import walk_forward_evidence


@dataclass(frozen=True)
class DiscoveryResult:
    """Ranked persistence-aware basket-discovery result.

    Parameters
    ----------
    table
        One row per successfully backtested candidate, ranked by discovery score.
    prices
        Aligned prices used for discovery.
    """

    table: pd.DataFrame
    prices: pd.DataFrame


def _persistence_metrics(
    prices: pd.DataFrame,
    basket: Sequence[str],
    *,
    train_window: int,
    horizon: int,
    step: int,
    min_trace_ratio: float,
) -> dict[str, float | int]:
    """Measure historical persistence using independent walk-forward fits.

    Parameters
    ----------
    prices
        Aligned adjusted prices for the full universe.
    basket
        Candidate ticker symbols.
    train_window
        Trailing observations used for each historical fit.
    horizon
        Forward outcome horizon in trading observations.
    step
        Spacing between historical evaluation dates.
    min_trace_ratio
        Johansen trace-ratio threshold.

    Returns
    -------
    dict
        Accepted evaluation count, nominal count, acceptance rate, and a
        sign-invariant mean weight-stability score.
    """
    basket_prices = prices.loc[:, list(basket)].dropna()
    stop = len(basket_prices) - horizon
    possible = max(0, len(range(train_window - 1, stop, step))) if stop > train_window - 1 else 0
    try:
        records = walk_forward_evidence(
            basket_prices,
            train_window=train_window,
            z_window=min(60, train_window),
            horizon=horizon,
            step=step,
            min_trace_ratio=min_trace_ratio,
        )
    except (ValueError, np.linalg.LinAlgError):
        records = pd.DataFrame()
    if records.empty:
        return {
            "accepted_evaluations": 0,
            "possible_evaluations": possible,
            "persistence": 0.0,
            "weight_stability": 0.0,
        }

    accepted = int(records["evaluation_date"].nunique())
    vectors: list[np.ndarray] = []
    members = tuple(basket)
    for _, group in records.groupby("evaluation_date"):
        vector = group.set_index("ticker")["weight"].reindex(members).to_numpy(dtype=float)
        norm = np.linalg.norm(vector)
        if np.all(np.isfinite(vector)) and norm > np.finfo(float).eps:
            vectors.append(vector / norm)
    if len(vectors) >= 2:
        reference = vectors[0]
        stability = float(
            np.mean([abs(float(np.dot(reference, vector))) for vector in vectors[1:]])
        )
    else:
        stability = 0.0
    return {
        "accepted_evaluations": accepted,
        "possible_evaluations": possible,
        "persistence": accepted / possible if possible else 0.0,
        "weight_stability": stability,
    }


def discover_baskets(
    tickers: Sequence[str],
    *,
    period: str = "5y",
    market_ticker: str = "SPY",
    distance_threshold: float = 0.8,
    min_trace_ratio: float = 1.0,
    cost_bps: float = 10.0,
    train_window: int = 252,
    horizon: int = 20,
    step: int = 20,
    min_persistence: float = 0.15,
    min_weight_stability: float = 0.60,
) -> DiscoveryResult:
    """Run thorough persistence-aware basket discovery.

    Candidate clustering and current Johansen screening are followed by a
    preliminary backtest and independent historical Johansen fits. Baskets that
    fail the persistence or weight-stability thresholds remain in the returned
    table with ``usable=False`` so rejection is inspectable.

    Parameters
    ----------
    tickers
        Universe ticker symbols.
    period
        Price-history period.
    market_ticker
        Market proxy used by residual-correlation clustering.
    distance_threshold
        Hierarchical clustering cut height.
    min_trace_ratio
        Current and historical Johansen trace-ratio threshold.
    cost_bps
        Backtest transaction cost in basis points.
    train_window
        Trailing observations per historical persistence fit.
    horizon
        Forward horizon used to space independent evaluations.
    step
        Evaluation spacing. Defaults to ``horizon`` for non-overlapping windows.
    min_persistence
        Minimum accepted fraction of nominal historical fits.
    min_weight_stability
        Minimum sign-invariant mean cosine similarity of historical weights.

    Returns
    -------
    DiscoveryResult
        Ranked diagnostics and the aligned price table.
    """
    confirmed, prices, _, _ = screen_universe(
        tickers,
        period=period,
        distance_threshold=distance_threshold,
        min_trace_stat_ratio=min_trace_ratio,
        market_ticker=market_ticker,
    )
    backtests = rank_confirmed_baskets(confirmed, prices, cost_bps=cost_bps)
    rows: list[dict[str, object]] = []
    for result in backtests:
        basket = tuple(result["basket"])
        metrics = _persistence_metrics(
            prices,
            basket,
            train_window=train_window,
            horizon=horizon,
            step=step,
            min_trace_ratio=min_trace_ratio,
        )
        trace_ratio = float(result["johansen_stat"] / result["johansen_crit"])
        persistence = float(metrics["persistence"])
        stability = float(metrics["weight_stability"])
        usable = persistence >= min_persistence and stability >= min_weight_stability
        sharpe = float(result["sharpe"])
        discovery_score = persistence * stability * max(sharpe, 0.0)
        rows.append(
            {
                "basket": basket,
                "usable": usable,
                "discovery_score": discovery_score,
                "trace_ratio": trace_ratio,
                "persistence": persistence,
                "weight_stability": stability,
                "accepted_evaluations": int(metrics["accepted_evaluations"]),
                "possible_evaluations": int(metrics["possible_evaluations"]),
                "sharpe": sharpe,
                "total_return": float(result["total_return"]),
                "max_drawdown": float(result["max_drawdown"]),
                "n_trades": int(result["n_trades"]),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["usable", "discovery_score", "persistence", "weight_stability"],
            ascending=[False, False, False, False],
            ignore_index=True,
        )
    return DiscoveryResult(table=table, prices=prices)
