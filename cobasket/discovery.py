"""Persistence-aware discovery and ranking of candidate stock baskets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from cobasket.backtest import rank_confirmed_baskets
from cobasket.cointegration import screen_universe
from cobasket.evidence.calibration import walk_forward_evidence
from cobasket.thresholds import MIN_ACCEPTED_EVALUATIONS


@dataclass(frozen=True)
class DiscoveryResult:
    """Ranked persistence-aware basket-discovery result.

    Parameters
    ----------
    table
        One row per successfully backtested candidate, ranked by discovery status
        and score.
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
        stability = float(np.mean([abs(float(np.dot(reference, vector))) for vector in vectors[1:]]))
    else:
        stability = 0.0
    return {
        "accepted_evaluations": accepted,
        "possible_evaluations": possible,
        "persistence": accepted / possible if possible else 0.0,
        "weight_stability": stability,
    }


def _discovery_status(
    *,
    persistence: float,
    accepted_evaluations: int,
    weight_stability: float,
    sharpe: float,
    min_persistence: float,
    min_weight_stability: float,
    promising_persistence: float,
    promising_evaluations: int,
    promising_weight_stability: float,
) -> str:
    """Classify a discovered basket as promising, borderline, or reject.

    Parameters
    ----------
    persistence
        Fraction of historical evaluation dates passing the Johansen threshold.
    accepted_evaluations
        Number of accepted non-overlapping historical evaluation dates.
    weight_stability
        Sign-invariant mean similarity of historical Johansen vectors.
    sharpe
        Preliminary out-of-fit Sharpe ratio.
    min_persistence, min_weight_stability
        Loose thresholds required to avoid outright rejection.
    promising_persistence, promising_evaluations, promising_weight_stability
        Stricter thresholds required for watchlist export.

    Returns
    -------
    str
        ``promising``, ``borderline``, or ``reject``.
    """
    if persistence < min_persistence or weight_stability < min_weight_stability:
        return "reject"
    if (
        persistence >= promising_persistence
        and accepted_evaluations >= promising_evaluations
        and weight_stability >= promising_weight_stability
        and sharpe > 0.0
    ):
        return "promising"
    return "borderline"


def discover_baskets(
    tickers: Sequence[str],
    *,
    period: str = "5y",
    market_ticker: str = "SPY",
    distance_threshold: float = 0.8,
    min_trace_ratio: float = 1.0,
    max_basket_size: int = 8,
    cost_bps: float = 10.0,
    train_window: int = 252,
    horizon: int = 20,
    step: int = 20,
    min_persistence: float = 0.15,
    min_weight_stability: float = 0.60,
    promising_persistence: float = 0.30,
    promising_evaluations: int = MIN_ACCEPTED_EVALUATIONS,
    promising_weight_stability: float = 0.80,
) -> DiscoveryResult:
    """Run thorough persistence-aware basket discovery.

    Candidate clustering and current Johansen screening are followed by a
    preliminary backtest and independent historical Johansen fits. Results are
    labelled ``promising``, ``borderline``, or ``reject``. Only ``promising``
    baskets should normally be exported to a live watchlist.

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
    max_basket_size
        Maximum dimension of nested hierarchical candidates sent to Johansen.
        The default remains eight, but larger correlation neighborhoods are
        decomposed instead of discarded.
    cost_bps
        Backtest transaction cost in basis points.
    train_window
        Trailing observations per historical persistence fit.
    horizon
        Forward horizon used to space independent evaluations.
    step
        Evaluation spacing. Defaults to ``horizon`` for non-overlapping windows.
    min_persistence, min_weight_stability
        Loose thresholds separating borderline candidates from rejects.
    promising_persistence, promising_evaluations, promising_weight_stability
        Stricter thresholds required for ``promising`` status.

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
        max_basket_size=max_basket_size,
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
        accepted = int(metrics["accepted_evaluations"])
        sharpe = float(result["sharpe"])
        status = _discovery_status(
            persistence=persistence,
            accepted_evaluations=accepted,
            weight_stability=stability,
            sharpe=sharpe,
            min_persistence=min_persistence,
            min_weight_stability=min_weight_stability,
            promising_persistence=promising_persistence,
            promising_evaluations=promising_evaluations,
            promising_weight_stability=promising_weight_stability,
        )
        discovery_score = persistence * stability * max(sharpe, 0.0)
        rows.append(
            {
                "basket": basket,
                "status": status,
                "usable": status == "promising",
                "discovery_score": discovery_score,
                "trace_ratio": trace_ratio,
                "persistence": persistence,
                "weight_stability": stability,
                "accepted_evaluations": accepted,
                "possible_evaluations": int(metrics["possible_evaluations"]),
                "sharpe": sharpe,
                "total_return": float(result["total_return"]),
                "max_drawdown": float(result["max_drawdown"]),
                "n_trades": int(result["n_trades"]),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        rank = {"promising": 0, "borderline": 1, "reject": 2}
        table["_status_rank"] = table["status"].map(rank)
        table = table.sort_values(
            ["_status_rank", "discovery_score", "persistence", "weight_stability"],
            ascending=[True, False, False, False],
            ignore_index=True,
        ).drop(columns="_status_rank")
    return DiscoveryResult(table=table, prices=prices)
