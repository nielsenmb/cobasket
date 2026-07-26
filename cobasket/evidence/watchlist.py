"""Persistent basket watchlists and repeatable long-only evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.cointegration import cluster_candidates, johansen_test, remove_market_factor

from .base import AssetEvidence
from .calibration import (
    CalibratedAssetEvidence,
    CalibratedRecommendation,
    ProbabilityCalibration,
    recommend_calibrated_assets,
)
from .cointegration import cointegration_evidence
from .recommendations import Recommendation, RecommendationPolicy, recommend_assets


@dataclass(frozen=True)
class BasketCandidate:
    """Candidate cointegrated basket discovered during universe screening.

    Parameters
    ----------
    tickers
        Asset symbols in the basket.
    trace_ratio
        Johansen rank-zero trace statistic divided by its 95 percent critical
        value.
    cluster_size
        Number of assets in the original correlation cluster.
    """

    tickers: tuple[str, ...]
    trace_ratio: float
    cluster_size: int


@dataclass(frozen=True)
class BasketWatchlist:
    """Collection of baskets evaluated independently of current holdings.

    Parameters
    ----------
    baskets
        Ticker groups to retain and re-evaluate. Assets remain present after
        their held quantity falls to zero, allowing later re-entry signals.
    name
        Human-readable watchlist name.
    """

    baskets: tuple[tuple[str, ...], ...]
    name: str = "Cobasket watchlist"

    def __post_init__(self) -> None:
        """Normalize ticker symbols and reject invalid baskets.

        Raises
        ------
        ValueError
            If no baskets are supplied or a basket has fewer than two assets.
        """
        normalized: list[tuple[str, ...]] = []
        for basket in self.baskets:
            clean = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in basket))
            if len(clean) < 2:
                raise ValueError("each watchlist basket must contain at least two assets")
            normalized.append(clean)
        if not normalized:
            raise ValueError("watchlist must contain at least one basket")
        object.__setattr__(self, "baskets", tuple(normalized))

    @property
    def tickers(self) -> tuple[str, ...]:
        """Return all unique watchlist tickers in stable order.

        Returns
        -------
        tuple of str
            Unique asset symbols.
        """
        return tuple(dict.fromkeys(ticker for basket in self.baskets for ticker in basket))

    def save(self, path: str | Path) -> Path:
        """Save the watchlist as human-readable JSON.

        Parameters
        ----------
        path
            Output JSON path.

        Returns
        -------
        pathlib.Path
            Written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "baskets": [list(item) for item in self.baskets]}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "BasketWatchlist":
        """Load a watchlist from JSON.

        Parameters
        ----------
        path
            Existing JSON path.

        Returns
        -------
        BasketWatchlist
            Loaded watchlist.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            baskets=tuple(tuple(item) for item in payload["baskets"]),
            name=str(payload.get("name", "Cobasket watchlist")),
        )


@dataclass(frozen=True)
class WatchlistEvaluation:
    """Current evidence and recommendations for a persistent watchlist.

    Parameters
    ----------
    evidence
        Aggregated evidence with one record per watchlist asset.
    recommendations
        Portfolio-aware long-only recommendations.
    basket_diagnostics
        One row per successfully evaluated basket.
    failed_baskets
        Baskets that could not be evaluated with the current data.
    calibrated_evidence
        Optional probability-calibrated records.
    calibrated_recommendations
        Optional portfolio-aware actions based on calibrated probabilities.
    """

    evidence: tuple[AssetEvidence, ...]
    recommendations: tuple[Recommendation, ...]
    basket_diagnostics: pd.DataFrame
    failed_baskets: tuple[tuple[str, ...], ...]
    calibrated_evidence: tuple[CalibratedAssetEvidence, ...] | None = None
    calibrated_recommendations: tuple[CalibratedRecommendation, ...] | None = None


def select_candidate_baskets(
    prices: pd.DataFrame,
    *,
    market_ticker: str | None = None,
    distance_threshold: float = 0.8,
    min_trace_ratio: float = 1.0,
    max_basket_size: int = 8,
) -> tuple[BasketCandidate, ...]:
    """Select candidate baskets from an initial stock universe.

    The function first clusters assets by residual-return correlation, then
    retains clusters that pass a Johansen cointegration threshold. This is a
    candidate generator, not proof that a basket will remain stable or
    profitable.

    Parameters
    ----------
    prices
        Aligned adjusted prices for the candidate universe. Include
        ``market_ticker`` when common-market removal is requested.
    market_ticker
        Optional market proxy removed before clustering, for example ``SPY``.
        The proxy is not included in candidate baskets.
    distance_threshold
        Hierarchical-clustering cut in correlation-distance units.
    min_trace_ratio
        Minimum Johansen trace-statistic ratio at the 95 percent level.
    max_basket_size
        Maximum number of assets retained from one cluster.

    Returns
    -------
    tuple of BasketCandidate
        Candidates sorted by decreasing cointegration trace ratio.

    Raises
    ------
    ValueError
        If insufficient price series or invalid settings are supplied.
    """
    if max_basket_size < 2:
        raise ValueError("max_basket_size must be at least two")
    clean = prices.astype(float).dropna(axis=0, how="any").sort_index()
    if clean.shape[1] < 2:
        raise ValueError("candidate selection requires at least two assets")

    returns = clean.pct_change().dropna()
    if market_ticker is not None:
        market_ticker = market_ticker.upper()
        residuals = remove_market_factor(returns, market_col=market_ticker)
    else:
        residuals = returns.sub(returns.mean(axis=1), axis=0)

    clusters, _, _ = cluster_candidates(
        residuals,
        distance_threshold=distance_threshold,
    )
    candidates: list[BasketCandidate] = []
    for cluster in clusters:
        selected = tuple(cluster[:max_basket_size])
        if len(selected) < 2:
            continue
        try:
            result = johansen_test(clean.loc[:, selected], verbose=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        critical = float(result.cvt[0, 1])
        trace_ratio = float(result.lr1[0] / critical)
        if trace_ratio >= min_trace_ratio:
            candidates.append(
                BasketCandidate(
                    tickers=selected,
                    trace_ratio=trace_ratio,
                    cluster_size=len(cluster),
                )
            )
    candidates.sort(key=lambda item: item.trace_ratio, reverse=True)
    return tuple(candidates)


def watchlist_from_candidates(
    candidates: Sequence[BasketCandidate],
    *,
    name: str = "Cobasket watchlist",
    top_n: int | None = None,
) -> BasketWatchlist:
    """Create a persistent watchlist from screened basket candidates.

    Parameters
    ----------
    candidates
        Candidate baskets from :func:`select_candidate_baskets`.
    name
        Human-readable watchlist name.
    top_n
        Optional number of highest-ranked baskets to retain.

    Returns
    -------
    BasketWatchlist
        Persistent basket collection.
    """
    chosen = tuple(candidates if top_n is None else candidates[:top_n])
    return BasketWatchlist(
        baskets=tuple(item.tickers for item in chosen),
        name=name,
    )


def _combine_evidence(items: Sequence[AssetEvidence]) -> AssetEvidence:
    """Combine repeated ticker evidence using confidence-weighted averaging.

    Parameters
    ----------
    items
        Evidence records for the same ticker from multiple baskets.

    Returns
    -------
    AssetEvidence
        Aggregated evidence record.
    """
    if not items:
        raise ValueError("at least one evidence item is required")
    weights = np.asarray([max(item.confidence, np.finfo(float).eps) for item in items])
    scores = np.asarray([item.score for item in items])
    score = float(np.average(scores, weights=weights))
    confidence = float(1.0 - np.prod(1.0 - np.clip(weights, 0.0, 1.0)))
    summaries = " ".join(item.summary for item in items)
    return AssetEvidence(
        ticker=items[0].ticker,
        score=float(np.clip(score, -1.0, 1.0)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        source="cointegration_watchlist",
        summary=f"Combined evidence from {len(items)} basket(s). {summaries}",
    )


def evaluate_watchlist(
    prices: pd.DataFrame,
    watchlist: BasketWatchlist,
    *,
    holdings: Mapping[str, float] | None = None,
    window: int = 60,
    min_trace_ratio: float = 1.0,
    policy: RecommendationPolicy | None = None,
    calibration: ProbabilityCalibration | None = None,
) -> WatchlistEvaluation:
    """Evaluate every basket and ticker in a persistent watchlist.

    Current holdings only affect recommendation wording. They do not determine
    which assets are evaluated, so a fully sold position remains eligible for a
    later buy recommendation.

    Parameters
    ----------
    prices
        Current aligned adjusted prices containing all watchlist tickers.
    watchlist
        Persistent basket watchlist.
    holdings
        Mapping from ticker to quantity currently owned.
    window
        Rolling z-score window.
    min_trace_ratio
        Minimum current Johansen trace ratio.
    policy
        Long-only recommendation thresholds.
    calibration
        Optional walk-forward probability calibration.

    Returns
    -------
    WatchlistEvaluation
        Aggregated evidence, recommendations, and basket diagnostics.
    """
    missing = sorted(set(watchlist.tickers).difference(prices.columns))
    if missing:
        raise ValueError(f"prices are missing watchlist tickers: {missing}")

    by_ticker: dict[str, list[AssetEvidence]] = {ticker: [] for ticker in watchlist.tickers}
    diagnostics: list[dict[str, object]] = []
    failed: list[tuple[str, ...]] = []

    for basket in watchlist.baskets:
        try:
            result = cointegration_evidence(
                prices.loc[:, list(basket)],
                window=window,
                min_trace_ratio=min_trace_ratio,
            )
        except (ValueError, np.linalg.LinAlgError):
            failed.append(basket)
            continue
        diagnostics.append(
            {
                "basket": ", ".join(basket),
                "trace_ratio": result.trace_ratio,
                "latest_z_score": result.latest_z_score,
            }
        )
        for item in result.asset_evidence:
            by_ticker[item.ticker].append(item)

    combined = tuple(
        _combine_evidence(items)
        for ticker in watchlist.tickers
        if (items := by_ticker[ticker])
    )
    recommendations = recommend_assets(
        combined,
        holdings=holdings,
        policy=policy,
    )
    calibrated = (
        tuple(calibration.calibrate(item) for item in combined)
        if calibration is not None
        else None
    )
    calibrated_recommendations = (
        recommend_calibrated_assets(calibrated, holdings=dict(holdings or {}))
        if calibrated is not None
        else None
    )
    return WatchlistEvaluation(
        evidence=combined,
        recommendations=recommendations,
        basket_diagnostics=pd.DataFrame(diagnostics),
        failed_baskets=tuple(failed),
        calibrated_evidence=calibrated,
        calibrated_recommendations=calibrated_recommendations,
    )


def candidate_table(candidates: Iterable[BasketCandidate]) -> pd.DataFrame:
    """Convert basket candidates into a display-friendly table.

    Parameters
    ----------
    candidates
        Candidate basket records.

    Returns
    -------
    pandas.DataFrame
        Candidate baskets ordered by trace ratio.
    """
    rows = [
        {
            "tickers": ", ".join(item.tickers),
            "trace_ratio": item.trace_ratio,
            "cluster_size": item.cluster_size,
        }
        for item in candidates
    ]
    return pd.DataFrame(rows).sort_values("trace_ratio", ascending=False, ignore_index=True)
