"""Translate evidence scores into configurable long-only recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .base import AssetEvidence


@dataclass(frozen=True)
class Recommendation:
    """Action-oriented interpretation of one asset evidence item.

    Parameters
    ----------
    ticker
        Asset symbol.
    action
        Human-readable long-only action.
    strength
        Ordinal recommendation strength from ``-2`` to ``+2``.
    score
        Underlying signed evidence score.
    confidence
        Strength of the underlying evidence.
    currently_held
        Whether the portfolio currently owns the asset.
    explanation
        Human-readable rationale.
    """

    ticker: str
    action: str
    strength: int
    score: float
    confidence: float
    currently_held: bool
    explanation: str


@dataclass(frozen=True)
class RecommendationPolicy:
    """Threshold policy for converting evidence into long-only actions.

    Parameters
    ----------
    weak_threshold
        Absolute score required for a weak action.
    strong_threshold
        Absolute score required for a strong action.
    min_confidence
        Minimum confidence required for any non-neutral action.

    Notes
    -----
    Thresholds encode a user's decision policy rather than a scientific law.
    They are analogous to selecting a detection threshold after considering the
    costs of false positives and false negatives.
    """

    weak_threshold: float = 0.25
    strong_threshold: float = 0.60
    min_confidence: float = 0.20

    def __post_init__(self) -> None:
        """Validate policy thresholds.

        Raises
        ------
        ValueError
            If thresholds are inconsistent or outside ``[0, 1]``.
        """
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if not 0.0 < self.weak_threshold < self.strong_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < weak_threshold < strong_threshold <= 1"
            )

    def classify(self, evidence: AssetEvidence, *, currently_held: bool) -> Recommendation:
        """Classify one evidence item under a long-only policy.

        Parameters
        ----------
        evidence
            Signed asset evidence.
        currently_held
            Whether the asset is already in the portfolio.

        Returns
        -------
        Recommendation
            Suggested long-only action and its explanation.
        """
        score = evidence.score
        if evidence.confidence < self.min_confidence:
            action, strength = ("Hold" if currently_held else "Watch"), 0
            reason = "evidence confidence is below the action threshold"
        elif score >= self.strong_threshold:
            action, strength = ("Strong add" if currently_held else "Strong buy"), 2
            reason = "strong positive relative-value evidence"
        elif score >= self.weak_threshold:
            action, strength = ("Add" if currently_held else "Buy"), 1
            reason = "positive relative-value evidence"
        elif score <= -self.strong_threshold:
            action, strength = ("Consider reducing" if currently_held else "Avoid buying"), -2
            reason = "strong negative relative-value evidence"
        elif score <= -self.weak_threshold:
            action, strength = ("Hold without adding" if currently_held else "Wait"), -1
            reason = "negative relative-value evidence"
        else:
            action, strength = ("Hold" if currently_held else "Watch"), 0
            reason = "evidence is close to neutral"

        explanation = f"{reason}. {evidence.summary}"
        return Recommendation(
            ticker=evidence.ticker,
            action=action,
            strength=strength,
            score=evidence.score,
            confidence=evidence.confidence,
            currently_held=currently_held,
            explanation=explanation,
        )


def recommend_assets(
    evidence: Iterable[AssetEvidence],
    *,
    holdings: Mapping[str, float] | None = None,
    policy: RecommendationPolicy | None = None,
) -> tuple[Recommendation, ...]:
    """Generate long-only recommendations for a collection of evidence items.

    Parameters
    ----------
    evidence
        Asset evidence items.
    holdings
        Mapping from ticker to quantity currently owned. Positive quantities are
        treated as held positions.
    policy
        Recommendation policy. Defaults to :class:`RecommendationPolicy`.

    Returns
    -------
    tuple of Recommendation
        Recommendations sorted from strongest positive to strongest negative.
    """
    holdings = {str(k).upper(): float(v) for k, v in (holdings or {}).items()}
    policy = policy or RecommendationPolicy()
    recommendations = [
        policy.classify(item, currently_held=holdings.get(item.ticker, 0.0) > 0.0)
        for item in evidence
    ]
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return tuple(recommendations)


def recommendation_table(recommendations: Iterable[Recommendation]) -> pd.DataFrame:
    """Convert recommendations into a display-friendly table.

    Parameters
    ----------
    recommendations
        Recommendation records.

    Returns
    -------
    pandas.DataFrame
        One row per asset in descending score order.
    """
    rows = [
        {
            "ticker": item.ticker,
            "action": item.action,
            "score": item.score,
            "confidence": item.confidence,
            "currently_held": item.currently_held,
            "explanation": item.explanation,
        }
        for item in recommendations
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)
