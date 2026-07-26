"""Tests for cointegration evidence and recommendation policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobasket.evidence import (
    AssetEvidence,
    RecommendationPolicy,
    cointegration_evidence,
    evidence_table,
    recommend_assets,
    recommendation_table,
)


def _cointegrated_prices(seed: int = 42, n: int = 500) -> pd.DataFrame:
    """Return a reproducible pair sharing one stochastic trend."""
    rng = np.random.default_rng(seed)
    common = 100.0 + np.cumsum(rng.normal(0.0, 0.7, n))
    stationary = np.zeros(n)
    for idx in range(1, n):
        stationary[idx] = 0.8 * stationary[idx - 1] + rng.normal(0.0, 0.4)
    return pd.DataFrame(
        {"AAA": common + stationary, "BBB": common - stationary},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_asset_evidence_validates_bounds():
    """Evidence scores and confidence values remain bounded."""
    with pytest.raises(ValueError):
        AssetEvidence("AAA", 1.1, 0.5, "test", "invalid")
    with pytest.raises(ValueError):
        AssetEvidence("AAA", 0.1, -0.1, "test", "invalid")


def test_cointegration_evidence_emits_opposing_asset_scores():
    """Two-leg relative-value evidence has opposite signed scores."""
    prices = _cointegrated_prices()
    prices.iloc[-1, 0] += 8.0
    result = cointegration_evidence(prices, window=60)
    scores = {item.ticker: item.score for item in result.asset_evidence}
    assert np.sign(scores["AAA"]) == -np.sign(scores["BBB"])
    assert result.trace_ratio >= 1.0
    assert np.isfinite(result.latest_z_score)


def test_evidence_table_is_sorted_descending():
    """Evidence tables put the most positive score first."""
    prices = _cointegrated_prices()
    prices.iloc[-1, 0] += 8.0
    table = evidence_table(cointegration_evidence(prices, window=60))
    assert table["score"].is_monotonic_decreasing


def test_policy_changes_wording_for_held_assets():
    """The same negative evidence maps to avoid or reduce by holding state."""
    evidence = AssetEvidence(
        ticker="AAA",
        score=-0.8,
        confidence=0.9,
        source="test",
        summary="AAA is relatively expensive.",
    )
    policy = RecommendationPolicy()
    not_held = policy.classify(evidence, currently_held=False)
    held = policy.classify(evidence, currently_held=True)
    assert not_held.action == "Avoid buying"
    assert held.action == "Consider reducing"


def test_low_confidence_suppresses_action():
    """Large scores do not trigger actions when confidence is inadequate."""
    evidence = AssetEvidence("AAA", 0.9, 0.1, "test", "uncertain")
    rec = RecommendationPolicy(min_confidence=0.2).classify(
        evidence, currently_held=False
    )
    assert rec.action == "Watch"
    assert rec.strength == 0


def test_recommendation_table_and_holdings():
    """Recommendation helpers respect positive portfolio quantities."""
    evidence = (
        AssetEvidence("AAA", 0.7, 0.8, "test", "positive"),
        AssetEvidence("BBB", -0.7, 0.8, "test", "negative"),
    )
    recommendations = recommend_assets(evidence, holdings={"AAA": 2, "BBB": 3})
    table = recommendation_table(recommendations)
    assert table.loc[table.ticker == "AAA", "action"].item() == "Strong add"
    assert table.loc[table.ticker == "BBB", "action"].item() == "Consider reducing"
