"""Tests for walk-forward probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobasket.evidence import (
    AssetEvidence,
    ProbabilityRecommendationPolicy,
    calibrate_evidence,
    fit_probability_calibration,
    recommend_calibrated_assets,
    walk_forward_evidence,
)


def cointegrated_prices(n: int = 520, seed: int = 123) -> pd.DataFrame:
    """Return deterministic synthetic cointegrated prices."""
    rng = np.random.default_rng(seed)
    common = 100.0 + np.cumsum(rng.normal(0.08, 0.7, n))
    noise_a = np.zeros(n)
    noise_b = np.zeros(n)
    for index in range(1, n):
        noise_a[index] = 0.75 * noise_a[index - 1] + rng.normal(0.0, 0.5)
        noise_b[index] = 0.65 * noise_b[index - 1] + rng.normal(0.0, 0.6)
    frame = pd.DataFrame(
        {
            "AAA": common + noise_a + 30.0,
            "BBB": 1.15 * common + noise_b + 20.0,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    return frame


def test_walk_forward_records_are_out_of_sample() -> None:
    prices = cointegrated_prices()
    records = walk_forward_evidence(
        prices,
        train_window=180,
        z_window=40,
        horizon=10,
        step=10,
        min_trace_ratio=0.8,
    )
    assert not records.empty
    assert (records["future_date"] > records["evaluation_date"]).all()
    assert set(records["ticker"]) == {"AAA", "BBB"}
    assert records["outperformed"].isin([0, 1]).all()


def test_beta_bin_calibration_includes_empty_bins() -> None:
    records = pd.DataFrame(
        {
            "score": [-0.9, -0.8, 0.7, 0.9, 0.8],
            "outperformed": [0, 0, 1, 1, 1],
        }
    )
    calibration = fit_probability_calibration(records, horizon=20)
    assert len(calibration.table) == 5
    assert calibration.table["probability_mean"].between(0.0, 1.0).all()
    empty = calibration.table.loc[calibration.table["sample_count"] == 0]
    assert (empty["probability_mean"] == 0.5).all()


def test_calibrated_recommendation_respects_holdings() -> None:
    records = pd.DataFrame(
        {
            "score": [0.8] * 30 + [-0.8] * 30,
            "outperformed": [1] * 27 + [0] * 3 + [1] * 3 + [0] * 27,
        }
    )
    calibration = fit_probability_calibration(records, horizon=20)
    evidence = (
        AssetEvidence("AAA", 0.8, 0.9, "test", "positive"),
        AssetEvidence("BBB", -0.8, 0.9, "test", "negative"),
    )
    calibrated = calibrate_evidence(evidence, calibration)
    recommendations = recommend_calibrated_assets(
        calibrated,
        holdings={"BBB": 2.0},
        policy=ProbabilityRecommendationPolicy(
            min_samples=20,
            require_interval_excludes_half=False,
        ),
    )
    actions = {item.ticker: item.action for item in recommendations}
    assert actions["AAA"] in {"Buy", "Strong buy"}
    assert actions["BBB"] in {"Hold without adding", "Consider reducing"}
