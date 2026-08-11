"""Tests for persistent basket validation profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobasket.basket_validation import (
    BasketValidationProfile,
    BasketValidationSet,
    BasketValidationThresholds,
    _classify_profile,
    _weight_stability,
)


def test_validation_set_round_trip(tmp_path):
    """Validation profiles should survive JSON persistence exactly."""
    profile = BasketValidationProfile(
        basket=("AAA", "BBB"),
        status="validated",
        current_trace_ratio=1.3,
        accepted_evaluations=24,
        possible_evaluations=30,
        acceptance_rate=0.8,
        weight_stability=0.9,
        score_return_correlation=0.2,
        positive_outperform_rate=0.7,
        negative_outperform_rate=0.4,
        calibration_contrast=0.3,
        records=48,
        reasons=("current and historical validation criteria passed",),
    )
    validation = BasketValidationSet(
        generated_at_utc="2026-08-11T12:00:00+00:00",
        train_window=252,
        z_window=60,
        horizon=20,
        step=20,
        min_trace_ratio=1.0,
        profiles=(profile,),
    )
    loaded = BasketValidationSet.load(validation.save(tmp_path / "validation.json"))
    assert loaded == validation
    assert loaded.by_key()["AAA, BBB"].status == "validated"


def test_classification_rejects_currently_broken_relation():
    """A currently failed Johansen relation should be rejected immediately."""
    status, reasons = _classify_profile(
        current_trace_ratio=0.8,
        accepted_evaluations=30,
        acceptance_rate=0.8,
        weight_stability=0.9,
        score_return_correlation=0.2,
        calibration_contrast=0.2,
        thresholds=BasketValidationThresholds(),
    )
    assert status == "rejected"
    assert "current cointegration" in reasons[0]


def test_classification_requires_predictive_evidence_for_validated_status():
    """Stable cointegration alone should not imply a validated trading signal."""
    status, reasons = _classify_profile(
        current_trace_ratio=1.2,
        accepted_evaluations=30,
        acceptance_rate=0.5,
        weight_stability=0.9,
        score_return_correlation=-0.1,
        calibration_contrast=-0.2,
        thresholds=BasketValidationThresholds(),
    )
    assert status == "weak"
    assert len(reasons) == 2


def test_weight_stability_is_invariant_to_overall_sign_flip():
    """Johansen vectors differing only by sign should be perfectly stable."""
    records = pd.DataFrame(
        {
            "evaluation_date": ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.6, -0.4, -0.6, 0.4],
        }
    )
    assert np.isclose(_weight_stability(records, ("AAA", "BBB")), 1.0)
