"""Tests for basket-specific probability calibration."""

from __future__ import annotations

import pandas as pd

from cobasket.basket_calibration import (
    BasketCalibrationSet,
    BasketProbabilityCalibration,
    fit_basket_calibrations,
)
from cobasket.basket_validation import BasketValidationProfile, BasketValidationSet
from cobasket.calibration_workflow import WatchlistCalibrationResult
from cobasket.evidence import fit_probability_calibration


def _profile(basket: tuple[str, ...], status: str, evaluations: int) -> BasketValidationProfile:
    """Create a compact validation profile for calibration tests."""
    return BasketValidationProfile(
        basket=basket,
        status=status,
        current_trace_ratio=1.2,
        accepted_evaluations=evaluations,
        possible_evaluations=25,
        acceptance_rate=evaluations / 25,
        weight_stability=0.9,
        score_return_correlation=0.2,
        positive_outperform_rate=0.7,
        negative_outperform_rate=0.3,
        calibration_contrast=0.4,
        records=2 * evaluations,
        reasons=("test",),
    )


def _records() -> pd.DataFrame:
    """Create synthetic historical evidence for two baskets."""
    rows = []
    for basket, ticker_a, ticker_b in [
        ("AAA, BBB", "AAA", "BBB"),
        ("CCC, DDD", "CCC", "DDD"),
    ]:
        for index in range(24):
            score = -0.8 + 1.6 * index / 23
            for ticker, signed_score in [(ticker_a, score), (ticker_b, -score)]:
                rows.append(
                    {
                        "basket": basket,
                        "evaluation_date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(index),
                        "ticker": ticker,
                        "score": signed_score,
                        "outperformed": int(signed_score > 0),
                    }
                )
    return pd.DataFrame(rows)


def test_basket_calibration_round_trip(tmp_path):
    """Basket-specific calibration sets should survive JSON persistence."""
    records = _records().query("basket == 'AAA, BBB'")
    calibration = fit_probability_calibration(records, horizon=20)
    item = BasketProbabilityCalibration(
        basket=("AAA", "BBB"),
        accepted_evaluations=24,
        calibration=calibration,
    )
    original = BasketCalibrationSet(
        generated_at_utc="2026-08-11T12:00:00+00:00",
        min_evaluations=20,
        calibrations=(item,),
        skipped={"CCC, DDD": "validation status is weak"},
    )
    loaded = BasketCalibrationSet.load(original.save(tmp_path / "basket_calibration.json"))
    assert loaded.min_evaluations == 20
    assert loaded.by_key()["AAA, BBB"].accepted_evaluations == 24
    assert loaded.by_key()["AAA, BBB"].calibration.lookup(0.8)["probability_mean"] > 0.5
    assert loaded.skipped["CCC, DDD"] == "validation status is weak"


def test_fit_basket_calibrations_only_uses_validated_baskets(tmp_path, monkeypatch):
    """Weak baskets should not receive probability calibrations."""
    validation = BasketValidationSet(
        generated_at_utc="2026-08-11T12:00:00+00:00",
        train_window=252,
        z_window=60,
        horizon=20,
        step=20,
        min_trace_ratio=1.0,
        profiles=(
            _profile(("AAA", "BBB"), "validated", 24),
            _profile(("CCC", "DDD"), "weak", 24),
        ),
    )
    validation_path = validation.save(tmp_path / "validation.json")
    records = _records()
    pooled = fit_probability_calibration(records, horizon=20)
    fake = WatchlistCalibrationResult(
        calibration=pooled,
        records=records,
        basket_summary=pd.DataFrame(),
    )
    monkeypatch.setattr("cobasket.basket_calibration.calibrate_watchlist", lambda *args, **kwargs: fake)

    result = fit_basket_calibrations(
        tmp_path / "portfolio.json",
        validation_path,
        min_evaluations=20,
    )
    assert set(result.by_key()) == {"AAA, BBB"}
    assert result.skipped["CCC, DDD"] == "validation status is weak"


def test_fit_basket_calibrations_requires_independent_history(tmp_path, monkeypatch):
    """Validated baskets with too little history should still be skipped."""
    validation = BasketValidationSet(
        generated_at_utc="2026-08-11T12:00:00+00:00",
        train_window=252,
        z_window=60,
        horizon=20,
        step=20,
        min_trace_ratio=1.0,
        profiles=(_profile(("AAA", "BBB"), "validated", 24),),
    )
    validation_path = validation.save(tmp_path / "validation.json")
    records = _records().query("basket == 'AAA, BBB'").copy()
    records["evaluation_date"] = pd.Timestamp("2024-01-01")
    pooled = fit_probability_calibration(records, horizon=20)
    fake = WatchlistCalibrationResult(
        calibration=pooled,
        records=records,
        basket_summary=pd.DataFrame(),
    )
    monkeypatch.setattr("cobasket.basket_calibration.calibrate_watchlist", lambda *args, **kwargs: fake)

    result = fit_basket_calibrations(
        tmp_path / "portfolio.json",
        validation_path,
        min_evaluations=20,
    )
    assert not result.calibrations
    assert "requires 20" in result.skipped["AAA, BBB"]
