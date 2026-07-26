"""Tests for robustness-aware basket selection and strategy filtering."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from cobasket.evidence import BasketCandidate
from cobasket.robust_strategy import (
    RobustnessGateConfig,
    filter_candidate_baskets_by_robustness,
    historical_stability_table,
)


def _prices() -> pd.DataFrame:
    """Return a compact positive price table."""
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    base = pd.Series(range(100, 180), index=index, dtype=float)
    return pd.DataFrame({"AAA": base, "BBB": base * 1.5, "CCC": base[::-1].to_numpy()})


def test_historical_stability_uses_no_future_prices(monkeypatch):
    """Each robustness fit should see only observations available by its date."""
    seen_lengths: list[int] = []

    def fake_robustness(history, **kwargs):
        seen_lengths.append(len(history))
        return SimpleNamespace(
            break_detected=False,
            stable_fraction=1.0,
            latest_trace_ratio=1.5,
            latest_half_life=10.0,
            latest_weight_drift=0.1,
            warnings=(),
        )

    monkeypatch.setattr("cobasket.robust_strategy.rolling_basket_robustness", fake_robustness)
    prices = _prices()
    dates = prices.index[[30, 50, 70]]
    result = historical_stability_table(
        prices,
        dates,
        gate=RobustnessGateConfig(window=20, step=5),
    )
    assert seen_lengths == [31, 51, 71]
    assert result["stable"].all()


def test_candidate_filter_rejects_latest_break(monkeypatch):
    """Candidate screening should retain only relationships passing the gate."""
    calls = 0

    def fake_robustness(history, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            break_detected=calls == 2,
            stable_fraction=0.9,
        )

    monkeypatch.setattr("cobasket.robust_strategy.rolling_basket_robustness", fake_robustness)
    candidates = (
        BasketCandidate(("AAA", "BBB"), 1.4, 2),
        BasketCandidate(("AAA", "CCC"), 1.3, 2),
    )
    accepted = filter_candidate_baskets_by_robustness(candidates, _prices())
    assert accepted == (candidates[0],)


def test_disabled_gate_preserves_all_candidates():
    """Disabling robustness filtering should return the original candidate order."""
    candidates = (
        BasketCandidate(("AAA", "BBB"), 1.4, 2),
        BasketCandidate(("AAA", "CCC"), 1.3, 2),
    )
    accepted = filter_candidate_baskets_by_robustness(
        candidates,
        _prices(),
        gate=RobustnessGateConfig(enabled=False),
    )
    assert accepted == candidates
