"""Tests for rolling basket robustness diagnostics."""

import numpy as np
import pandas as pd

from cobasket.robustness import mean_reversion_half_life, rolling_basket_robustness


def _stable_prices(seed: int = 3, n: int = 520) -> pd.DataFrame:
    """Create a synthetic pair with a stationary difference."""
    rng = np.random.default_rng(seed)
    common = 100.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.8 * residual[i - 1] + rng.normal(0.0, 0.3)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"AAA": common + residual, "BBB": common - residual}, index=index)


def test_mean_reversion_half_life_is_finite_for_stationary_ar1():
    """A stationary AR(1) process should have a finite positive half-life."""
    rng = np.random.default_rng(7)
    values = np.zeros(500)
    for i in range(1, len(values)):
        values[i] = 0.85 * values[i - 1] + rng.normal()
    half_life = mean_reversion_half_life(pd.Series(values))
    assert np.isfinite(half_life)
    assert half_life > 0.0


def test_stable_pair_produces_rolling_diagnostics():
    """A controlled cointegrated pair should produce successful rolling fits."""
    result = rolling_basket_robustness(
        _stable_prices(), window=180, step=60, max_weight_drift=1.5
    )
    assert len(result.rolling) >= 5
    assert 0.0 <= result.stable_fraction <= 1.0
    assert "trace_ratio" in result.rolling
    assert "half_life" in result.rolling
    assert "weight_drift" in result.rolling


def test_relationship_break_is_flagged_or_reduces_stability():
    """Breaking the second half of a pair should weaken its robustness summary."""
    prices = _stable_prices()
    rng = np.random.default_rng(11)
    split = len(prices) // 2
    prices.loc[prices.index[split]:, "BBB"] += np.cumsum(rng.normal(0.5, 1.2, len(prices) - split))
    result = rolling_basket_robustness(
        prices,
        window=180,
        step=40,
        max_half_life=80.0,
        max_weight_drift=0.5,
    )
    assert result.break_detected or result.stable_fraction < 0.8 or result.warnings
