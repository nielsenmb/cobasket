"""Tests for spread construction, signals, and portfolio accounting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from cobasket.backtest import run_backtest, spread_units
from cobasket.cointegration import (
    build_spread,
    johansen_test,
    normalize_cointegration_weights,
)
from cobasket.signals import zscore_signal


def test_weight_normalization_uses_absolute_sum():
    weights = normalize_cointegration_weights([2.0, -1.0, 1.0])
    np.testing.assert_allclose(weights, [0.5, -0.25, 0.25])
    assert np.isclose(np.abs(weights).sum(), 1.0)


def test_tiny_imaginary_weights_are_made_real():
    weights = normalize_cointegration_weights([1.0 + 1e-15j, -1.0 - 1e-15j])
    assert not np.iscomplexobj(weights)
    np.testing.assert_allclose(weights, [0.5, -0.5])


def test_material_imaginary_weights_raise():
    with pytest.raises(ValueError, match="imaginary"):
        normalize_cointegration_weights([1.0 + 0.1j, -1.0])


def test_build_spread_prevents_complex_values():
    index = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [10, 11, 12, 13], "B": [9, 9, 10, 10]}, index=index)
    result = SimpleNamespace(evec=np.array([[1.0 + 1e-15j], [-1.0]]))

    spread, weights = build_spread(prices, result)

    assert spread.dtype == float
    assert not np.iscomplexobj(spread.to_numpy())
    np.testing.assert_allclose(weights, [0.5, -0.5])


def test_zscore_signal_accepts_roundoff_complex_spread():
    index = pd.date_range("2024-01-01", periods=20)
    values = np.linspace(-2, 2, 20).astype(complex) + 1e-15j
    spread = pd.Series(values, index=index)

    z_score, signal = zscore_signal(spread, window=5, entry_z=1.0, exit_z=0.25)

    assert not np.iscomplexobj(z_score.dropna().to_numpy())
    assert set(signal.unique()).issubset({-1, 0, 1})


def test_signal_state_exits_instead_of_forward_filling():
    index = pd.date_range("2024-01-01", periods=9)
    spread = pd.Series([0, 0, 0, 0, 10, 0, 0, 0, 0], index=index, dtype=float)

    _, signal = zscore_signal(spread, window=3, entry_z=1.0, exit_z=0.5)

    assert -1 in signal.values
    assert signal.iloc[-1] == 0


def test_spread_units_have_requested_initial_gross_exposure():
    prices = np.array([100.0, 50.0])
    units = spread_units(prices, [1.0, -1.0], gross_notional=2.0)
    assert np.isclose(np.sum(np.abs(units * prices)), 2.0)


def test_backtest_uses_price_pnl_and_delayed_position():
    index = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [100.0, 100.0, 100.0, 100.0]},
        index=index,
    )
    signal = pd.Series([1, 1, 1, 1], index=index)

    result = run_backtest(prices, [1.0, -1.0], signal, cost_bps=0)

    assert result.position.iloc[0] == 0
    assert result.position.iloc[1] == 1
    assert result.pnl.iloc[1] > 0
    assert np.isclose(result.equity.iloc[-1], 1.015)


def test_synthetic_cointegrated_pair_has_stationary_fitted_spread():
    rng = np.random.default_rng(1234)
    n = 1200
    common_walk = 100 + np.cumsum(rng.normal(0, 0.5, n))
    stationary_noise = np.zeros(n)
    for i in range(1, n):
        stationary_noise[i] = 0.85 * stationary_noise[i - 1] + rng.normal(0, 0.2)
    prices = pd.DataFrame(
        {"A": common_walk + stationary_noise, "B": common_walk},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )

    result = johansen_test(prices, verbose=False)
    spread, _ = build_spread(prices, result)

    assert result.lr1[0] > result.cvt[0][1]
    assert adfuller(spread)[1] < 0.05


def test_independent_random_walks_are_less_cointegrated_than_shared_pair():
    rng = np.random.default_rng(4321)
    n = 1200
    independent = pd.DataFrame(
        {
            "A": 100 + np.cumsum(rng.normal(0, 0.5, n)),
            "B": 100 + np.cumsum(rng.normal(0, 0.5, n)),
        }
    )
    common = 100 + np.cumsum(rng.normal(0, 0.5, n))
    paired = pd.DataFrame(
        {"A": common + rng.normal(0, 0.2, n), "B": common}
    )

    independent_result = johansen_test(independent, verbose=False)
    paired_result = johansen_test(paired, verbose=False)

    independent_ratio = independent_result.lr1[0] / independent_result.cvt[0][1]
    paired_ratio = paired_result.lr1[0] / paired_result.cvt[0][1]
    assert paired_ratio > independent_ratio
