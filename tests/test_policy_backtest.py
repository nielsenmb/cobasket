import numpy as np
import pandas as pd

from cobasket.evidence import LongOnlyPolicy, run_long_only_policy_backtest


def test_policy_reenters_after_complete_sale():
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    prices = pd.DataFrame({"AAA": [10, 10, 11, 10, 9, 9, 10]}, index=dates)
    probabilities = pd.DataFrame({"AAA": [0.8, 0.2, 0.8]}, index=dates[[0, 2, 4]])
    result = run_long_only_policy_backtest(
        prices,
        probabilities,
        policy=LongOnlyPolicy(transaction_cost_bps=0, minimum_trade_value=0),
    )
    assert list(result.trades["side"]) == ["buy", "sell", "buy"]
    assert result.positions.loc[dates[3], "AAA"] == 0
    assert result.positions.loc[dates[-1], "AAA"] > 0


def test_signals_execute_on_next_observation():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAA": [10, 20, 20]}, index=dates)
    probabilities = pd.DataFrame({"AAA": [0.8]}, index=[dates[0]])
    result = run_long_only_policy_backtest(
        prices,
        probabilities,
        policy=LongOnlyPolicy(transaction_cost_bps=0, minimum_trade_value=0),
    )
    assert result.trades.iloc[0]["date"] == dates[1]
    assert result.trades.iloc[0]["price"] == 20


def test_position_weights_respect_maximum():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=dates)
    probabilities = pd.DataFrame({"AAA": [0.9], "BBB": [0.9]}, index=[dates[0]])
    policy = LongOnlyPolicy(maximum_weight=0.3, standard_weight=0.2, transaction_cost_bps=0, minimum_trade_value=0)
    result = run_long_only_policy_backtest(prices, probabilities, policy=policy)
    assert np.all(result.weights.max() <= 0.3000001)
    assert np.all(result.cash >= -1e-8)
