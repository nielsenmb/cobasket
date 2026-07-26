"""Long-only portfolio simulation for calibrated Cobasket recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongOnlyPolicy:
    """Trading rules for converting probabilities into target portfolio weights.

    Parameters
    ----------
    buy_probability
        Probability required to open or maintain a standard position.
    strong_buy_probability
        Probability required for the maximum position.
    reduce_probability
        Probability below which an existing position is reduced.
    sell_probability
        Probability below which an existing position is fully sold.
    standard_weight
        Target portfolio fraction for a normal buy signal.
    maximum_weight
        Maximum target fraction for a strong buy signal.
    reduced_weight
        Target portfolio fraction after a reduce signal.
    transaction_cost_bps
        One-way transaction cost in basis points of traded notional.
    minimum_trade_value
        Trades below this cash value are ignored.
    """

    buy_probability: float = 0.60
    strong_buy_probability: float = 0.70
    reduce_probability: float = 0.40
    sell_probability: float = 0.30
    standard_weight: float = 0.10
    maximum_weight: float = 0.20
    reduced_weight: float = 0.05
    transaction_cost_bps: float = 10.0
    minimum_trade_value: float = 1.0

    def __post_init__(self) -> None:
        """Validate probability and allocation thresholds."""
        if not 0.5 < self.buy_probability < self.strong_buy_probability <= 1.0:
            raise ValueError("buy thresholds must satisfy 0.5 < buy < strong_buy <= 1")
        if not 0.0 <= self.sell_probability < self.reduce_probability < 0.5:
            raise ValueError("sell thresholds must satisfy 0 <= sell < reduce < 0.5")
        if not 0.0 <= self.reduced_weight <= self.standard_weight <= self.maximum_weight <= 1.0:
            raise ValueError("weights must satisfy 0 <= reduced <= standard <= maximum <= 1")
        if self.transaction_cost_bps < 0 or self.minimum_trade_value < 0:
            raise ValueError("cost and minimum trade settings must be non-negative")

    def target_weight(self, probability: float, current_weight: float) -> float:
        """Return the desired long-only weight for one asset.

        Parameters
        ----------
        probability
            Calibrated probability of relative outperformance.
        current_weight
            Current portfolio fraction allocated to the asset.

        Returns
        -------
        float
            Desired portfolio fraction after applying the policy.
        """
        probability = float(probability)
        if probability >= self.strong_buy_probability:
            return self.maximum_weight
        if probability >= self.buy_probability:
            return self.standard_weight
        if probability <= self.sell_probability:
            return 0.0
        if probability <= self.reduce_probability and current_weight > 0.0:
            return min(current_weight, self.reduced_weight)
        return current_weight


@dataclass(frozen=True)
class PolicyBacktestResult:
    """Outputs from a long-only probability-policy simulation.

    Parameters
    ----------
    equity
        Portfolio equity through time.
    cash
        Cash balance through time.
    positions
        Share quantities held after each rebalance.
    weights
        End-of-day asset weights.
    trades
        Executed trade ledger.
    metrics
        Summary performance statistics.
    """

    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: Mapping[str, float]


def _performance_metrics(equity: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Compute compact performance statistics from an equity curve."""
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((len(equity) - 1) / periods_per_year, 1.0 / periods_per_year)
    annualized_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
    }


def run_long_only_policy_backtest(
    prices: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    policy: LongOnlyPolicy | None = None,
    initial_cash: float = 10_000.0,
    periods_per_year: int = 252,
) -> PolicyBacktestResult:
    """Simulate periodic long-only trading from calibrated probabilities.

    Forecasts are acted upon at the next available price observation, avoiding
    same-bar look-ahead. Assets absent from a forecast row retain their current
    positions. A sold asset remains eligible for re-entry when a later forecast
    crosses the buy threshold.

    Parameters
    ----------
    prices
        Positive aligned prices with dates in rows and tickers in columns.
    probabilities
        Forecast table indexed by evaluation date with ticker columns. Values
        are calibrated outperformance probabilities.
    policy
        Long-only trading policy.
    initial_cash
        Starting uninvested capital.
    periods_per_year
        Number of observations used for annualized metrics.

    Returns
    -------
    PolicyBacktestResult
        Equity, cash, holdings, trades, and summary metrics.
    """
    policy = policy or LongOnlyPolicy()
    clean = prices.astype(float).sort_index().dropna(how="any")
    if clean.empty or clean.shape[1] < 1 or (clean <= 0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    forecast = probabilities.reindex(columns=clean.columns).sort_index()
    if ((forecast < 0) | (forecast > 1)).any().any():
        raise ValueError("probabilities must lie in [0, 1]")

    tickers = list(clean.columns)
    shares = pd.Series(0.0, index=tickers)
    cash_value = float(initial_cash)
    equity_rows: list[float] = []
    cash_rows: list[float] = []
    position_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    pending: pd.Series | None = None
    for date, price in clean.iterrows():
        if pending is not None:
            portfolio_before = cash_value + float((shares * price).sum())
            current_weights = shares * price / portfolio_before
            desired = current_weights.copy()
            for ticker, probability in pending.dropna().items():
                desired.loc[ticker] = policy.target_weight(probability, current_weights.loc[ticker])
            total_desired = float(desired.sum())
            if total_desired > 1.0:
                desired /= total_desired
            target_values = desired * portfolio_before
            current_values = shares * price
            trade_values = target_values - current_values
            # Sell before buying so released capital is available.
            for ticker in sorted(tickers, key=lambda item: trade_values.loc[item]):
                trade_value = float(trade_values.loc[ticker])
                if abs(trade_value) < policy.minimum_trade_value:
                    continue
                if trade_value > 0:
                    trade_value = min(trade_value, cash_value / (1.0 + policy.transaction_cost_bps / 10_000.0))
                quantity = trade_value / float(price.loc[ticker])
                cost = abs(trade_value) * policy.transaction_cost_bps / 10_000.0
                shares.loc[ticker] += quantity
                cash_value -= trade_value + cost
                trades.append({
                    "date": date,
                    "ticker": ticker,
                    "side": "buy" if trade_value > 0 else "sell",
                    "quantity": abs(quantity),
                    "price": float(price.loc[ticker]),
                    "notional": abs(trade_value),
                    "cost": cost,
                    "probability": float(pending.loc[ticker]),
                })
            pending = None

        equity_value = cash_value + float((shares * price).sum())
        values = shares * price
        equity_rows.append(equity_value)
        cash_rows.append(cash_value)
        position_rows.append(shares.copy())
        weight_rows.append(values / equity_value)
        if date in forecast.index:
            pending = forecast.loc[date].copy()

    index = clean.index
    equity = pd.Series(equity_rows, index=index, name="equity")
    cash = pd.Series(cash_rows, index=index, name="cash")
    positions = pd.DataFrame(position_rows, index=index)
    weights = pd.DataFrame(weight_rows, index=index)
    trade_table = pd.DataFrame(trades, columns=["date", "ticker", "side", "quantity", "price", "notional", "cost", "probability"])
    metrics = _performance_metrics(equity, periods_per_year)
    metrics["transaction_costs"] = float(trade_table["cost"].sum()) if not trade_table.empty else 0.0
    metrics["trade_count"] = float(len(trade_table))
    return PolicyBacktestResult(equity, cash, positions, weights, trade_table, metrics)
