"""Backtesting utilities for mean-reverting spread strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cobasket.cointegration import build_spread, normalize_cointegration_weights
from cobasket.signals import zscore_signal


@dataclass(frozen=True)
class BacktestResult:
    """Container for the output of a spread-strategy backtest.

    Attributes
    ----------
    units
        Number of units of each asset held when the spread position is ``+1``.
    position
        Executed position after applying the one-observation signal delay.
    gross_exposure
        Absolute market value of all long and short legs, expressed relative to
        the initial capital unit.
    pnl
        Daily profit and loss in units of initial capital.
    returns
        Daily PnL divided by the initial capital, which is one by construction.
    costs
        Transaction costs in units of initial capital.
    equity
        Initial capital plus cumulative net PnL.
    drawdown
        Fractional decline from the previous equity maximum.
    sharpe
        Annualized mean return divided by return standard deviation.
    total_return
        Final equity minus initial equity.
    max_drawdown
        Most negative drawdown.
    n_trades
        Total absolute position change. A reversal from ``+1`` to ``-1`` counts
        as two one-way trades because every leg is closed and reopened reversed.
    """

    units: pd.Series
    position: pd.Series
    gross_exposure: pd.Series
    pnl: pd.Series
    returns: pd.Series
    costs: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int


def spread_units(
    initial_prices: Sequence[float],
    weights: Sequence[float],
    gross_notional: float = 1.0,
) -> np.ndarray:
    """Convert spread coefficients into tradable asset units.

    The scale is chosen so that the sum of the absolute market values of all
    legs equals ``gross_notional`` at the initial prices. This quantity is
    called *gross exposure*: long and short market values are both counted as
    positive because both consume risk capacity.

    Parameters
    ----------
    initial_prices
        Asset prices at the start of the trading window.
    weights
        Cointegration coefficients defining the direction of the synthetic
        spread.
    gross_notional
        Desired sum of absolute initial leg values. A value of one expresses
        all PnL relative to one unit of initial capital.

    Returns
    -------
    numpy.ndarray
        Fixed number of units held in each asset for a ``+1`` spread position.

    Raises
    ------
    ValueError
        If prices, weights, or gross notional are invalid.
    """
    prices = np.asarray(initial_prices, dtype=float).reshape(-1)
    normalized_weights = normalize_cointegration_weights(weights)
    if prices.size != normalized_weights.size:
        raise ValueError("initial prices and weights must have the same length")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("initial prices must be finite and strictly positive")
    if not np.isfinite(gross_notional) or gross_notional <= 0:
        raise ValueError("gross_notional must be finite and positive")

    initial_gross = float(np.sum(np.abs(normalized_weights * prices)))
    if initial_gross <= np.finfo(float).eps:
        raise ValueError("initial gross exposure is numerically zero")
    return normalized_weights * (gross_notional / initial_gross)


def run_backtest(
    prices: pd.DataFrame,
    weights: Sequence[float],
    signal: pd.Series,
    cost_bps: float = 10.0,
    gross_notional: float = 1.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Evaluate a fixed-unit long/short spread strategy.

    Parameters
    ----------
    prices
        Aligned asset prices with dates in rows and assets in columns.
    weights
        Cointegration coefficients defining the spread direction.
    signal
        Desired position state: ``+1`` long spread, ``-1`` short spread, and
        ``0`` no position.
    cost_bps
        Transaction cost in basis points per unit of gross position change.
        One basis point is 0.01 percent.
    gross_notional
        Initial absolute market value across all spread legs.
    periods_per_year
        Number of observations per year used to annualize the Sharpe ratio.

    Returns
    -------
    BacktestResult
        Daily accounting series and summary statistics.

    Notes
    -----
    The fitted spread coefficients have arbitrary scale, like an eigenvector.
    They are converted into fixed asset units whose initial gross exposure is
    ``gross_notional``. Daily PnL is then the dot product of price changes and
    those units. Equity is initial capital plus cumulative PnL; it is not
    compounded by silently reinvesting profits.

    Raises
    ------
    TypeError
        If ``prices`` or ``signal`` has the wrong type.
    ValueError
        If inputs are empty, misaligned, non-finite, or otherwise invalid.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if prices.empty:
        raise ValueError("prices must not be empty")
    if len(weights) != prices.shape[1]:
        raise ValueError("weight vector length must match the number of assets")
    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("cost_bps must be finite and non-negative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    aligned_signal = signal.reindex(prices.index)
    if aligned_signal.isna().any():
        raise ValueError("signal must be defined on every price date")
    if not aligned_signal.isin([-1, 0, 1]).all():
        raise ValueError("signal values must be -1, 0, or +1")

    price_values = prices.to_numpy(dtype=float)
    if not np.isfinite(price_values).all() or (price_values <= 0).any():
        raise ValueError("prices must be finite and strictly positive")

    unit_values = spread_units(price_values[0], weights, gross_notional)
    units = pd.Series(unit_values, index=prices.columns, name="spread_units")

    # Delay the signal by one observation so today's close cannot be used to
    # generate a position that profits from today's already-observed move.
    position = aligned_signal.shift(1).fillna(0).astype(float).rename("position")

    price_changes = prices.diff().fillna(0.0)
    unit_pnl = pd.Series(
        price_changes.to_numpy(dtype=float) @ unit_values,
        index=prices.index,
        name="unit_spread_pnl",
    )
    gross_exposure = pd.Series(
        np.abs(prices.to_numpy(dtype=float) * unit_values).sum(axis=1),
        index=prices.index,
        name="gross_exposure",
    )

    gross_pnl = (position * unit_pnl).rename("gross_pnl")
    turnover = position.diff().abs().fillna(position.abs())
    costs = (turnover * gross_exposure * (cost_bps / 10_000.0)).rename("costs")
    net_pnl = (gross_pnl - costs).rename("pnl")

    # Initial capital is one by construction; therefore numerical PnL and
    # return have the same values but retain separate names and semantics.
    returns = net_pnl.rename("returns")
    equity = (1.0 + returns.cumsum()).rename("equity")
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).rename("drawdown")

    volatility = float(returns.std())
    sharpe = (
        float(np.sqrt(periods_per_year) * returns.mean() / volatility)
        if np.isfinite(volatility) and volatility > np.finfo(float).eps
        else float("nan")
    )
    total_return = float(equity.iloc[-1] - 1.0)
    max_drawdown = float(drawdown.min())
    n_trades = int(turnover.sum())

    return BacktestResult(
        units=units,
        position=position,
        gross_exposure=gross_exposure,
        pnl=net_pnl,
        returns=returns,
        costs=costs,
        equity=equity,
        drawdown=drawdown,
        sharpe=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
    )


def print_backtest_summary(result: BacktestResult) -> None:
    """Print the principal statistics from a backtest result.

    Parameters
    ----------
    result
        Result returned by :func:`run_backtest`.
    """
    print("\nBacktest summary:")
    print(f"  Total return: {result.total_return * 100:.1f}%")
    print(f"  Annualized Sharpe: {result.sharpe:.2f}")
    print(f"  Max drawdown: {result.max_drawdown * 100:.1f}%")
    print(f"  Number of trades: {result.n_trades}")


def backtest(
    prices: pd.DataFrame,
    weights: Sequence[float],
    signal: pd.Series,
    cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series]:
    """Run a spread backtest and return its equity and drawdown series.

    This function preserves the original public API. New code should generally
    call :func:`run_backtest` to access the complete accounting output.

    Parameters
    ----------
    prices
        Aligned asset prices.
    weights
        Cointegration coefficients defining the spread.
    signal
        Position state: ``+1`` long spread, ``-1`` short spread, ``0`` flat.
    cost_bps
        Transaction cost in basis points per unit of gross position change.

    Returns
    -------
    equity
        Initial capital plus cumulative net PnL.
    drawdown
        Fractional decline from the previous equity maximum.
    """
    result = run_backtest(prices, weights, signal, cost_bps=cost_bps)
    print_backtest_summary(result)
    return result.equity, result.drawdown


def backtest_single_basket(
    prices: pd.DataFrame,
    tickers: Sequence[str] | None = None,
    cost_bps: float = 10.0,
) -> dict[str, object]:
    """Fit a basket on the first half and trade it on the second half.

    Parameters
    ----------
    prices
        Complete aligned price history for the basket.
    tickers
        Optional ticker labels used when printing weights. Defaults to the
        DataFrame column names.
    cost_bps
        Transaction cost in basis points per unit of gross position change.

    Returns
    -------
    dict
        Estimation and trading products, including weights, spread, signal,
        equity, drawdown, and the full :class:`BacktestResult`.

    Raises
    ------
    ValueError
        If either half contains too few observations.
    """
    split = len(prices) // 2
    estimation_prices = prices.iloc[:split]
    trading_prices = prices.iloc[split:]
    if len(estimation_prices) < 3 or len(trading_prices) < 3:
        raise ValueError("both estimation and trading windows need at least 3 rows")

    johansen_result = coint_johansen(
        estimation_prices.to_numpy(dtype=float),
        0,
        1,
    )
    estimation_spread, weights = build_spread(estimation_prices, johansen_result)

    labels = list(tickers) if tickers is not None else list(prices.columns)
    print("\nBasket weights (fit on estimation window only):")
    for ticker, weight in zip(labels, weights):
        print(f"  {ticker}: {weight:.3f}")

    trading_spread = pd.Series(
        trading_prices.to_numpy(dtype=float) @ weights,
        index=trading_prices.index,
        name="spread",
        dtype=float,
    )
    z_score, signal = zscore_signal(trading_spread)
    result = run_backtest(trading_prices, weights, signal, cost_bps=cost_bps)
    print_backtest_summary(result)

    return {
        "weights": weights,
        "estimation_prices": estimation_prices,
        "estimation_spread": estimation_spread,
        "trading_prices": trading_prices,
        "trading_spread": trading_spread,
        "z": z_score,
        "signal": signal,
        "units": result.units,
        "position": result.position,
        "pnl": result.pnl,
        "returns": result.returns,
        "costs": result.costs,
        "equity": result.equity,
        "drawdown": result.drawdown,
        "backtest_result": result,
    }


def rank_confirmed_baskets(
    confirmed,
    prices: pd.DataFrame,
    cost_bps: float = 10.0,
) -> list[dict[str, object]]:
    """Refit, backtest, and rank confirmed baskets by Sharpe ratio.

    Parameters
    ----------
    confirmed
        Basket tuples returned by ``cointegration.screen_universe``.
    prices
        Price table containing all basket symbols.
    cost_bps
        Transaction cost in basis points per unit of gross position change.

    Returns
    -------
    list of dict
        Basket statistics sorted from highest to lowest finite Sharpe ratio.
    """
    ranked: list[dict[str, object]] = []
    for basket, _, statistic, critical in confirmed:
        basket_prices = prices[basket].dropna()
        split = len(basket_prices) // 2
        estimation_prices = basket_prices.iloc[:split]
        trading_prices = basket_prices.iloc[split:]

        if len(estimation_prices) < 50 or len(trading_prices) < 50:
            continue

        try:
            johansen_result = coint_johansen(
                estimation_prices.to_numpy(dtype=float),
                0,
                1,
            )
            _, weights = build_spread(estimation_prices, johansen_result)
            trading_spread = pd.Series(
                trading_prices.to_numpy(dtype=float) @ weights,
                index=trading_prices.index,
                name="spread",
                dtype=float,
            )
            _, signal = zscore_signal(trading_spread)
            result = run_backtest(
                trading_prices,
                weights,
                signal,
                cost_bps=cost_bps,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  Skipping {basket} (backtest failed): {exc}")
            continue

        if not np.isfinite(result.sharpe):
            continue

        ranked.append(
            {
                "basket": basket,
                "weights": weights,
                "sharpe": result.sharpe,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
                "n_trades": result.n_trades,
                "johansen_stat": statistic,
                "johansen_crit": critical,
            }
        )

    ranked.sort(key=lambda item: float(item["sharpe"]), reverse=True)
    return ranked


def print_ranked_results(results: Sequence[dict[str, object]], top_n: int = 10) -> None:
    """Print a compact table of ranked basket results.

    Parameters
    ----------
    results
        Output returned by :func:`rank_confirmed_baskets`.
    top_n
        Maximum number of baskets to print.
    """
    print(f"\nTop {min(top_n, len(results))} baskets by Sharpe ratio:\n")
    for index, result in enumerate(results[:top_n]):
        print(f"{index + 1}. {result['basket']}")
        print(
            f"   Sharpe: {result['sharpe']:.2f}  |  "
            f"Return: {result['total_return'] * 100:.1f}%  |  "
            f"Max DD: {result['max_drawdown'] * 100:.1f}%  |  "
            f"Trades: {result['n_trades']}"
        )
