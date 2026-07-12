"""Backtesting a spread strategy, single-basket and multi-basket ranking."""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cobasket.cointegration import build_spread
from cobasket.signals import zscore_signal


def backtest(prices, weights, signal, cost_bps=10):
    """
    Simple long/short spread backtest.
    signal: +1 = long the spread (buy low leg, sell high leg per `weights`)
            -1 = short the spread
             0 = flat
    cost_bps: round-trip transaction cost in basis points (1 bps = 0.01%)
              charged whenever the position changes.
    """
    spread_returns = prices.pct_change().values @ weights
    spread_returns = pd.Series(spread_returns, index=prices.index)

    # shift signal by 1 day: you trade on today's close using yesterday's
    # signal, so you're not using today's return to generate today's trade
    position = signal.shift(1).fillna(0)

    strat_returns = position * spread_returns

    # transaction cost applied whenever position changes
    trades = position.diff().abs().fillna(0)
    costs = trades * (cost_bps / 10000)
    strat_returns = strat_returns - costs

    equity = (1 + strat_returns).cumprod()

    sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    print("\nBacktest summary:")
    print(f"  Total return: {(equity.iloc[-1] - 1) * 100:.1f}%")
    print(f"  Annualized Sharpe: {sharpe:.2f}")
    print(f"  Max drawdown: {max_drawdown * 100:.1f}%")
    print(f"  Number of trades: {int(trades.sum())}")

    return equity, drawdown


def backtest_single_basket(prices, tickers, cost_bps=10):
    """
    Convenience wrapper: split prices in half, fit Johansen on the first
    half (estimation window), freeze those weights, and backtest on the
    second half (trading window). Avoids look-ahead bias -- the trading
    window never influences the weights used to trade it.
    """
    split = len(prices) // 2
    estimation_prices = prices.iloc[:split]
    trading_prices = prices.iloc[split:]

    result = coint_johansen(estimation_prices.values, 0, 1)
    spread_est, weights = build_spread(estimation_prices, result)

    print("\nBasket weights (fit on estimation window only):")
    for t, w in zip(tickers, weights):
        print(f"  {t}: {w:.3f}")

    trading_spread = pd.Series(
        trading_prices.values @ weights, index=trading_prices.index, name="spread"
    )
    z, signal = zscore_signal(trading_spread)
    equity, drawdown = backtest(trading_prices, weights, signal, cost_bps=cost_bps)

    return {
        "weights": weights,
        "trading_prices": trading_prices,
        "trading_spread": trading_spread,
        "z": z,
        "signal": signal,
        "equity": equity,
        "drawdown": drawdown,
    }


def rank_confirmed_baskets(confirmed, prices, cost_bps=10):
    """
    For each confirmed basket (from cointegration.screen_universe), refit
    weights on an estimation window (first half) and backtest on the
    trading window (second half). Ranks by Sharpe ratio.
    """
    results = []
    for basket, _, stat, crit in confirmed:
        basket_prices = prices[basket].dropna()
        split = len(basket_prices) // 2
        estimation_prices = basket_prices.iloc[:split]
        trading_prices = basket_prices.iloc[split:]

        if len(estimation_prices) < 50 or len(trading_prices) < 50:
            continue

        try:
            est_result = coint_johansen(estimation_prices.values, 0, 1)
        except Exception as e:
            print(f"  Skipping {basket} (re-fit failed): {e}")
            continue

        _, weights = build_spread(estimation_prices, est_result)
        trading_spread = pd.Series(
            trading_prices.values @ weights, index=trading_prices.index
        )
        z, signal = zscore_signal(trading_spread)

        spread_returns = trading_prices.pct_change().values @ weights
        spread_returns = pd.Series(spread_returns, index=trading_prices.index)
        position = signal.shift(1).fillna(0)
        strat_returns = position * spread_returns
        trades = position.diff().abs().fillna(0)
        strat_returns = strat_returns - trades * (cost_bps / 10000)
        equity = (1 + strat_returns).cumprod()

        if strat_returns.std() == 0 or strat_returns.std() != strat_returns.std():
            continue  # no variance (e.g. never traded) -> skip

        sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
        total_return = equity.iloc[-1] - 1
        running_max = equity.cummax()
        max_drawdown = ((equity - running_max) / running_max).min()
        n_trades = int(trades.sum())

        results.append({
            "basket": basket,
            "weights": weights,
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "johansen_stat": stat,
            "johansen_crit": crit,
        })

    # drop any NaN Sharpe (e.g. degenerate all-zero signal) before ranking
    results = [r for r in results if r["sharpe"] == r["sharpe"]]
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def print_ranked_results(results, top_n=10):
    print(f"\nTop {min(top_n, len(results))} baskets by Sharpe ratio:\n")
    for i, r in enumerate(results[:top_n]):
        print(f"{i + 1}. {r['basket']}")
        print(
            f"   Sharpe: {r['sharpe']:.2f}  |  Return: {r['total_return']*100:.1f}%  "
            f"|  Max DD: {r['max_drawdown']*100:.1f}%  |  Trades: {r['n_trades']}"
        )
