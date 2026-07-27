# Cobasket

Cobasket is a Python research and decision-support tool for finding related stock baskets, measuring relative-value evidence, defining explicit long-only trading rules, and testing those rules without using future information.

It is intended for learning and research. It does not place trades and should not be treated as financial advice.

## Installation

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[test]"
pytest
```

For notebooks or the PyQt interface:

```bash
pip install -e ".[notebook,gui]"
cobasket-gui
```

The package version and changes planned for the next release are recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Current capabilities

Cobasket supports:

- validated adjusted-price downloads and per-ticker caching;
- correlation- and PCA-based candidate-basket screening;
- Johansen cointegration tests and fitted spread diagnostics;
- rolling robustness checks, including trace strength, weight drift, and mean-reversion half-life;
- persistent holdings and watchlists, including zero-quantity re-entry candidates;
- calibrated per-ticker probabilities with uncertainty intervals;
- long-only recommendations and configurable position limits;
- momentum, trend, volatility, and cointegration metrics;
- ordered declarative buy, hold, reduce, and sell rules;
- transaction costs, cash constraints, position sizing, exits, and re-entry;
- single-split train/validation/test experiments;
- repeated walk-forward experiments across several market regimes;
- continuous walk-forward deployment with one account carried through time;
- recommendation, user-action, and outcome history in SQLite;
- a PyQt dashboard for live reports, diagnostics, strategy experiments, and backtests.

## Core workflow

```text
Choose a stock universe
        ↓
Screen and validate candidate baskets
        ↓
Maintain a watchlist and current holdings
        ↓
Calculate cointegration and price-based metrics
        ↓
Define explicit decision rules
        ↓
Select strategies using validation data only
        ↓
Evaluate on unseen historical periods
        ↓
Generate and store live recommendations
```

Metrics are observations, not trading instructions. The strategy layer states explicitly how those observations map to portfolio actions.

## First live report

Copy the example files into a working directory:

```bash
cp examples/portfolio.json .
cp examples/portfolio_watchlist.json .
```

The example starts with no shares held and $10,000 in uninvested cash. Edit the ticker quantities and baskets before treating it as your portfolio state.

Generate a report:

```bash
cobasket-report \
    --portfolio portfolio.json \
    --watchlist portfolio_watchlist.json \
    --output report.json
```

Or launch the dashboard:

```bash
cobasket-gui
```

## Other command-line workflows

Backtest a basket:

```bash
cobasket-backtest XOM CVX COP OXY --period 2y
```

Screen a universe using residual correlation:

```bash
cobasket-screen --period 2y --distance-threshold 0.8
```

Screen using PCA factor loadings:

```bash
cobasket-pca-screen --period 2y --n-remove 1 --distance-threshold 1.5
```

## Python example

The stable top-level API covers the main research and reporting interfaces:

```python
from cobasket import (
    DataManager,
    MetricCondition,
    StrategyRule,
    StrategyRules,
    build_price_metrics,
)

prices = DataManager(cache_dir="price_cache").prices(
    ["AAPL", "MSFT", "GOOG"],
    period="5y",
)
metrics = build_price_metrics(prices)

strategy = StrategyRules(
    name="probability, trend, and volatility",
    rules=(
        StrategyRule(
            action="sell",
            conditions=(MetricCondition("probability", "<=", 0.30),),
            target_weight=0.0,
        ),
        StrategyRule(
            action="buy",
            conditions=(
                MetricCondition("probability", ">=", 0.60),
                MetricCondition("trend", ">=", 0.0),
                MetricCondition("high_volatility", "==", False),
            ),
            target_weight=0.10,
        ),
    ),
)
```

Rules are evaluated from top to bottom. The first matching rule determines the target portfolio weight. Missing metrics do not match.

## Historical validation levels

Cobasket provides three increasingly realistic forms of historical evaluation.

### Single controlled split

One chronological training, validation, and test split. Candidate strategies are selected using validation performance; the selected strategy is evaluated once on the untouched test interval.

### Repeated walk-forward evaluation

The entire selection procedure is repeated across several folds. Each fold has its own training, validation, and test periods. This tests whether the preferred strategy and its performance remain stable across market regimes.

### Continuous walk-forward deployment

One simulated brokerage account is carried through successive test periods. Cash and positions persist while Cobasket periodically reselects the strategy. The boundary policy can either retain positions or liquidate them when the selected strategy changes.

All historical decisions execute on the next available price observation rather than the same observation used to calculate the metrics.

## Important terminology

- **Long-only:** Cobasket can hold an asset or cash, but does not require short selling.
- **Target weight:** the desired fraction of total portfolio value allocated to a ticker.
- **Fold:** one complete training, validation, and test trial within repeated walk-forward analysis.
- **Drawdown:** the percentage decline from the portfolio's previous highest value.
- **Turnover:** how much portfolio value is traded over time; high turnover generally increases costs.
- **Benchmark:** a simpler comparison strategy, such as equal-weight buy-and-hold or remaining in cash.

The fitted Johansen weight vector defines a synthetic statistical spread. It is not automatically a recommended long-only portfolio allocation.

## GUI workflows

The **Portfolio** menu includes:

- portfolio and watchlist editing;
- ticker and basket investigation;
- recommendation history;
- basket strategy simulation;
- single-split strategy experiments;
- repeated walk-forward experiments;
- continuous walk-forward deployment;
- historical policy and calibration validation.

The continuous deployment view shows the account equity, benchmark values, drawdown, invested fraction, selected-strategy timeline, trade ledger, decisions, and warnings.

## Notebooks

The notebooks are ordered to follow the development and research workflow. See [`notebooks/README.md`](notebooks/README.md) for the current map. CI validates every notebook file and syntax-checks ordinary Python code cells.

## Release checks

Pull requests run:

- the core suite on Python 3.10 and 3.12;
- notebook structure and code-cell smoke tests;
- source and wheel builds followed by `twine check`;
- installation and import from the built wheel;
- the PyQt suite under Xvfb.

## Caveats

- Historical profitability does not imply future profitability.
- Testing many rule sets can overfit noise; use small, pre-declared candidate families.
- Cointegration relationships and market regimes can break down.
- Adjusted closing prices do not reproduce intraday execution, bid-ask spreads, taxes, or all broker-specific costs.
- The equal-weight benchmark is useful but not necessarily the appropriate benchmark for every portfolio.
- Yahoo Finance downloads can be incomplete or rate-limited; inspect data-quality warnings before interpreting results.
- Cobasket currently supports research and manual decision support rather than automatic brokerage execution.
