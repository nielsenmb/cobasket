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

## Current capabilities

Cobasket supports validated adjusted-price downloads and caching; correlation- and PCA-based candidate-basket screening; Johansen cointegration diagnostics; persistent holdings and watchlists; historical basket validation profiles; pooled and basket-specific probability calibration; long-only recommendations; declarative trading rules; transaction costs and position sizing; single-split, repeated walk-forward, and continuous-account backtests; recommendation history; and a PyQt dashboard.

## Intended workflow

```text
Choose a stock universe
        ↓
Screen for plausible candidate baskets
        ↓
Historically validate each basket
        ↓
Fit probability calibration only for validated baskets
        ↓
Maintain current holdings and watchlist
        ↓
Refresh current prices and basket state
        ↓
Generate Buy / Add / Hold / Reduce / Wait recommendations
```

The screening stage is deliberately permissive. Passing the initial Johansen screen or preliminary backtest does **not** make a basket suitable for live recommendations. Validation tests whether the relationship persists across independent historical periods before calibration is allowed to drive an action.

## Basket discovery

Residual-correlation screening:

```bash
cobasket-screen \
    --period 5y \
    --top-n 20 \
    --watchlist-out screened_watchlist.json
```

PCA screening:

```bash
cobasket-pca-screen \
    --period 5y \
    --top-n 20 \
    --watchlist-out pca_watchlist.json
```

The output baskets are hypotheses to investigate, not recommendations.

## Validate discovered baskets

After placing the screened watchlist in `portfolio.json`, run:

```bash
cobasket-validate \
    --portfolio portfolio.json \
    --output basket_validation.json \
    --update-portfolio
```

Each basket is classified as:

- `validated`: current and historical criteria support using the relation;
- `weak`: reasonably stable, but predictive evidence or independent history is insufficient;
- `unstable`: the relation appears only intermittently or the fitted weights vary too much;
- `rejected`: the relationship currently fails the required cointegration test.

By default validation uses a 20-trading-day outcome horizon and 20-day spacing, so successive validation outcomes do not overlap.

## Fit basket-specific probabilities

Once validation exists:

```bash
cobasket-basket-calibrate \
    --portfolio portfolio.json \
    --validation basket_validation.json \
    --output basket_calibration.json \
    --update-portfolio
```

Only `validated` baskets with enough independent historical evaluation dates receive a probability calibration. Skipped baskets are recorded with a reason. When basket-specific calibration is active, Cobasket does not silently fall back to the pooled calibration for missing baskets.

A reported probability refers to historical **relative outperformance versus the equal-weight return of the same basket** over the configured horizon. It is not the probability that the stock price rises in absolute terms.

## Live report

Generate a current report with:

```bash
cobasket-report \
    --portfolio portfolio.json \
    --output report.json
```

Or launch the dashboard:

```bash
cobasket-gui
```

The portfolio editor preserves the watchlist, pooled calibration, basket validation, and basket-specific calibration paths. If none of a ticker's supporting baskets is validated, or no validated basket has enough independent history for calibration, the live action is conservatively reduced to `Wait` when unheld or `Hold` when already owned while retaining the underlying diagnostics.

## Historical evaluation

Cobasket provides three increasingly realistic levels of strategy testing.

### Single controlled split

One chronological training, validation, and test split. Candidate strategies are selected using validation performance and evaluated once on the untouched test interval.

### Repeated walk-forward evaluation

The selection procedure is repeated across several folds. This tests whether preferred strategies and their performance remain stable across market regimes.

### Continuous walk-forward deployment

One simulated account is carried through successive test periods. Cash and positions persist while Cobasket periodically reselects the strategy. All historical decisions execute on the next available price observation rather than the same observation used to calculate the metrics.

## Important terminology

- **Long-only:** Cobasket can hold an asset or cash; it does not require short selling.
- **Target weight:** desired fraction of portfolio value allocated to a ticker.
- **Fold:** one complete training, validation, and test trial.
- **Drawdown:** decline from the portfolio's previous highest value.
- **Turnover:** amount of portfolio value traded over time.
- **Benchmark:** a simpler comparison strategy, such as equal-weight buy-and-hold or cash.
- **Johansen weights:** coefficients defining a synthetic statistical spread; they are not portfolio allocations.

## GUI workflows

The **Portfolio** menu includes portfolio/watchlist editing, ticker investigation, recommendation history, basket strategy simulation, single-split strategy experiments, repeated walk-forward experiments, continuous deployment, and historical validation. A final simplified workflow view is planned to surface basket discovery, validation status, and current recommendations more directly.

## Notebooks

See [`notebooks/README.md`](notebooks/README.md) for the current notebook sequence. CI validates notebook structure and syntax-checks ordinary Python code cells.

## Caveats

- Historical profitability does not imply future profitability.
- Initial basket screening can produce relationships that are temporary, unstable, or non-predictive; use validation before acting on them.
- Testing many baskets or rule sets can overfit noise.
- Cointegration relationships and market regimes can break down.
- Adjusted closing prices do not reproduce intraday execution, bid-ask spreads, taxes, or all broker-specific costs.
- Yahoo Finance downloads can be incomplete or rate-limited; inspect data-quality warnings before interpreting results.
- Cobasket currently supports research and manual decision support rather than automatic brokerage execution.
