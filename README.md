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

Cobasket supports validated adjusted-price downloads and caching; persistence-aware basket discovery; correlation- and PCA-based candidate screening; Johansen cointegration diagnostics; persistent holdings and watchlists; historical basket validation profiles; pooled and basket-specific probability calibration; long-only recommendations; FX-aware portfolio valuation; declarative trading rules; transaction costs and position sizing; single-split, repeated walk-forward, and continuous-account backtests; recommendation history; and a PyQt dashboard.

## Intended workflow

```text
Choose a stock universe
        ↓
Discover persistent candidate baskets
        ↓
Historically validate each basket
        ↓
Fit probability calibration only for validated baskets
        ↓
Maintain current holdings
        ↓
Refresh current prices and basket state
        ↓
Generate Buy / Add / Hold / Reduce / Wait recommendations
```

Discovery is a candidate-selection stage rather than a recommendation. Validation tests whether each relationship persists across independent historical periods before calibration is allowed to drive an action.

## Basket discovery

The normal discovery command combines candidate clustering, current Johansen screening, historical persistence and weight-stability checks, and a preliminary backtest:

```bash
cobasket-discover \
    --universe sp500 \
    --period 5y \
    --watchlist-out discovered_watchlist.json \
    --table-out discovery_results.csv
```

Built-in universes include `sp500`, `nasdaq100`, `ftse100`, and `eurostoxx50`. Discovery classifies surviving candidates as `promising`, `borderline`, or `reject`; only promising baskets are exported by default.

Lower-level residual-correlation and PCA screens remain available through `cobasket-screen` and `cobasket-pca-screen` for research and diagnostics.

## Validate discovered baskets

After discovery has created or updated `portfolio.json`, run:

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

## GUI workflow

The GUI is organized around a **workspace directory**, not around manually choosing individual JSON files. Use **Open / update workspace…** on the main screen for normal operation. Cobasket inspects the workspace and recommends the next action.

Typical states are:

```text
Empty workspace
→ Discover baskets
→ Edit holdings/base currency if desired
→ Validate
→ Calibrate
→ Generate live report

Partial workspace
→ Cobasket detects the first missing or stale downstream stage
→ Continue from that stage

Complete workspace
→ Refresh recommendations
→ Re-run discovery only when you want to reconsider the basket universe
```

The workspace window shows the status of Discovery, Validation, Calibration, and Live report. **Next step** runs only the recommended stage. **Update required stages to report** runs all missing or stale downstream stages in order. Re-running discovery is deliberately separate because it can replace the watchlist and invalidate validation/calibration for the previous baskets.

Cobasket also applies age-based freshness recommendations. The defaults are:

```text
Live report       3 days
Validation       90 days
Calibration      90 days
Discovery       180 days
```

A report or model artifact older than its configured interval is marked for refresh even when no upstream file has changed. An old discovery is advisory rather than automatic: Cobasket warns that the basket universe should be reconsidered but does not silently replace the current watchlist. Use **Freshness settings…** in the workspace window to change these intervals. Settings are stored in `workspace_freshness.json` inside the workspace and do not alter the statistical models themselves.

Manual `portfolio.json` and `report.json` controls are hidden by default but can be enabled from **Workspace → Show manual file controls**.

The **Portfolio** menu contains everyday portfolio actions such as editing holdings, ticker investigation, and recommendation history. Historical simulations, strategy experiments, repeated walk-forward tests, continuous deployment, and policy validation are under **Research** so they are not confused with the normal recommendation workflow.

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

## Notebooks

See [`notebooks/README.md`](notebooks/README.md) for the current notebook sequence. CI validates notebook structure and syntax-checks ordinary Python code cells.

## Caveats

- Historical profitability does not imply future profitability.
- Initial basket discovery can produce relationships that are temporary, unstable, or non-predictive; use validation before acting on them.
- Testing many baskets or rule sets can overfit noise.
- Cointegration relationships and market regimes can break down.
- Adjusted closing prices do not reproduce intraday execution, bid-ask spreads, taxes, or all broker-specific costs.
- Yahoo Finance downloads can be incomplete or rate-limited; inspect data-quality warnings before interpreting results.
- Cobasket currently supports research and manual decision support rather than automatic brokerage execution.
