# Continuous walk-forward deployment GUI

The continuous deployment window answers a practical historical question:

> What would have happened to one account if Cobasket periodically reselected its strategy while carrying cash and holdings forward?

Open it from **Portfolio → Continuous walk-forward deployment…**.

## Inputs

The window uses the same historical tables and strategy JSON format as the other experiment views:

- a dated adjusted-price table;
- a dated calibrated-probability table with matching ticker columns;
- one or more declarative candidate strategies.

Momentum, trend, volatility, and high-volatility metrics are generated from the supplied prices using trailing windows.

## Fold controls

Each fold contains:

1. a training interval for development diagnostics;
2. a validation interval used to select one candidate strategy;
3. a test interval in which the selected strategy manages the simulated account.

The fold step must be at least as long as the test interval. Continuous deployment does not permit overlapping test intervals because one account cannot follow two independently selected strategies on the same date.

**Rolling training** keeps a fixed training length and drops old observations. **Expanding training** retains all observations before the validation period.

## Strategy-change policy

### Retain

Existing holdings remain in the account when the selected strategy changes. The new strategy modifies them only when one of its rules matches.

### Liquidate

All holdings are sold at the first observation of a new test interval when the selected strategy differs from the preceding fold. Transaction costs are charged before the new strategy begins operating.

Comparing the two policies helps identify whether apparent performance depends on excessive strategy-switching turnover.

## Results

The interface displays:

- continuous strategy, equal-weight, and cash values;
- strategy drawdown;
- invested portfolio fraction;
- summary performance metrics;
- selected strategy and fold boundaries;
- all trades, including boundary liquidations;
- rule decisions and target weights;
- model-selection and data-quality warnings.

Trades generated from a decision at date `t` execute at the next available price observation. This prevents same-observation look-ahead.

## Interpretation

Continuous deployment is more operationally realistic than resetting the account in every fold, but it is still a historical simulation. It does not model intraday execution, bid-ask spreads, taxes, broker restrictions, or market impact.

The equal-weight benchmark buys each ticker at the start of the continuous evaluation interval and holds it. Cash remains constant and represents the no-investment baseline.
