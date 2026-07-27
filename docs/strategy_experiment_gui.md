# Strategy editor and controlled experiments

Cobasket's **Portfolio → Strategy experiments…** window provides a graphical layer over the declarative strategy rules and controlled train/validation/test experiment framework.

## Required input tables

The experiment runner currently expects two CSV or Parquet files with dates in rows and tickers in columns:

- historical adjusted prices;
- historical calibrated outperformance probabilities.

The ticker columns should match. Price-based momentum, trend, and volatility metrics are generated automatically from the price table using trailing windows.

## Editing a strategy

Each row in the strategy editor is one ordered rule. Rules are evaluated from top to bottom, and the first matching rule determines the target weight.

The **Conditions** field uses semicolon-separated expressions:

```text
probability >= 0.60; stable == True; momentum >= 0.0
```

The **Match** column controls how those conditions are combined:

- `all`: every condition must pass;
- `any`: at least one condition must pass.

The target weight is a portfolio fraction. For example, `0.10` means a target allocation of 10%. Enter `hold`, `none`, or leave the field empty to preserve the current allocation.

Rules can be moved up or down to make priority explicit. Complete strategies can be saved to and loaded from the same JSON format used by the Python API.

## Running a controlled experiment

The runner divides the price history chronologically into:

1. training data, used for development diagnostics;
2. validation data, used to select one candidate strategy;
3. test data, used once to evaluate the selected strategy.

The training and validation fractions are configurable. Their sum must remain below one so that an untouched test interval remains.

The result tabs contain:

- **Training:** all candidate strategies;
- **Validation:** all candidate strategies, ranked using the selected metric;
- **Test:** the validation-selected strategy, equal-weight buy-and-hold, and cash;
- **Warnings:** multiple-testing and data-quality warnings from the experiment backend.

The GUI deliberately does not show test performance for every candidate. Doing so would encourage selecting a strategy using the test interval and would invalidate it as an independent check.

## Metric names

The automatically available price metrics are:

- `momentum`;
- `momentum_return`;
- `trend`;
- `trend_distance`;
- `volatility`;
- `volatility_percentile`;
- `high_volatility`.

The supplied probability table is exposed as `probability`. A strategy condition referencing a missing metric does not match and therefore cannot trigger a trade.

## Interpretation

This interface is an experiment builder, not an optimiser. Keep the candidate family small and pre-declared. Repeatedly modifying strategies after inspecting test results converts the test interval into another validation interval and overstates expected performance.
