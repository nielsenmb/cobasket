# Repeated walk-forward experiments

Cobasket's repeated walk-forward window runs the complete strategy-selection procedure across several chronological market regimes.

Open it from:

```text
Portfolio → Repeated walk-forward experiments…
```

The input files use the same format as the single-split strategy experiment window:

- a dated price table with tickers in columns;
- a dated probability table with the same ticker columns;
- one or more declarative strategy JSON files, or strategies created in the editor.

## Fold controls

A fold contains three successive intervals:

```text
training → validation → test
```

- **Training observations** define the historical development interval.
- **Validation observations** are used to select one candidate strategy.
- **Test observations** evaluate only that selected strategy.
- **Fold step** determines how far the complete experiment moves forward before the next fold.

By default, the fold step is at least as long as the test interval. This keeps test intervals non-overlapping, so the same market movement is not counted repeatedly.

With **Use expanding training window** enabled, the training start remains fixed and the training sample grows. Otherwise, the training window rolls forward and old observations drop out.

Overlapping test intervals require explicit opt-in. Their results are correlated and must not be treated as independent experiments.

## Results

The Overview tab shows:

- the strategy, equal-weight, and cash values compounded across test folds;
- the fraction of folds in which each candidate strategy was selected.

The compounded curve is an out-of-sample summary, not a literal continuous brokerage simulation. Every fold starts with the configured capital and no inherited positions.

The Folds tab reports each fold's dates, selected strategy, test return, Sharpe ratio, drawdown, trade count, benchmark return, and excess return.

The Selections tab shows how frequently each candidate strategy was chosen. Frequent changes indicate that the preferred rules depend on the market regime.

The Warnings tab retains both fold-specific warnings and aggregate warnings, including:

- different strategies selected across folds;
- failure to beat equal-weight buy-and-hold in most test folds;
- overlapping test intervals;
- small validation samples or excessive candidate counts.

## Interpretation

Repeated walk-forward testing evaluates the strategy-selection process, not merely one fixed rule set. A robust process should ideally:

- choose similar strategies across comparable regimes;
- beat the benchmark in more than an isolated fold;
- avoid depending on a single unusually favourable test period;
- retain acceptable drawdown and turnover across folds.

The export action writes fold summaries, selection frequencies, compounded curves, warnings, and the strategy selected in every fold.
