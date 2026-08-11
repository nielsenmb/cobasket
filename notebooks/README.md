# Cobasket notebooks

The notebooks are ordered from the statistical foundations through the current long-only workflow, strategy testing, and calibration diagnostics.

| Notebook | Purpose |
|---|---|
| `01_data_and_basic_workflow.ipynb` | Foundations: adjusted-price data, Johansen cointegration, spread construction, and the original classical long/short spread backtest. This is useful for understanding the model, but it is not the recommended live long-only workflow. |
| `02_long_only_cointegration_evidence.ipynb` | Convert the cointegration relation into long-only relative-value evidence and raw recommendation labels. |
| `03_selection_watchlists_and_calibration.ipynb` | Current discovery workflow: screened watchlists, watchlist-level walk-forward calibration, saved calibration records, and linking calibration into `portfolio.json`. |
| `04_live_portfolio_report.ipynb` | Generate the same calibrated live portfolio report used by the CLI and GUI, inspect warnings/diagnostics, and save `report.json`. |
| `05_declarative_strategy_rules.ipynb` | Separate metrics from ordered buy/hold/reduce/sell rules and compare explicit strategies. |
| `06_price_metrics_and_strategy_comparison.ipynb` | Calculate momentum, trend, and volatility metrics and test their incremental value. |
| `07_controlled_strategy_experiments.ipynb` | Use a chronological training, validation, and untouched test split for strategy selection. |
| `08_repeated_walk_forward.ipynb` | Repeat the selection-and-test procedure across several market regimes. |
| `09_continuous_walk_forward.ipynb` | Carry one simulated account through successive folds and compare retain versus liquidate boundary policies. |
| `10_calibration_diagnostics.ipynb` | Diagnose evidence-score sign conventions, per-basket calibration, overlapping outcomes, and evidence-versus-future-return structure. |

## Recommended path

If the goal is to understand the statistics from first principles, start with Notebook 01 and continue in order.

If the goal is to use the current Cobasket long-only workflow, start with Notebook 02, then use 03 → 04. Notebooks 05–09 explain and test the strategy layer. Notebook 10 is a focused diagnostic for deciding whether the current evidence-to-probability calibration is statistically meaningful.

Notebook 01 intentionally retains the classical long/short spread model because that is the statistical object from which the long-only evidence model is derived. It is not obsolete code that the live portfolio is expected to execute.

## Statistical distinction

- Notebook 03 creates the empirical probability mapping used by live reports.
- Notebook 04 applies that mapping to the current portfolio/watchlist state.
- Notebook 07 evaluates one selected strategy on one untouched test interval.
- Notebook 08 repeats that procedure across several test periods, resetting capital in each fold.
- Notebook 09 carries cash and holdings continuously through time.
- Notebook 10 tests the calibration itself rather than a trading strategy. It checks Johansen sign invariance, separates baskets, compares overlapping and non-overlapping forecast windows, and inspects evidence continuously rather than only through fixed bins.

All experiment notebooks use trailing metrics and next-observation execution. Strategy selection uses validation data rather than the final test period.

## Inputs

Notebook 03 can generate `probability_calibration.json` and `calibration_records.parquet`. For normal use the equivalent command is:

```bash
cobasket-calibrate \
    --portfolio portfolio.json \
    --output probability_calibration.json \
    --records-out calibration_records.parquet \
    --update-portfolio
```

Notebook 10 expects `calibration_records.parquet`. Its non-overlapping-window section also uses `portfolio.json` to rerun calibration with `step=horizon`.

Notebook outputs are research diagnostics. They do not place trades or connect to a brokerage account.
