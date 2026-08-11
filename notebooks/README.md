# Cobasket notebooks

The notebooks are ordered from the basic data and cointegration workflow to the most rigorous historical strategy simulations and calibration diagnostics.

| Notebook | Purpose |
|---|---|
| `01_data_and_basic_workflow.ipynb` | Download adjusted prices, fit a cointegrated spread, interpret long/short spread signals, and run the original spread backtest. |
| `02_long_only_evidence.ipynb` | Convert relative-value evidence into long-only asset-level diagnostics and recommendations. |
| `03_watchlists_and_calibration.ipynb` | Select candidate baskets, persist watchlists, and calibrate historical evidence into probabilities. |
| `04_live_portfolio_report.ipynb` | Generate a structured live portfolio report with probabilities, uncertainty, recommendations, and warnings. |
| `05_declarative_strategy_rules.ipynb` | Define ordered buy, hold, reduce, and sell rules and compare explicit strategies. |
| `06_price_metrics_and_strategy_comparison.ipynb` | Calculate momentum, trend, and volatility metrics and test their incremental value. |
| `07_controlled_strategy_experiments.ipynb` | Use a chronological training, validation, and untouched test split for strategy selection. |
| `08_repeated_walk_forward.ipynb` | Repeat the selection-and-test procedure across several market regimes. |
| `09_continuous_walk_forward.ipynb` | Carry one account through successive folds and compare retain versus liquidate boundary policies. |
| `10_calibration_diagnostics.ipynb` | Diagnose evidence-score sign conventions, per-basket calibration, overlapping outcomes, and evidence-versus-future-return structure. |

## Recommended path

For learning the package, run notebooks 01–06 in order. Notebooks 07–09 address increasingly strict forms of historical validation and should be used before interpreting a strategy as potentially useful. Notebook 10 is a focused diagnostic for deciding whether the current evidence-to-probability calibration is statistically meaningful.

## Statistical distinction

- Notebook 07 evaluates one selected strategy on one untouched test interval.
- Notebook 08 repeats that procedure across several independent test periods, but resets capital in each fold.
- Notebook 09 carries cash and holdings through time, producing one continuous simulated account.
- Notebook 10 tests the calibration itself rather than a trading strategy. It checks Johansen sign invariance, separates baskets, compares overlapping and non-overlapping forecast windows, and inspects evidence continuously rather than only through fixed bins.

All experiment notebooks use only trailing metrics and next-observation execution. Strategy selection uses validation data rather than the test period.

## Inputs

Most later notebooks use aligned historical price and probability tables. The probability represents expected relative outperformance within a basket over the configured horizon; it is not necessarily the probability that the ticker's absolute price will rise.

Notebook 10 expects `calibration_records.parquet` from `cobasket-calibrate --records-out calibration_records.parquet`. Its non-overlapping-window section also uses `portfolio.json` to rerun calibration with `step=horizon`.

Notebook outputs are research diagnostics. They do not place trades or connect to a brokerage account.
