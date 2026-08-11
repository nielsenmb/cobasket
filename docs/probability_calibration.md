# Probability calibration

Cobasket's raw cointegration evidence score is a signed relative-value quantity. It is useful for ranking evidence, but a score such as `0.35` is not itself a probability.

The watchlist calibration workflow converts that score into an empirical probability using historical walk-forward outcomes.

## What the probability means

For each historical evaluation date, Cobasket:

1. uses only the preceding training window to fit the basket relation;
2. calculates the evidence score available on that date;
3. waits a fixed forecast horizon;
4. measures each ticker's return over that horizon;
5. compares the ticker return with the equal-weight return of the other basket context;
6. records whether the ticker outperformed the equal-weight basket benchmark.

Historical records are pooled across the monitored watchlist only after each basket has been evaluated without future information.

A calibrated probability of `0.65` therefore means approximately:

> Historical examples in this evidence-score range outperformed their equal-weight basket benchmark with posterior probability 0.65 over the configured horizon.

It does **not** mean there is a 65% probability that the ticker's absolute price will rise.

## Build a calibration

Starting from a portfolio configuration and screened watchlist:

```bash
cobasket-calibrate \
    --portfolio portfolio.json \
    --output probability_calibration.json \
    --records-out calibration_records.parquet \
    --train-window 252 \
    --horizon 20 \
    --step 5 \
    --update-portfolio
```

`--update-portfolio` writes the calibration file into `calibration_path` in `portfolio.json`. The next `cobasket-report` or GUI analysis then uses calibrated probabilities automatically.

The defaults correspond roughly to:

- 252 trading observations for each historical basket fit;
- a 20-trading-observation forward outcome horizon;
- one evaluation every 5 trading observations;
- the z-score window and minimum Johansen trace ratio already stored in `portfolio.json`.

## Calibration table

The output JSON contains one row per evidence-score interval. Important columns are:

- `score_lower`, `score_upper`: evidence-score interval;
- `sample_count`: number of historical asset outcomes in that interval;
- `successes`: number that outperformed the equal-weight basket;
- `probability_mean`: posterior mean outperformance probability;
- `probability_lower`, `probability_upper`: central credible interval.

Cobasket uses a beta-binomial model with a `Beta(1, 1)` prior by default. This prevents small or empty bins from producing unjustified probabilities of exactly zero or one.

## Interpreting uncertainty

A probability estimate from a small bin is weak evidence even if its posterior mean is high. For example, a bin with three historical examples should not be treated the same as a bin with several hundred examples.

The live report therefore carries both the calibration sample count and probability interval, and warns when a bin has few examples or a wide interval.

## Relationship to strategy testing

Calibration and strategy testing answer different questions:

- calibration asks whether a given evidence score historically corresponded to relative outperformance;
- strategy testing asks whether explicit buy/hold/reduce/sell rules based on those probabilities would have produced acceptable portfolio performance after costs and risk.

A well-calibrated probability mapping does not by itself imply a profitable trading strategy.
