# Controlled strategy experiments

Cobasket separates metric calculation, trading rules, and evaluation. A strategy
experiment adds a further separation between the data used while developing a
strategy and the data used to assess whether it generalises.

## Three chronological intervals

`ExperimentSplit` defines three disjoint intervals:

- **Training:** inspect behaviour and develop a small family of plausible rules.
- **Validation:** compare the pre-declared candidate strategies and choose one.
- **Test:** evaluate the chosen strategy once, without changing it afterward.

This is analogous to fitting several physical models, selecting among them using
one dataset, and reserving a final independent dataset for the reported result.
The test interval ceases to be independent if its result is repeatedly inspected
while thresholds or rules are adjusted.

```python
from cobasket.strategy_experiments import (
    ExperimentSplit,
    StrategyExperimentConfig,
    run_strategy_experiment,
)

split = ExperimentSplit.from_fractions(
    prices.index,
    train_fraction=0.50,
    validation_fraction=0.25,
)

experiment = run_strategy_experiment(
    prices,
    metrics,
    candidate_strategies,
    split,
    config=StrategyExperimentConfig(
        selection_metric="sharpe_ratio",
        initial_cash=10_000.0,
        maximum_candidates=10,
    ),
)
```

Candidate performance is reported for both training and validation, but the
winner is selected from validation results only. Only that winner is run on the
test interval. The test table also includes:

- an equal-weight buy-and-hold basket;
- an all-cash benchmark.

## Selection metrics

Supported validation criteria are:

- total return;
- annualised return;
- Sharpe ratio;
- maximum drawdown;
- annualised volatility.

A higher value is preferred for returns and Sharpe ratio. Maximum drawdown is
usually negative, so the value closest to zero is preferred. Lower volatility is
preferred when volatility is the selection criterion.

No one metric is universally best. For example, selecting purely by return may
favour a strategy with an unacceptable drawdown. It is usually better to declare
a primary selection metric and inspect the remaining metrics as constraints.

## Bounded parameter grids

Small threshold grids can be generated with a hard combination limit:

```python
from cobasket.strategy_experiments import strategies_from_grid

strategies = strategies_from_grid(
    strategy_factory,
    {
        "buy_probability": [0.60, 0.65],
        "minimum_momentum": [0.0, 0.1],
    },
    maximum_combinations=10,
)
```

The limit is intentional. Testing many combinations and retaining the best one
creates a multiple-comparisons problem: one strategy can look successful merely
because many alternatives were tried.

Cobasket warns when:

- more than ten candidates are compared;
- the candidate count is large relative to the validation sample;
- training and validation prefer different strategies;
- the selected strategy makes no test-period trades.

These are warnings rather than automatic rejection rules, because the acceptable
number of candidates depends on the amount and independence of the data.

## Exporting an experiment

```python
experiment.save("results/experiment_001")
```

The directory contains:

- `train_results.csv`;
- `validation_results.csv`;
- `test_results.csv`;
- `selected_strategy.json`;
- `experiment.json` with the split and warnings.

This makes the experiment reproducible and records which strategy was selected
before the test result was examined.

## Important limitations

The current framework resets the portfolio to the configured starting cash at
the beginning of each interval. This is appropriate for comparing behaviour on
isolated periods, but it is not a continuous live-portfolio simulation.

The framework also assumes that supplied metrics are already leakage-safe. The
price metrics added in Stage 9B use trailing windows, while cointegration
probabilities require their own walk-forward calculation.

A single train/validation/test split can be sensitive to the chosen dates. The
next validation stage should add repeated or nested walk-forward experiments to
measure how stable the selected strategy is across market regimes.
