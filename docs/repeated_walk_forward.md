# Repeated walk-forward evaluation

A single train/validation/test split can give a misleading answer when its test interval happens to represent one unusually favourable or unfavourable market regime. Repeated walk-forward evaluation runs the same controlled experiment several times as the calendar advances.

Each fold contains three disjoint intervals:

1. **Training** — used for strategy development diagnostics.
2. **Validation** — used to choose one pre-declared candidate strategy.
3. **Test** — used once to evaluate the selected strategy.

The next fold moves all three intervals forward. Test periods are non-overlapping by default.

```python
from cobasket.repeated_walk_forward import (
    WalkForwardConfig,
    run_repeated_walk_forward,
)
from cobasket.strategy_experiments import StrategyExperimentConfig

result = run_repeated_walk_forward(
    prices,
    metrics,
    strategies,
    walk_forward=WalkForwardConfig(
        train_observations=504,
        validation_observations=126,
        test_observations=126,
        step_observations=126,
    ),
    experiment=StrategyExperimentConfig(
        selection_metric="sharpe_ratio",
        initial_cash=10_000.0,
    ),
)
```

## Outputs

`fold_table` contains one row per untouched test interval, including:

- selected strategy;
- test return and Sharpe ratio;
- maximum drawdown;
- trade count;
- equal-weight benchmark return;
- excess return over the benchmark.

`selection_frequency` shows how often each rule set was selected. Frequent changes in the selected strategy indicate regime dependence: the preferred rules are not stable through time.

`compounded_equity` chains the non-overlapping test returns. It is a compact out-of-sample summary, not a literal continuous brokerage simulation, because each fold starts with the same configured capital and no inherited positions.

## Statistical cautions

- Candidate strategies must be declared before reviewing fold test results.
- Adding candidates after inspecting the results turns the repeated tests into further validation data.
- Overlapping test intervals are disabled by default. If enabled, fold outcomes are correlated and cannot be interpreted as independent replications.
- A strategy selected in most folds may still underperform buy-and-hold.
- The number of folds is usually small, so uncertainty remains substantial.

This stage answers a stronger question than a single backtest:

> Does the strategy-selection procedure repeatedly work across different historical regimes?
