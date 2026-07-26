# Declarative strategy rules

Cobasket separates **metrics** from **trading rules**. A metric is a measured quantity, such as a calibrated cointegration probability or a basket-stability flag. A strategy is the ordered set of conditions that maps those measurements onto target portfolio weights.

This is analogous to separating observables from the physical model used to interpret them. Adding another observable does not automatically improve the model; its contribution must be tested through an explicit rule set.

## Example

```python
from cobasket.strategy_rules import (
    MetricCondition,
    StrategyRule,
    StrategyRules,
)

strategy = StrategyRules(
    name="cointegration and stability",
    rules=(
        StrategyRule(
            action="sell",
            conditions=(MetricCondition("probability", "<=", 0.30),),
            target_weight=0.0,
        ),
        StrategyRule(
            action="strong buy",
            conditions=(
                MetricCondition("probability", ">=", 0.70),
                MetricCondition("stable", "==", True),
            ),
            target_weight=0.20,
        ),
        StrategyRule(
            action="buy",
            conditions=(
                MetricCondition("probability", ">=", 0.60),
                MetricCondition("stable", "==", True),
            ),
            target_weight=0.10,
        ),
        StrategyRule(
            action="reduce",
            conditions=(
                MetricCondition("probability", "<=", 0.40),
                MetricCondition("is_held", "==", True),
            ),
            target_weight=0.05,
        ),
    ),
)
```

Rules are evaluated from top to bottom. The first matching rule wins. This makes overlaps explicit: a probability of 0.75 matches both `strong buy` and `buy`, but the earlier `strong buy` rule takes priority.

If no rule matches, Cobasket keeps the current weight. Missing or non-finite metrics do not satisfy a condition.

## Historical simulation

```python
from cobasket.strategy_rules import run_rule_strategy_backtest

result = run_rule_strategy_backtest(
    prices,
    metrics={
        "probability": probability_history,
        "stable": stability_history,
    },
    strategy=strategy,
    initial_cash=10_000.0,
)
```

A decision made on one price observation is executed on the next available observation. Metric tables are never forward-filled by the rule engine, so a value is only used on a date where it was explicitly supplied.

The result contains:

- equity, cash, positions, and weights;
- transaction costs and trade history;
- the action and target weight selected at every evaluation date;
- total and annualized return, volatility, Sharpe ratio, and maximum drawdown.

## Comparing strategies

```python
from cobasket.strategy_rules import compare_rule_strategies

summary, detailed = compare_rule_strategies(
    prices,
    metrics,
    strategies=(cointegration_only, cointegration_and_stability),
)
```

Each strategy receives the same prices, metric history, initial capital, and execution timing. This allows an incremental comparison such as:

1. cointegration probability alone;
2. cointegration plus robustness;
3. cointegration plus momentum;
4. cointegration plus momentum and volatility.

A more complicated strategy should only be retained when it improves performance on validation and held-out periods, not merely on the data used to choose its thresholds.

## Saving strategies

```python
strategy.save("strategies/cointegration_stable.json")
loaded = StrategyRules.load("strategies/cointegration_stable.json")
```

The JSON representation preserves rule order. Strategy files can therefore be version-controlled and associated with specific backtest results.

## Overfitting warning

Every added metric and threshold increases the number of possible strategies. Selecting the best result from many combinations can produce an apparently successful strategy by chance. Cobasket should use training, validation, and untouched test periods, or a nested walk-forward procedure, when comparing rule sets.
