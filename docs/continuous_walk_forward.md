# Continuous walk-forward deployment

Repeated walk-forward evaluation asks whether the strategy-selection procedure works in several historical regimes. By default, each fold is evaluated as a separate experiment with a fresh account.

Continuous deployment answers a different, more practical question:

> What would have happened to one account if Cobasket periodically reselected its strategy and continued managing the same holdings?

## Workflow

For every fold, Cobasket still keeps the statistical separation:

1. candidate strategies are compared on the fold's validation interval;
2. one strategy is selected without using the fold's test data;
3. that strategy controls the account during the following test interval;
4. cash and holdings are carried into later folds.

The portfolio is marked to market through gaps between test intervals. No new rule decisions are made in those gaps, but existing holdings continue to gain or lose value.

## Boundary policies

`ContinuousDeploymentConfig` supports two behaviours when the selected strategy changes.

### `retain`

Existing holdings remain in the account. The newly selected rules alter them only when their conditions subsequently match.

This normally produces lower turnover, but it means a position opened by an earlier strategy can persist temporarily under a different strategy.

### `liquidate`

All holdings are sold at the first price observation of the new test interval when the selected strategy changes. The newly selected strategy then starts from cash.

This gives a cleaner separation between strategies, but it can create substantial turnover and transaction costs.

## Execution timing

Metric values observed on one date generate a decision that executes at the next available price observation. This is the same next-observation convention used elsewhere in Cobasket and avoids trading at the same closing price that produced the signal.

Strategy selection uses validation data only. Test prices can change the resulting account value, but they cannot change which strategy was selected for that fold.

## Outputs

`run_continuous_walk_forward()` returns:

- continuous strategy, equal-weight, and cash equity curves;
- cash, positions, and portfolio weights;
- a complete trade ledger;
- rule decisions and target weights;
- the strategy selected in each fold;
- performance statistics and transaction costs;
- warnings about strategy changes, liquidation turnover, and benchmark underperformance.

## Interpretation

This is closer to a brokerage-account simulation than the compounded repeated-fold summary, but it remains a historical model. Results still depend on:

- the candidate strategies supplied;
- fold lengths and selection metric;
- transaction-cost assumptions;
- the quality and survivorship properties of the ticker universe;
- whether the historical price source represents prices that could actually have been traded.

A strong result should remain reasonably stable when fold boundaries, transaction costs, and boundary policies are varied. A strategy that is profitable only under one exact configuration is likely overfit.
