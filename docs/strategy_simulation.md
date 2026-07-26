# Basket strategy simulation

Cobasket can test one monitored basket end to end from **Portfolio → Simulate basket strategy…**.

The simulator repeatedly refits the basket using only prices available before each historical decision date. It then creates relative-value evidence, calibrates that evidence using only outcomes that had already completed by that date, converts the resulting probabilities into long-only target allocations, and executes trades on the next available price observation.

## Main controls

- **Training window**: history used to refit the cointegration relation.
- **Z-score window**: local history used to measure spread displacement.
- **Outcome horizon**: period over which relative outperformance is judged.
- **Evaluation step**: spacing between successive decisions.
- **Minimum calibration samples**: mature historical outcomes required before the probability may depart from 50%.
- **Buy, strong-buy, reduce, and sell probabilities**: decision thresholds.
- **Transaction cost**: one-way cost in basis points of traded value.

## Results

The window reports starting and ending value, net profit, equal-weight benchmark profit, excess profit, maximum drawdown, and trade count. It also shows the complete equity curve, drawdown, invested fraction, and trade ledger.

A positive historical profit does not imply that the same strategy will remain profitable. The result is most useful for comparing strategy choices under the same walk-forward assumptions and transaction-cost model.
