# Price-based strategy metrics

Cobasket provides three deliberately simple metric families for the declarative
strategy engine. They are inputs to a strategy, not recommendations by themselves.

## Momentum

Momentum is the fractional return over a trailing window:

\[
m_t = \frac{P_t}{P_{t-w}} - 1.
\]

A positive value means the price has risen over the lookback interval. Cobasket
also supplies a bounded score,

\[
\tilde m_t = \tanh(m_t/s_m),
\]

which lies between -1 and +1. The scale `s_m` controls how quickly the score
saturates. This transformation makes rule thresholds easier to compare across
assets, but it does not turn momentum into a probability.

## Trend

Trend is the fractional distance from a trailing simple moving average:

\[
q_t = \frac{P_t}{\overline{P}_{t,w}} - 1.
\]

Positive values indicate that the current price is above its recent baseline.
The bounded `trend` score applies the same hyperbolic-tangent mapping as momentum.
Momentum and trend are related, but not identical: momentum compares two epochs,
whereas trend compares the current value with an average over many epochs.

## Volatility

Volatility is the trailing sample standard deviation of daily returns, annualised
with `sqrt(periods_per_year)`. It measures variability rather than direction.
A rapidly rising stock and a rapidly falling stock can both have high volatility.

Cobasket also calculates `volatility_percentile`, the current value's rank within
its trailing historical volatility window. `high_volatility` is a binary flag set
when that percentile exceeds the configured threshold.

## Leakage controls

All calculations are trailing:

- moving averages are not centred;
- no backward filling is used;
- volatility percentiles use only values observed by the current date;
- early rows remain missing until enough history exists.

Appending future prices must not alter metric values already calculated for past
dates. This property is tested directly.

## Rule examples

A price metric should generally modify a strategy rather than replace the
cointegration probability. For example:

```python
StrategyRule(
    action="buy",
    conditions=(
        MetricCondition("probability", ">=", 0.60),
        MetricCondition("stable", "==", True),
        MetricCondition("momentum", ">", 0.0),
        MetricCondition("high_volatility", "==", False),
    ),
    target_weight=0.10,
)
```

A useful comparison sequence is:

1. probability only;
2. probability plus robustness;
3. probability plus robustness and momentum;
4. add a trend requirement;
5. add volatility as an exclusion or position-size control.

The thresholds should be declared before examining the final test period. Adding
metrics and selecting whichever historical combination performs best creates a
multiple-testing problem and can easily fit noise.

## API

```python
from cobasket.price_metrics import PriceMetricConfig, build_price_metrics

metrics = build_price_metrics(
    prices,
    config=PriceMetricConfig(
        momentum_window=60,
        trend_window=100,
        volatility_window=20,
        volatility_baseline_window=252,
    ),
)
```

The returned mapping contains:

- `momentum`;
- `momentum_return`;
- `trend`;
- `trend_distance`;
- `volatility`;
- `volatility_percentile`;
- `high_volatility`.

Use `merge_metric_tables` to combine them with cointegration probabilities and
robustness flags. Use `compare_incremental_metric_strategies` to run several
pre-declared rule sets under identical historical assumptions.
