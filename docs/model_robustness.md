# Basket model robustness

A basket can pass a cointegration test today and still be unsuitable for a strategy if its relationship is unstable through time. Cobasket now evaluates monitored baskets in rolling windows and separates several failure modes.

## Diagnostics

- **Trace ratio** measures whether each rolling window still supports a cointegrating relation. Values below one fail the configured 95% threshold.
- **Mean-reversion half-life** estimates how many trading observations a displacement takes to decay by half under an AR(1) approximation. A very long or infinite half-life means the spread is not returning quickly enough to be useful.
- **Weight drift** is the L1 distance between normalized Johansen weight vectors in consecutive windows. Large changes mean the fitted definition of the spread itself is unstable.
- **Stable fraction** is the fraction of successful rolling fits that pass all configured limits.
- **Structural-break flag** is raised when the latest window fails one or more limits.

The diagnostics are available through:

```python
from cobasket.robustness import rolling_basket_robustness

result = rolling_basket_robustness(
    prices,
    window=252,
    step=20,
    min_trace_ratio=1.0,
    max_half_life=120,
    max_weight_drift=0.5,
)

print(result.stable_fraction)
print(result.warnings)
result.rolling.plot(subplots=True)
```

`investigate_basket()` also attaches a robustness result automatically when enough history is available.

These thresholds are filters, not physical constants. They should eventually be tuned inside the same walk-forward framework used for strategy evaluation. A basket that appears profitable only when unstable periods are included is not a reliable candidate for live use.
