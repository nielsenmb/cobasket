# Robustness-aware selection and simulation

A cointegrated basket can stop behaving like the relationship used to fit it. Cobasket can now use the rolling robustness diagnostics as a gate during candidate selection and historical strategy simulation.

## Strategy gate

At each historical evaluation date, Cobasket uses only prices available on or before that date. The gate checks:

- the latest Johansen trace ratio;
- the estimated spread half-life;
- drift in the fitted cointegration weights;
- the fraction of recent rolling fits classified as stable.

When the gate fails, the historical probability is replaced with the neutral value `0.5`. Under the current long-only policy, this means:

- no new position is opened;
- no existing position is increased;
- no forced sale is generated solely by the robustness failure;
- the current position is retained until valid evidence returns.

This is deliberately more conservative than interpreting instability as a sell signal.

## Comparing filtered and unfiltered strategies

The strategy simulator displays three equity curves:

1. robustness-filtered strategy;
2. the same strategy without filtering;
3. equal-weight buy-and-hold.

The comparison reports profit, drawdown, trade count, and the fraction of evaluation dates that passed the stability gate. Robustness filtering is useful only if it improves out-of-sample behaviour; fewer trades by itself is not evidence of a better strategy.

## Candidate selection

`filter_candidate_baskets_by_robustness()` can be applied after the normal universe screen. It retains only candidates whose latest rolling window is stable and whose historical stable fraction exceeds the configured threshold.

The thresholds remain configurable research choices. They should be tuned using walk-forward validation rather than selected to maximise performance on the complete historical sample.
