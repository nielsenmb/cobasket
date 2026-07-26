"""Rolling diagnostics for cointegrated basket stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cobasket.cointegration import build_spread, johansen_test


@dataclass(frozen=True)
class BasketRobustnessResult:
    """Summary of rolling basket stability diagnostics.

    Parameters
    ----------
    rolling
        Table containing trace ratios, half-lives, weight drift, and break flags.
    latest_trace_ratio
        Most recent Johansen trace-statistic ratio.
    latest_half_life
        Most recent estimated mean-reversion half-life in observations.
    latest_weight_drift
        L1 distance between the latest and preceding normalized weight vectors.
    stable_fraction
        Fraction of successful rolling fits that pass the trace-ratio threshold.
    break_detected
        Whether the latest window is flagged as a possible structural break.
    warnings
        Human-readable diagnostic warnings.
    """

    rolling: pd.DataFrame
    latest_trace_ratio: float
    latest_half_life: float
    latest_weight_drift: float
    stable_fraction: float
    break_detected: bool
    warnings: tuple[str, ...]


def mean_reversion_half_life(spread: pd.Series) -> float:
    """Estimate the AR(1) mean-reversion half-life of a spread.

    Parameters
    ----------
    spread
        Finite spread observations in chronological order.

    Returns
    -------
    float
        Estimated half-life in observations. ``inf`` indicates no estimated
        mean reversion.
    """
    values = pd.Series(spread, dtype=float).dropna()
    if len(values) < 5:
        return float("nan")
    lagged = values.iloc[:-1].to_numpy()
    changed = values.iloc[1:].to_numpy() - lagged
    design = np.column_stack([np.ones(len(lagged)), lagged])
    slope = float(np.linalg.lstsq(design, changed, rcond=None)[0][1])
    if not np.isfinite(slope) or slope >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / slope)


def rolling_basket_robustness(
    prices: pd.DataFrame,
    *,
    window: int = 252,
    step: int = 20,
    min_trace_ratio: float = 1.0,
    max_half_life: float = 120.0,
    max_weight_drift: float = 0.50,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> BasketRobustnessResult:
    """Evaluate whether a fitted basket relationship remains stable through time.

    Parameters
    ----------
    prices
        Positive aligned price table with dates in rows and tickers in columns.
    window
        Number of observations in each rolling fit.
    step
        Number of observations between rolling fits.
    min_trace_ratio
        Minimum acceptable Johansen trace-statistic ratio.
    max_half_life
        Maximum acceptable spread half-life in observations.
    max_weight_drift
        Maximum acceptable L1 change between successive normalized weights.
    det_order, k_ar_diff
        Johansen deterministic-term and lag settings.

    Returns
    -------
    BasketRobustnessResult
        Rolling diagnostics and latest stability assessment.
    """
    clean = prices.astype(float).dropna(how="any").sort_index()
    if clean.shape[1] < 2:
        raise ValueError("robustness analysis requires at least two assets")
    if window < max(k_ar_diff + 5, 20) or len(clean) < window:
        raise ValueError("price history is too short for the requested rolling window")
    if step < 1:
        raise ValueError("step must be positive")

    rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    for end in range(window, len(clean) + 1, step):
        sample = clean.iloc[end - window : end]
        try:
            result = johansen_test(
                sample,
                det_order=det_order,
                k_ar_diff=k_ar_diff,
                verbose=False,
            )
            spread, weight_values = build_spread(sample, result)
            weights = pd.Series(weight_values, index=sample.columns, dtype=float)
            half_life = mean_reversion_half_life(spread)
            drift = (
                float((weights - previous_weights).abs().sum())
                if previous_weights is not None
                else 0.0
            )
            critical_value = float(result.cvt[0, 1])
            trace_ratio = float(result.lr1[0] / critical_value)
            stable = bool(
                trace_ratio >= min_trace_ratio
                and np.isfinite(half_life)
                and half_life <= max_half_life
                and drift <= max_weight_drift
            )
            rows.append(
                {
                    "date": sample.index[-1],
                    "trace_ratio": trace_ratio,
                    "half_life": half_life,
                    "weight_drift": drift,
                    "stable": stable,
                    **{
                        f"weight_{ticker}": float(weights.loc[ticker])
                        for ticker in weights.index
                    },
                }
            )
            previous_weights = weights
        except (ValueError, np.linalg.LinAlgError):
            rows.append(
                {
                    "date": sample.index[-1],
                    "trace_ratio": np.nan,
                    "half_life": np.nan,
                    "weight_drift": np.nan,
                    "stable": False,
                }
            )

    rolling = pd.DataFrame(rows).set_index("date")
    if rolling.empty:
        raise ValueError("no rolling robustness fits were produced")
    latest = rolling.iloc[-1]
    successful = rolling["trace_ratio"].notna()
    stable_fraction = (
        float(rolling.loc[successful, "stable"].mean()) if successful.any() else 0.0
    )
    break_detected = not bool(latest["stable"])
    warnings: list[str] = []
    if latest["trace_ratio"] < min_trace_ratio or pd.isna(latest["trace_ratio"]):
        warnings.append("The latest window does not provide strong cointegration evidence.")
    if not np.isfinite(latest["half_life"]) or latest["half_life"] > max_half_life:
        warnings.append("The latest spread mean-reverts too slowly for the configured limit.")
    if pd.notna(latest["weight_drift"]) and latest["weight_drift"] > max_weight_drift:
        warnings.append("The fitted basket weights changed substantially in the latest window.")
    if stable_fraction < 0.60:
        warnings.append("The basket has been stable in fewer than 60% of rolling fits.")
    return BasketRobustnessResult(
        rolling=rolling,
        latest_trace_ratio=float(latest["trace_ratio"]),
        latest_half_life=float(latest["half_life"]),
        latest_weight_drift=float(latest["weight_drift"]),
        stable_fraction=stable_fraction,
        break_detected=break_detected,
        warnings=tuple(warnings),
    )
