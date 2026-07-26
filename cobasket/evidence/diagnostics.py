"""Diagnostics for probabilistic recommendation calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Summary statistics for probabilistic forecasts.

    Parameters
    ----------
    brier_score
        Mean squared error between forecast probabilities and binary outcomes.
    log_loss
        Mean binary cross-entropy, with probabilities clipped away from zero.
    expected_calibration_error
        Weighted mean absolute difference between predicted and observed rates.
    baseline_brier_score
        Brier score of a constant forecast equal to the sample event rate.
    sample_count
        Number of valid forecast/outcome pairs.
    reliability
        Table containing reliability-bin counts, mean forecasts, and outcomes.
    """

    brier_score: float
    log_loss: float
    expected_calibration_error: float
    baseline_brier_score: float
    sample_count: int
    reliability: pd.DataFrame


def reliability_table(
    probabilities: pd.Series | np.ndarray,
    outcomes: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Construct a reliability table for binary probability forecasts.

    Parameters
    ----------
    probabilities
        Forecast probabilities in ``[0, 1]``.
    outcomes
        Binary observations containing zero or one.
    n_bins
        Number of equal-width probability bins.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin with count, mean forecast, observed rate,
        calibration error, and bin limits.

    Raises
    ------
    ValueError
        If no valid pairs are available or ``n_bins`` is less than two.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    frame = pd.DataFrame({"probability": probabilities, "outcome": outcomes})
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[frame["outcome"].isin([0, 1])]
    frame = frame[frame["probability"].between(0.0, 1.0)]
    if frame.empty:
        raise ValueError("no valid probability/outcome pairs are available")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.searchsorted(edges, frame["probability"].to_numpy(), side="right") - 1
    index = np.clip(index, 0, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        selected = frame.iloc[np.flatnonzero(index == bin_index)]
        if selected.empty:
            continue
        predicted = float(selected["probability"].mean())
        observed = float(selected["outcome"].mean())
        rows.append(
            {
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "sample_count": int(len(selected)),
                "mean_probability": predicted,
                "observed_frequency": observed,
                "calibration_error": observed - predicted,
            }
        )
    return pd.DataFrame(rows)


def calibration_diagnostics(
    probabilities: pd.Series | np.ndarray,
    outcomes: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-12,
) -> CalibrationDiagnostics:
    """Evaluate discrimination-independent probability calibration metrics.

    Parameters
    ----------
    probabilities
        Forecast probabilities in ``[0, 1]``.
    outcomes
        Binary observations containing zero or one.
    n_bins
        Number of bins used for the reliability table and ECE.
    epsilon
        Probability clipping value used for logarithmic loss.

    Returns
    -------
    CalibrationDiagnostics
        Calibration metrics and reliability table.
    """
    table = reliability_table(probabilities, outcomes, n_bins=n_bins)
    frame = pd.DataFrame({"probability": probabilities, "outcome": outcomes})
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[frame["outcome"].isin([0, 1])]
    frame = frame[frame["probability"].between(0.0, 1.0)]
    probability = frame["probability"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=float)
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    brier = float(np.mean((probability - outcome) ** 2))
    log_loss = float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)))
    event_rate = float(outcome.mean())
    baseline = float(np.mean((event_rate - outcome) ** 2))
    ece = float(
        np.sum(
            table["sample_count"]
            * np.abs(table["calibration_error"])
        )
        / len(frame)
    )
    return CalibrationDiagnostics(
        brier_score=brier,
        log_loss=log_loss,
        expected_calibration_error=ece,
        baseline_brier_score=baseline,
        sample_count=int(len(frame)),
        reliability=table,
    )


def attach_calibrated_probabilities(
    records: pd.DataFrame,
    calibration,
) -> pd.DataFrame:
    """Attach fitted calibration probabilities to walk-forward records.

    Parameters
    ----------
    records
        Walk-forward table containing an evidence ``score`` column.
    calibration
        Object providing a ``lookup(score)`` method, such as
        :class:`~cobasket.evidence.ProbabilityCalibration`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``records`` with posterior probability and interval columns.
    """
    if "score" not in records:
        raise ValueError("records must contain a score column")
    output = records.copy()
    rows = [calibration.lookup(float(score)) for score in output["score"]]
    output["probability_outperform"] = [float(row["probability_mean"]) for row in rows]
    output["probability_lower"] = [float(row["probability_lower"]) for row in rows]
    output["probability_upper"] = [float(row["probability_upper"]) for row in rows]
    output["calibration_sample_count"] = [int(row["sample_count"]) for row in rows]
    return output
