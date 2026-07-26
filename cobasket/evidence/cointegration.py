"""Cointegration evidence for long-only portfolio decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.cointegration import build_spread, johansen_test
from cobasket.signals import _as_real_series

from .base import AssetEvidence


@dataclass(frozen=True)
class CointegrationEvidenceResult:
    """Result of evaluating a cointegrated basket at its latest observation.

    Parameters
    ----------
    spread
        Historical weighted basket spread.
    z_score
        Rolling standardized displacement of the latest spread value.
    weights
        L1-normalized basket coefficients indexed by ticker.
    trace_statistic
        Johansen rank-zero trace statistic.
    critical_value_95
        Corresponding 95 percent critical value.
    trace_ratio
        Trace statistic divided by the 95 percent critical value.
    asset_evidence
        Long-only evidence for each asset in the basket.
    """

    spread: pd.Series
    z_score: pd.Series
    weights: pd.Series
    trace_statistic: float
    critical_value_95: float
    trace_ratio: float
    asset_evidence: tuple[AssetEvidence, ...]

    @property
    def latest_z_score(self) -> float:
        """Return the latest finite spread z-score.

        Returns
        -------
        float
            Most recent finite standardized spread displacement.

        Raises
        ------
        ValueError
            If no finite z-score is available.
        """
        finite = self.z_score.dropna()
        if finite.empty:
            raise ValueError("no finite z-score is available")
        return float(finite.iloc[-1])


def rolling_z_score(spread: pd.Series, window: int = 60) -> pd.Series:
    """Calculate a rolling z-score for a real-valued basket spread.

    Parameters
    ----------
    spread
        Basket spread time series.
    window
        Number of observations used for the local mean and standard deviation.

    Returns
    -------
    pandas.Series
        Rolling standardized displacement.

    Raises
    ------
    ValueError
        If ``window`` is smaller than two.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    values = _as_real_series(spread)
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    displacement = values - mean
    z_score = displacement / std.replace(0.0, np.nan)
    z_score = z_score.mask((std == 0.0) & (displacement == 0.0), 0.0)
    return z_score.rename("z_score")


def _confidence_from_statistics(
    z_score: float,
    trace_ratio: float,
    *,
    reference_z: float = 3.0,
    reference_trace_ratio: float = 1.5,
) -> float:
    """Map displacement and test strength to a bounded heuristic confidence.

    Parameters
    ----------
    z_score
        Latest standardized spread displacement.
    trace_ratio
        Johansen trace statistic divided by its 95 percent critical value.
    reference_z
        Absolute z-score at which the displacement component saturates.
    reference_trace_ratio
        Trace ratio at which the statistical-strength component saturates.

    Returns
    -------
    float
        Heuristic confidence in ``[0, 1]``.

    Notes
    -----
    This quantity is deliberately not called a probability. It combines two
    diagnostics monotonically and must later be calibrated against genuinely
    out-of-sample outcomes before receiving a probabilistic interpretation.
    """
    displacement_strength = np.clip(abs(z_score) / reference_z, 0.0, 1.0)
    relation_strength = np.clip(
        (trace_ratio - 1.0) / max(reference_trace_ratio - 1.0, np.finfo(float).eps),
        0.0,
        1.0,
    )
    return float(np.sqrt(displacement_strength * relation_strength))


def cointegration_evidence(
    prices: pd.DataFrame,
    *,
    window: int = 60,
    det_order: int = 0,
    k_ar_diff: int = 1,
    min_trace_ratio: float = 1.0,
) -> CointegrationEvidenceResult:
    """Convert a fitted cointegration relation into long-only asset evidence.

    Parameters
    ----------
    prices
        Aligned adjusted prices with dates in rows and assets in columns.
    window
        Rolling window used to standardize the basket spread.
    det_order
        Deterministic-term setting passed to the Johansen test.
    k_ar_diff
        Number of lagged first differences in the Johansen model.
    min_trace_ratio
        Minimum rank-zero trace-statistic ratio required before evidence is
        emitted.

    Returns
    -------
    CointegrationEvidenceResult
        Basket diagnostics and one evidence item per asset.

    Raises
    ------
    ValueError
        If the basket is not sufficiently cointegrated, has too few data, or
        does not yield a finite latest z-score.

    Notes
    -----
    For a positive spread displacement, mean reversion favours moving opposite
    to the fitted weight vector. Therefore an asset's signed long-only evidence
    is proportional to ``-z_score * weight``. Positive evidence means the asset
    is relatively cheap within this particular basket; negative evidence means
    relatively expensive. This is relative evidence, not an estimate of the
    company's intrinsic value.
    """
    if prices.shape[1] < 2:
        raise ValueError("cointegration evidence requires at least two assets")
    if len(prices) < window:
        raise ValueError("price history is shorter than the z-score window")

    result = johansen_test(
        prices,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        verbose=False,
    )
    trace_statistic = float(result.lr1[0])
    critical_value_95 = float(result.cvt[0, 1])
    trace_ratio = trace_statistic / critical_value_95
    if trace_ratio < min_trace_ratio:
        raise ValueError(
            "basket does not meet the required cointegration strength: "
            f"trace ratio={trace_ratio:.3f}, required={min_trace_ratio:.3f}"
        )

    spread, weight_array = build_spread(prices, result)
    z_scores = rolling_z_score(spread, window=window)
    finite = z_scores.dropna()
    if finite.empty:
        raise ValueError("the selected window produced no finite z-score")
    latest_z = float(finite.iloc[-1])

    weights = pd.Series(weight_array, index=prices.columns, name="weight", dtype=float)
    confidence = _confidence_from_statistics(latest_z, trace_ratio)
    raw_asset_scores = -latest_z * weights
    scale = float(np.max(np.abs(raw_asset_scores)))
    if scale <= np.finfo(float).eps:
        normalized_scores = pd.Series(0.0, index=weights.index)
    else:
        normalized_scores = (raw_asset_scores / scale).clip(-1.0, 1.0)

    evidence: list[AssetEvidence] = []
    for ticker in prices.columns:
        score = float(normalized_scores.loc[ticker] * confidence)
        relation = "relatively cheap" if score > 0 else "relatively expensive" if score < 0 else "neutral"
        summary = (
            f"{ticker} appears {relation} within this basket because the latest "
            f"spread displacement is {latest_z:+.2f} standard deviations and its "
            f"basket weight is {weights.loc[ticker]:+.3f}."
        )
        evidence.append(
            AssetEvidence(
                ticker=str(ticker),
                score=score,
                confidence=confidence,
                source="cointegration",
                summary=summary,
                z_score=latest_z,
                weight=float(weights.loc[ticker]),
            )
        )

    return CointegrationEvidenceResult(
        spread=spread,
        z_score=z_scores,
        weights=weights,
        trace_statistic=trace_statistic,
        critical_value_95=critical_value_95,
        trace_ratio=trace_ratio,
        asset_evidence=tuple(evidence),
    )


def evidence_table(result: CointegrationEvidenceResult) -> pd.DataFrame:
    """Convert cointegration evidence into a compact tabular representation.

    Parameters
    ----------
    result
        Result returned by :func:`cointegration_evidence`.

    Returns
    -------
    pandas.DataFrame
        One row per asset, sorted from most positive to most negative evidence.
    """
    rows = [
        {
            "ticker": item.ticker,
            "score": item.score,
            "confidence": item.confidence,
            "weight": item.weight,
            "z_score": item.z_score,
            "summary": item.summary,
        }
        for item in result.asset_evidence
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)
