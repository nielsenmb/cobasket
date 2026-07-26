"""Walk-forward calibration for long-only cointegration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from .base import AssetEvidence
from .cointegration import cointegration_evidence


DEFAULT_SCORE_EDGES = (-1.0, -0.60, -0.25, 0.25, 0.60, 1.0)


@dataclass(frozen=True)
class CalibratedAssetEvidence:
    """Asset evidence augmented by an empirical outperformance probability.

    Parameters
    ----------
    evidence
        Original transparent evidence record.
    probability_outperform
        Posterior mean probability that the asset outperforms its equal-weight
        basket benchmark over the selected forecast horizon.
    probability_lower
        Lower credible bound for ``probability_outperform``.
    probability_upper
        Upper credible bound for ``probability_outperform``.
    sample_count
        Number of walk-forward examples in the relevant score bin.
    horizon
        Forecast horizon in trading observations.
    benchmark
        Description of the return benchmark used during calibration.
    """

    evidence: AssetEvidence
    probability_outperform: float
    probability_lower: float
    probability_upper: float
    sample_count: int
    horizon: int
    benchmark: str = "equal-weight basket"

    @property
    def ticker(self) -> str:
        """Return the normalized asset ticker.

        Returns
        -------
        str
            Asset symbol.
        """
        return self.evidence.ticker


@dataclass(frozen=True)
class ProbabilityCalibration:
    """Bayesian binned calibration from evidence score to outcome probability.

    Parameters
    ----------
    table
        Calibration table with one row per evidence-score interval.
    score_edges
        Monotonically increasing score-bin edges.
    horizon
        Forward-return horizon in trading observations.
    prior_alpha
        Beta-prior success shape parameter.
    prior_beta
        Beta-prior failure shape parameter.
    credible_level
        Central posterior credible interval probability.
    benchmark
        Description of the return benchmark.
    """

    table: pd.DataFrame
    score_edges: tuple[float, ...]
    horizon: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    credible_level: float = 0.68
    benchmark: str = "equal-weight basket"

    def lookup(self, score: float) -> pd.Series:
        """Return the calibration row containing an evidence score.

        Parameters
        ----------
        score
            Signed evidence score in ``[-1, 1]``.

        Returns
        -------
        pandas.Series
            Matching calibration-bin statistics.

        Raises
        ------
        ValueError
            If ``score`` is non-finite or outside ``[-1, 1]``.
        """
        score = float(score)
        if not np.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("score must be finite and lie in [-1, 1]")
        index = np.searchsorted(np.asarray(self.score_edges), score, side="right") - 1
        index = min(max(index, 0), len(self.table) - 1)
        return self.table.iloc[index]

    def calibrate(self, evidence: AssetEvidence) -> CalibratedAssetEvidence:
        """Attach an empirical outperformance probability to evidence.

        Parameters
        ----------
        evidence
            Current asset evidence.

        Returns
        -------
        CalibratedAssetEvidence
            Evidence plus posterior probability and uncertainty interval.
        """
        row = self.lookup(evidence.score)
        return CalibratedAssetEvidence(
            evidence=evidence,
            probability_outperform=float(row["probability_mean"]),
            probability_lower=float(row["probability_lower"]),
            probability_upper=float(row["probability_upper"]),
            sample_count=int(row["sample_count"]),
            horizon=self.horizon,
            benchmark=self.benchmark,
        )


def walk_forward_evidence(
    prices: pd.DataFrame,
    *,
    train_window: int = 252,
    z_window: int = 60,
    horizon: int = 20,
    step: int = 5,
    min_trace_ratio: float = 1.0,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Generate genuinely out-of-sample evidence and outcomes through time.

    At each evaluation date, the basket relation is fitted using only the
    preceding ``train_window`` observations. Asset returns are then measured
    over the next ``horizon`` observations. The target is whether each asset
    outperformed the equal-weight return of the same basket.

    Parameters
    ----------
    prices
        Aligned adjusted prices with dates in rows and assets in columns.
    train_window
        Number of historical observations available at each model fit.
    z_window
        Rolling window used to standardize the fitted spread.
    horizon
        Number of future observations over which outcomes are measured.
    step
        Number of observations between successive evaluation dates.
    min_trace_ratio
        Minimum Johansen trace-statistic ratio accepted at each fit.
    det_order
        Deterministic-term setting passed to the Johansen test.
    k_ar_diff
        Lagged-difference order passed to the Johansen test.

    Returns
    -------
    pandas.DataFrame
        One row per evaluation date and asset. Columns include the evidence
        score, confidence, future asset return, equal-weight basket return,
        excess return, and binary outperformance outcome.

    Raises
    ------
    ValueError
        If the input dimensions or window settings cannot produce at least one
        walk-forward evaluation.
    """
    if prices.shape[1] < 2:
        raise ValueError("walk-forward calibration requires at least two assets")
    if train_window < max(z_window, k_ar_diff + 4):
        raise ValueError("train_window is too short for the requested model settings")
    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive integers")
    if len(prices) < train_window + horizon:
        raise ValueError("price history is too short for one train/forecast split")

    clean = prices.astype(float).dropna(axis=0, how="any").sort_index()
    records: list[dict[str, object]] = []
    stop = len(clean) - horizon

    for end_position in range(train_window - 1, stop, step):
        train = clean.iloc[end_position - train_window + 1 : end_position + 1]
        evaluation_date = clean.index[end_position]
        future_date = clean.index[end_position + horizon]
        try:
            result = cointegration_evidence(
                train,
                window=z_window,
                det_order=det_order,
                k_ar_diff=k_ar_diff,
                min_trace_ratio=min_trace_ratio,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        initial = clean.iloc[end_position]
        final = clean.iloc[end_position + horizon]
        asset_returns = final / initial - 1.0
        basket_return = float(asset_returns.mean())

        for item in result.asset_evidence:
            asset_return = float(asset_returns.loc[item.ticker])
            excess_return = asset_return - basket_return
            records.append(
                {
                    "evaluation_date": evaluation_date,
                    "future_date": future_date,
                    "ticker": item.ticker,
                    "score": item.score,
                    "confidence": item.confidence,
                    "z_score": item.z_score,
                    "weight": item.weight,
                    "trace_ratio": result.trace_ratio,
                    "asset_return": asset_return,
                    "basket_return": basket_return,
                    "excess_return": excess_return,
                    "outperformed": int(excess_return > 0.0),
                }
            )

    columns = [
        "evaluation_date",
        "future_date",
        "ticker",
        "score",
        "confidence",
        "z_score",
        "weight",
        "trace_ratio",
        "asset_return",
        "basket_return",
        "excess_return",
        "outperformed",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def fit_probability_calibration(
    records: pd.DataFrame,
    *,
    score_edges: Sequence[float] = DEFAULT_SCORE_EDGES,
    horizon: int = 20,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.68,
) -> ProbabilityCalibration:
    """Fit a transparent beta-binomial probability calibration.

    Each evidence-score interval receives a beta posterior for the probability
    that an asset outperforms its equal-weight basket benchmark. A uniform
    ``Beta(1, 1)`` prior is used by default, preventing empty or tiny bins from
    returning exact probabilities of zero or one.

    Parameters
    ----------
    records
        Output from :func:`walk_forward_evidence` containing ``score`` and
        ``outperformed`` columns.
    score_edges
        Monotonically increasing bin edges spanning ``[-1, 1]``.
    horizon
        Forecast horizon represented by ``records``.
    prior_alpha
        Beta-prior success shape parameter.
    prior_beta
        Beta-prior failure shape parameter.
    credible_level
        Probability mass of the central posterior credible interval.

    Returns
    -------
    ProbabilityCalibration
        Fitted binned probability mapping.

    Raises
    ------
    ValueError
        If inputs are invalid or no calibration records are available.
    """
    required = {"score", "outperformed"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"records are missing required columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("records must contain at least one walk-forward outcome")
    edges = np.asarray(score_edges, dtype=float)
    if len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("score_edges must be a strictly increasing sequence")
    if edges[0] > -1.0 or edges[-1] < 1.0:
        raise ValueError("score_edges must span the full interval [-1, 1]")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("beta-prior shape parameters must be positive")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie in (0, 1)")

    scores = pd.to_numeric(records["score"], errors="coerce")
    outcomes = pd.to_numeric(records["outperformed"], errors="coerce")
    valid = scores.notna() & outcomes.isin([0, 1])
    scores = scores[valid].clip(-1.0, 1.0)
    outcomes = outcomes[valid].astype(int)
    if scores.empty:
        raise ValueError("records contain no finite scores and binary outcomes")

    bin_index = np.searchsorted(edges, scores.to_numpy(), side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    tail_probability = (1.0 - credible_level) / 2.0
    rows: list[dict[str, float | int]] = []

    for index in range(len(edges) - 1):
        selected = outcomes.to_numpy()[bin_index == index]
        successes = int(selected.sum())
        count = int(len(selected))
        failures = count - successes
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + failures
        rows.append(
            {
                "score_lower": float(edges[index]),
                "score_upper": float(edges[index + 1]),
                "sample_count": count,
                "successes": successes,
                "probability_mean": float(
                    posterior_alpha / (posterior_alpha + posterior_beta)
                ),
                "probability_lower": float(
                    beta_distribution.ppf(
                        tail_probability, posterior_alpha, posterior_beta
                    )
                ),
                "probability_upper": float(
                    beta_distribution.ppf(
                        1.0 - tail_probability, posterior_alpha, posterior_beta
                    )
                ),
            }
        )

    table = pd.DataFrame(rows)
    return ProbabilityCalibration(
        table=table,
        score_edges=tuple(float(value) for value in edges),
        horizon=int(horizon),
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        credible_level=float(credible_level),
    )


def calibrate_evidence(
    evidence: Iterable[AssetEvidence],
    calibration: ProbabilityCalibration,
) -> tuple[CalibratedAssetEvidence, ...]:
    """Calibrate a collection of current asset evidence records.

    Parameters
    ----------
    evidence
        Current evidence items.
    calibration
        Fitted probability calibration.

    Returns
    -------
    tuple of CalibratedAssetEvidence
        Calibrated records in input order.
    """
    return tuple(calibration.calibrate(item) for item in evidence)


def calibration_table(calibration: ProbabilityCalibration) -> pd.DataFrame:
    """Return a display-friendly copy of a calibration table.

    Parameters
    ----------
    calibration
        Fitted probability calibration.

    Returns
    -------
    pandas.DataFrame
        Calibration-bin statistics.
    """
    return calibration.table.copy()

@dataclass(frozen=True)
class CalibratedRecommendation:
    """Long-only action based on a calibrated outperformance probability.

    Parameters
    ----------
    ticker
        Asset symbol.
    action
        Human-readable long-only action.
    probability_outperform
        Posterior mean probability of outperforming the basket benchmark.
    probability_lower
        Lower posterior credible bound.
    probability_upper
        Upper posterior credible bound.
    sample_count
        Number of historical examples supporting the probability bin.
    currently_held
        Whether the portfolio currently owns the asset.
    explanation
        Human-readable interpretation.
    """

    ticker: str
    action: str
    probability_outperform: float
    probability_lower: float
    probability_upper: float
    sample_count: int
    currently_held: bool
    explanation: str


@dataclass(frozen=True)
class ProbabilityRecommendationPolicy:
    """Decision thresholds for calibrated long-only recommendations.

    Parameters
    ----------
    buy_probability
        Posterior mean probability required for a positive action.
    strong_buy_probability
        Posterior mean probability required for a strong positive action.
    reduce_probability
        Probability below which a held asset may receive a reducing action.
    strong_reduce_probability
        Probability below which a held asset may receive a strong reducing
        action.
    min_samples
        Minimum calibration examples required for a non-neutral action.
    require_interval_excludes_half
        Require the posterior credible interval to lie entirely above or below
        0.5 before issuing a non-neutral action.
    """

    buy_probability: float = 0.60
    strong_buy_probability: float = 0.70
    reduce_probability: float = 0.40
    strong_reduce_probability: float = 0.30
    min_samples: int = 20
    require_interval_excludes_half: bool = True

    def __post_init__(self) -> None:
        """Validate calibrated-decision thresholds.

        Raises
        ------
        ValueError
            If probabilities are inconsistent or ``min_samples`` is negative.
        """
        if not 0.5 < self.buy_probability < self.strong_buy_probability <= 1.0:
            raise ValueError(
                "positive thresholds must satisfy 0.5 < buy < strong_buy <= 1"
            )
        if not 0.0 <= self.strong_reduce_probability < self.reduce_probability < 0.5:
            raise ValueError(
                "negative thresholds must satisfy 0 <= strong_reduce < reduce < 0.5"
            )
        if self.min_samples < 0:
            raise ValueError("min_samples must be non-negative")

    def classify(
        self,
        item: CalibratedAssetEvidence,
        *,
        currently_held: bool,
    ) -> CalibratedRecommendation:
        """Classify one calibrated evidence record.

        Parameters
        ----------
        item
            Calibrated asset evidence.
        currently_held
            Whether the asset is currently owned.

        Returns
        -------
        CalibratedRecommendation
            Portfolio-aware action and probability explanation.
        """
        probability = item.probability_outperform
        sufficiently_sampled = item.sample_count >= self.min_samples
        positive_interval = item.probability_lower > 0.5
        negative_interval = item.probability_upper < 0.5
        positive_allowed = sufficiently_sampled and (
            positive_interval or not self.require_interval_excludes_half
        )
        negative_allowed = sufficiently_sampled and (
            negative_interval or not self.require_interval_excludes_half
        )

        if positive_allowed and probability >= self.strong_buy_probability:
            action = "Strong add" if currently_held else "Strong buy"
        elif positive_allowed and probability >= self.buy_probability:
            action = "Add" if currently_held else "Buy"
        elif negative_allowed and probability <= self.strong_reduce_probability:
            action = "Consider reducing" if currently_held else "Avoid buying"
        elif negative_allowed and probability <= self.reduce_probability:
            action = "Hold without adding" if currently_held else "Wait"
        else:
            action = "Hold" if currently_held else "Watch"

        explanation = (
            f"Estimated probability of outperforming the {item.benchmark} over "
            f"{item.horizon} trading observations is {probability:.1%} "
            f"({item.probability_lower:.1%} to {item.probability_upper:.1%}); "
            f"calibration bin contains {item.sample_count} historical examples."
        )
        return CalibratedRecommendation(
            ticker=item.ticker,
            action=action,
            probability_outperform=probability,
            probability_lower=item.probability_lower,
            probability_upper=item.probability_upper,
            sample_count=item.sample_count,
            currently_held=currently_held,
            explanation=explanation,
        )


def recommend_calibrated_assets(
    evidence: Iterable[CalibratedAssetEvidence],
    *,
    holdings: dict[str, float] | None = None,
    policy: ProbabilityRecommendationPolicy | None = None,
) -> tuple[CalibratedRecommendation, ...]:
    """Generate long-only actions from calibrated evidence.

    Parameters
    ----------
    evidence
        Probability-calibrated evidence items.
    holdings
        Mapping from ticker to quantity currently owned.
    policy
        Calibrated recommendation policy.

    Returns
    -------
    tuple of CalibratedRecommendation
        Recommendations sorted by decreasing outperformance probability.
    """
    holdings = {str(key).upper(): float(value) for key, value in (holdings or {}).items()}
    policy = policy or ProbabilityRecommendationPolicy()
    recommendations = [
        policy.classify(item, currently_held=holdings.get(item.ticker, 0.0) > 0.0)
        for item in evidence
    ]
    recommendations.sort(key=lambda item: item.probability_outperform, reverse=True)
    return tuple(recommendations)


def calibrated_recommendation_table(
    recommendations: Iterable[CalibratedRecommendation],
) -> pd.DataFrame:
    """Convert calibrated recommendations into a display table.

    Parameters
    ----------
    recommendations
        Calibrated recommendation records.

    Returns
    -------
    pandas.DataFrame
        Recommendations ordered by outperformance probability.
    """
    rows = [
        {
            "ticker": item.ticker,
            "action": item.action,
            "probability_outperform": item.probability_outperform,
            "probability_lower": item.probability_lower,
            "probability_upper": item.probability_upper,
            "sample_count": item.sample_count,
            "currently_held": item.currently_held,
            "explanation": item.explanation,
        }
        for item in recommendations
    ]
    return pd.DataFrame(rows).sort_values(
        "probability_outperform", ascending=False, ignore_index=True
    )
