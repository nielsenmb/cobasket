"""Persistent historical validation profiles for monitored baskets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cobasket.config_paths import resolve_portfolio_config_paths
from cobasket.data import DataManager
from cobasket.evidence import BasketWatchlist, cointegration_evidence, walk_forward_evidence
from cobasket.workflow import PortfolioConfig


@dataclass(frozen=True)
class BasketValidationThresholds:
    """Transparent thresholds used to classify one basket.

    Parameters
    ----------
    min_current_trace_ratio
        Minimum current Johansen trace ratio required for a usable basket.
    min_evaluations
        Minimum accepted historical evaluation dates for ``validated`` status.
    min_acceptance_rate
        Minimum fraction of possible historical evaluation dates that pass the
        Johansen requirement.
    min_weight_stability
        Minimum mean absolute cosine similarity of historical weight vectors.
    min_score_return_correlation
        Minimum Spearman correlation between evidence score and future excess
        return for ``validated`` status.
    min_calibration_contrast
        Minimum difference between positive- and negative-evidence historical
        outperformance rates.
    """

    min_current_trace_ratio: float = 1.0
    min_evaluations: int = 20
    min_acceptance_rate: float = 0.15
    min_weight_stability: float = 0.60
    min_score_return_correlation: float = 0.05
    min_calibration_contrast: float = 0.05


@dataclass(frozen=True)
class BasketValidationProfile:
    """Historical reliability summary for one monitored basket."""

    basket: tuple[str, ...]
    status: str
    current_trace_ratio: float | None
    accepted_evaluations: int
    possible_evaluations: int
    acceptance_rate: float
    weight_stability: float | None
    score_return_correlation: float | None
    positive_outperform_rate: float | None
    negative_outperform_rate: float | None
    calibration_contrast: float | None
    records: int
    reasons: tuple[str, ...]

    @property
    def key(self) -> str:
        """Return a stable comma-separated basket identifier."""
        return ", ".join(self.basket)


@dataclass(frozen=True)
class BasketValidationSet:
    """Serializable collection of basket validation profiles."""

    generated_at_utc: str
    train_window: int
    z_window: int
    horizon: int
    step: int
    min_trace_ratio: float
    profiles: tuple[BasketValidationProfile, ...]

    def save(self, path: str | Path) -> Path:
        """Write validation profiles to JSON.

        Parameters
        ----------
        path
            Destination JSON path.

        Returns
        -------
        pathlib.Path
            Written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "BasketValidationSet":
        """Load validation profiles from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["profiles"] = tuple(
            BasketValidationProfile(
                **{
                    **item,
                    "basket": tuple(item["basket"]),
                    "reasons": tuple(item.get("reasons", ())),
                }
            )
            for item in payload["profiles"]
        )
        return cls(**payload)

    def by_key(self) -> dict[str, BasketValidationProfile]:
        """Return profiles keyed by comma-separated basket membership."""
        return {profile.key: profile for profile in self.profiles}


def _possible_evaluations(n_observations: int, train_window: int, horizon: int, step: int) -> int:
    """Return the number of nominal walk-forward evaluation dates."""
    stop = n_observations - horizon
    if stop <= train_window - 1:
        return 0
    return len(range(train_window - 1, stop, step))


def _weight_stability(records: pd.DataFrame, basket: tuple[str, ...]) -> float | None:
    """Measure sign-invariant similarity of historical Johansen weight vectors."""
    vectors: list[np.ndarray] = []
    for _, group in records.groupby("evaluation_date"):
        weights = group.set_index("ticker")["weight"].reindex(basket)
        if weights.isna().any():
            continue
        vector = weights.to_numpy(dtype=float)
        norm = np.linalg.norm(vector)
        if norm > np.finfo(float).eps:
            vectors.append(vector / norm)
    if len(vectors) < 2:
        return None
    reference = vectors[0]
    similarities = [abs(float(np.dot(reference, vector))) for vector in vectors[1:]]
    return float(np.mean(similarities)) if similarities else None


def _finite_or_none(value: float) -> float | None:
    """Convert a finite scalar to float, otherwise return ``None``."""
    return float(value) if np.isfinite(value) else None


def _classify_profile(
    *,
    current_trace_ratio: float | None,
    accepted_evaluations: int,
    acceptance_rate: float,
    weight_stability: float | None,
    score_return_correlation: float | None,
    calibration_contrast: float | None,
    thresholds: BasketValidationThresholds,
) -> tuple[str, tuple[str, ...]]:
    """Classify a basket using transparent ordered rules."""
    reasons: list[str] = []
    if current_trace_ratio is None or current_trace_ratio < thresholds.min_current_trace_ratio:
        reasons.append("current cointegration is below the required threshold")
        return "rejected", tuple(reasons)
    if accepted_evaluations == 0:
        reasons.append("no accepted historical walk-forward evaluations")
        return "rejected", tuple(reasons)
    if acceptance_rate < thresholds.min_acceptance_rate:
        reasons.append("historical cointegration acceptance rate is low")
    if weight_stability is None or weight_stability < thresholds.min_weight_stability:
        reasons.append("historical Johansen weights are insufficiently stable")
    if reasons:
        return "unstable", tuple(reasons)

    if accepted_evaluations < thresholds.min_evaluations:
        reasons.append("too few accepted historical evaluation dates")
    if score_return_correlation is None or score_return_correlation < thresholds.min_score_return_correlation:
        reasons.append("evidence score has weak historical rank correlation with future excess return")
    if calibration_contrast is None or calibration_contrast < thresholds.min_calibration_contrast:
        reasons.append("positive evidence has not clearly outperformed negative evidence historically")
    if reasons:
        return "weak", tuple(reasons)
    return "validated", ("current and historical validation criteria passed",)


def validate_watchlist_baskets(
    config_path: str | Path,
    *,
    train_window: int = 252,
    z_window: int | None = None,
    horizon: int = 20,
    step: int = 20,
    min_trace_ratio: float | None = None,
    thresholds: BasketValidationThresholds | None = None,
    force_refresh: bool = False,
    data_manager: DataManager | None = None,
) -> BasketValidationSet:
    """Build persistent reliability profiles for every watchlist basket.

    Parameters
    ----------
    config_path
        Portfolio configuration JSON.
    train_window
        Trailing observations used for each historical Johansen fit.
    z_window
        Spread standardization window. Defaults to the portfolio setting.
    horizon
        Forward relative-performance horizon in trading observations.
    step
        Spacing between historical evaluations. The default equals ``horizon``
        so outcome windows do not overlap.
    min_trace_ratio
        Historical Johansen threshold. Defaults to the portfolio setting.
    thresholds
        Status-classification thresholds.
    force_refresh
        Whether to bypass reusable price-cache files.
    data_manager
        Optional data manager, mainly for testing.

    Returns
    -------
    BasketValidationSet
        Persistent validation result for the whole watchlist.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = resolve_portfolio_config_paths(PortfolioConfig.load(config_path), config_path)
    watchlist = BasketWatchlist.load(config.watchlist_path)
    manager = data_manager or DataManager()
    z_window = config.z_window if z_window is None else int(z_window)
    min_trace_ratio = config.min_trace_ratio if min_trace_ratio is None else float(min_trace_ratio)
    thresholds = thresholds or BasketValidationThresholds(min_current_trace_ratio=min_trace_ratio)
    prices = manager.prices(
        watchlist.tickers,
        period=config.period,
        force_refresh=force_refresh,
        min_coverage=1.0,
    )

    profiles: list[BasketValidationProfile] = []
    for raw_basket in watchlist.baskets:
        basket = tuple(raw_basket)
        basket_prices = prices.loc[:, list(basket)].dropna()
        possible = _possible_evaluations(len(basket_prices), train_window, horizon, step)
        try:
            current = cointegration_evidence(basket_prices, window=z_window, min_trace_ratio=0.0)
            current_trace_ratio = float(current.trace_ratio)
        except (ValueError, np.linalg.LinAlgError):
            current_trace_ratio = None

        try:
            records = walk_forward_evidence(
                basket_prices,
                train_window=train_window,
                z_window=z_window,
                horizon=horizon,
                step=step,
                min_trace_ratio=min_trace_ratio,
            )
        except (ValueError, np.linalg.LinAlgError):
            records = pd.DataFrame()

        accepted = int(records["evaluation_date"].nunique()) if not records.empty else 0
        acceptance_rate = accepted / possible if possible else 0.0
        stability = _weight_stability(records, basket) if not records.empty else None
        if len(records) >= 3 and records["score"].nunique() > 1 and records["excess_return"].nunique() > 1:
            correlation = _finite_or_none(spearmanr(records["score"], records["excess_return"]).statistic)
        else:
            correlation = None
        positive = records.loc[records["score"] >= 0.25, "outperformed"] if not records.empty else pd.Series(dtype=float)
        negative = records.loc[records["score"] <= -0.25, "outperformed"] if not records.empty else pd.Series(dtype=float)
        positive_rate = float(positive.mean()) if not positive.empty else None
        negative_rate = float(negative.mean()) if not negative.empty else None
        contrast = (
            positive_rate - negative_rate
            if positive_rate is not None and negative_rate is not None
            else None
        )
        status, reasons = _classify_profile(
            current_trace_ratio=current_trace_ratio,
            accepted_evaluations=accepted,
            acceptance_rate=acceptance_rate,
            weight_stability=stability,
            score_return_correlation=correlation,
            calibration_contrast=contrast,
            thresholds=thresholds,
        )
        profiles.append(
            BasketValidationProfile(
                basket=basket,
                status=status,
                current_trace_ratio=current_trace_ratio,
                accepted_evaluations=accepted,
                possible_evaluations=possible,
                acceptance_rate=float(acceptance_rate),
                weight_stability=stability,
                score_return_correlation=correlation,
                positive_outperform_rate=positive_rate,
                negative_outperform_rate=negative_rate,
                calibration_contrast=contrast,
                records=int(len(records)),
                reasons=reasons,
            )
        )

    return BasketValidationSet(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        train_window=int(train_window),
        z_window=int(z_window),
        horizon=int(horizon),
        step=int(step),
        min_trace_ratio=float(min_trace_ratio),
        profiles=tuple(profiles),
    )


def validation_table(validation: BasketValidationSet) -> pd.DataFrame:
    """Return a compact display table for validation profiles."""
    rows = []
    for profile in validation.profiles:
        row = asdict(profile)
        row["basket"] = profile.key
        row["reasons"] = "; ".join(profile.reasons)
        rows.append(row)
    return pd.DataFrame(rows)
