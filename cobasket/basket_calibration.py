"""Basket-specific probability calibration for validated relationships."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from cobasket.basket_validation import BasketValidationSet
from cobasket.calibration_workflow import calibrate_watchlist
from cobasket.evidence import ProbabilityCalibration, fit_probability_calibration


@dataclass(frozen=True)
class BasketProbabilityCalibration:
    """Probability calibration for one validated basket.

    Parameters
    ----------
    basket
        Tickers defining the basket.
    accepted_evaluations
        Number of independent historical evaluation dates supporting the fit.
    calibration
        Evidence-score to relative-outperformance probability mapping.
    """

    basket: tuple[str, ...]
    accepted_evaluations: int
    calibration: ProbabilityCalibration

    def to_dict(self) -> dict[str, object]:
        """Convert the calibration to JSON-compatible objects.

        Returns
        -------
        dict
            Serialized basket calibration.
        """
        return {
            "basket": list(self.basket),
            "accepted_evaluations": int(self.accepted_evaluations),
            "calibration": {
                "table": self.calibration.table.to_dict(orient="records"),
                "score_edges": list(self.calibration.score_edges),
                "horizon": self.calibration.horizon,
                "prior_alpha": self.calibration.prior_alpha,
                "prior_beta": self.calibration.prior_beta,
                "credible_level": self.calibration.credible_level,
                "benchmark": self.calibration.benchmark,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BasketProbabilityCalibration":
        """Reconstruct a basket calibration from serialized data.

        Parameters
        ----------
        payload
            Serialized calibration mapping.

        Returns
        -------
        BasketProbabilityCalibration
            Reconstructed calibration.
        """
        raw = dict(payload["calibration"])
        calibration = ProbabilityCalibration(
            table=pd.DataFrame(raw["table"]),
            score_edges=tuple(float(value) for value in raw["score_edges"]),
            horizon=int(raw["horizon"]),
            prior_alpha=float(raw.get("prior_alpha", 1.0)),
            prior_beta=float(raw.get("prior_beta", 1.0)),
            credible_level=float(raw.get("credible_level", 0.68)),
            benchmark=str(raw.get("benchmark", "equal-weight basket")),
        )
        return cls(
            basket=tuple(str(item).upper() for item in payload["basket"]),
            accepted_evaluations=int(payload["accepted_evaluations"]),
            calibration=calibration,
        )


@dataclass(frozen=True)
class BasketCalibrationSet:
    """Persistent basket-specific calibrations used by live reports.

    Parameters
    ----------
    generated_at_utc
        Timestamp inherited from the validation set used for selection.
    min_evaluations
        Minimum independent historical evaluations required for a basket fit.
    calibrations
        One calibration per eligible basket.
    skipped
        Mapping from skipped basket key to the reason it was not calibrated.
    """

    generated_at_utc: str
    min_evaluations: int
    calibrations: tuple[BasketProbabilityCalibration, ...]
    skipped: Mapping[str, str]

    def by_key(self) -> dict[str, BasketProbabilityCalibration]:
        """Return calibrations keyed by normalized basket membership.

        Returns
        -------
        dict
            Mapping from comma-separated basket key to calibration.
        """
        return {", ".join(item.basket): item for item in self.calibrations}

    def save(self, path: str | Path) -> Path:
        """Save basket calibrations to JSON.

        Parameters
        ----------
        path
            Output JSON path.

        Returns
        -------
        pathlib.Path
            Written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": self.generated_at_utc,
            "min_evaluations": self.min_evaluations,
            "calibrations": [item.to_dict() for item in self.calibrations],
            "skipped": dict(self.skipped),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "BasketCalibrationSet":
        """Load basket calibrations from JSON.

        Parameters
        ----------
        path
            Existing JSON path.

        Returns
        -------
        BasketCalibrationSet
            Loaded calibration collection.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            generated_at_utc=str(payload["generated_at_utc"]),
            min_evaluations=int(payload["min_evaluations"]),
            calibrations=tuple(
                BasketProbabilityCalibration.from_dict(item)
                for item in payload.get("calibrations", [])
            ),
            skipped={str(key): str(value) for key, value in payload.get("skipped", {}).items()},
        )


def fit_basket_calibrations(
    portfolio_path: str | Path,
    validation_path: str | Path,
    *,
    min_evaluations: int = 20,
    train_window: int = 252,
    z_window: int | None = None,
    horizon: int = 20,
    step: int = 20,
    score_edges: Sequence[float] = (-1.0, -0.60, -0.25, 0.25, 0.60, 1.0),
    force_refresh: bool = False,
) -> BasketCalibrationSet:
    """Fit independent probability calibrations for validated baskets.

    Only baskets whose validation profile has status ``validated`` and at least
    ``min_evaluations`` accepted non-overlapping historical evaluation dates are
    calibrated. Other baskets are retained in ``skipped`` with an explicit
    reason.

    Parameters
    ----------
    portfolio_path
        Portfolio configuration used to obtain the watchlist and price period.
    validation_path
        Persistent basket validation file.
    min_evaluations
        Minimum accepted independent evaluation dates required for calibration.
    train_window
        Trailing observations used by each historical relation fit.
    z_window
        Spread z-score window. Defaults to the portfolio configuration.
    horizon
        Forward relative-return horizon in trading observations.
    step
        Spacing between historical evaluations. Defaults to ``horizon`` to avoid
        overlapping outcomes.
    score_edges
        Evidence-score bin edges.
    force_refresh
        Whether to bypass valid market-data cache files.

    Returns
    -------
    BasketCalibrationSet
        Calibrations for eligible baskets plus explicit skip reasons.

    Raises
    ------
    ValueError
        If ``min_evaluations`` is smaller than one.
    """
    if min_evaluations < 1:
        raise ValueError("min_evaluations must be at least one")

    validation = BasketValidationSet.load(validation_path)
    result = calibrate_watchlist(
        portfolio_path,
        train_window=train_window,
        z_window=z_window,
        horizon=horizon,
        step=step,
        force_refresh=force_refresh,
    )
    profiles = validation.by_key()
    calibrations: list[BasketProbabilityCalibration] = []
    skipped: dict[str, str] = {}

    for basket_key, group in result.records.groupby("basket"):
        key = str(basket_key)
        profile = profiles.get(key)
        if profile is None:
            skipped[key] = "no matching basket validation profile"
            continue
        if profile.status != "validated":
            skipped[key] = f"validation status is {profile.status}"
            continue
        evaluations = int(group["evaluation_date"].nunique())
        if evaluations < min_evaluations:
            skipped[key] = f"only {evaluations} independent evaluations; requires {min_evaluations}"
            continue
        calibration = fit_probability_calibration(group, score_edges=score_edges, horizon=horizon)
        calibrations.append(
            BasketProbabilityCalibration(
                basket=tuple(part.strip().upper() for part in key.split(",")),
                accepted_evaluations=evaluations,
                calibration=calibration,
            )
        )

    for key, profile in profiles.items():
        if key not in skipped and key not in {", ".join(item.basket) for item in calibrations}:
            skipped[key] = f"validation status is {profile.status}"

    return BasketCalibrationSet(
        generated_at_utc=validation.generated_at_utc,
        min_evaluations=min_evaluations,
        calibrations=tuple(calibrations),
        skipped=skipped,
    )
