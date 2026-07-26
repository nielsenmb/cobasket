"""JSON persistence helpers for probability calibration objects."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .calibration import ProbabilityCalibration


def save_probability_calibration(
    calibration: ProbabilityCalibration,
    path: str | Path,
) -> Path:
    """Write a fitted probability calibration to JSON.

    Parameters
    ----------
    calibration
        Fitted binned beta-binomial calibration.
    path
        Destination JSON path.

    Returns
    -------
    pathlib.Path
        Written path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "table": calibration.table.to_dict(orient="records"),
        "score_edges": list(calibration.score_edges),
        "horizon": calibration.horizon,
        "prior_alpha": calibration.prior_alpha,
        "prior_beta": calibration.prior_beta,
        "credible_level": calibration.credible_level,
        "benchmark": calibration.benchmark,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_probability_calibration(path: str | Path) -> ProbabilityCalibration:
    """Load a fitted probability calibration from JSON.

    Parameters
    ----------
    path
        Existing calibration JSON path.

    Returns
    -------
    ProbabilityCalibration
        Reconstructed calibration object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProbabilityCalibration(
        table=pd.DataFrame(payload["table"]),
        score_edges=tuple(float(value) for value in payload["score_edges"]),
        horizon=int(payload["horizon"]),
        prior_alpha=float(payload.get("prior_alpha", 1.0)),
        prior_beta=float(payload.get("prior_beta", 1.0)),
        credible_level=float(payload.get("credible_level", 0.68)),
        benchmark=str(payload.get("benchmark", "equal-weight basket")),
    )


def _save_method(self: ProbabilityCalibration, path: str | Path) -> Path:
    """Method adapter for :func:`save_probability_calibration`."""
    return save_probability_calibration(self, path)


@classmethod
def _load_method(cls, path: str | Path) -> ProbabilityCalibration:
    """Class-method adapter for :func:`load_probability_calibration`."""
    return load_probability_calibration(path)


ProbabilityCalibration.save = _save_method
ProbabilityCalibration.load = _load_method
