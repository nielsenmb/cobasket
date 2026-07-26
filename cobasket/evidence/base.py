"""Common data structures for transparent investment evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetEvidence:
    """Evidence concerning the relative attractiveness of one asset.

    Parameters
    ----------
    ticker
        Asset symbol.
    score
        Signed evidence score in the closed interval ``[-1, 1]``. Positive
        values favour buying or adding, while negative values favour avoiding
        or reducing the asset. This is not yet a calibrated probability.
    confidence
        Strength of the available evidence in the closed interval ``[0, 1]``.
        A high confidence indicates a strong and statistically supported
        relative displacement, not certainty of a profitable outcome.
    source
        Name of the evidence model.
    summary
        Human-readable explanation of the score.
    z_score
        Latest standardized basket displacement, when applicable.
    weight
        Asset coefficient in the normalized basket relation, when applicable.
    """

    ticker: str
    score: float
    confidence: float
    source: str
    summary: str
    z_score: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize evidence fields after construction.

        Raises
        ------
        ValueError
            If ``score`` or ``confidence`` lies outside its allowed interval.
        """
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("score must lie in [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        object.__setattr__(self, "ticker", ticker)
