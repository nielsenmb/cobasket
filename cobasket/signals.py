"""Trading-signal generation from a real-valued spread series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_real_series(spread: pd.Series) -> pd.Series:
    """Return a finite-real-compatible copy of a spread series.

    Parameters
    ----------
    spread
        Candidate spread values.

    Returns
    -------
    pandas.Series
        Floating-point spread with the original index and name.

    Raises
    ------
    TypeError
        If ``spread`` is not a pandas Series.
    ValueError
        If values contain a material imaginary component.
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series")

    values = np.real_if_close(spread.to_numpy(), tol=1000)
    if np.iscomplexobj(values):
        max_imag = float(np.nanmax(np.abs(np.imag(values))))
        raise ValueError(
            "spread contains a material imaginary component "
            f"(maximum |imaginary|={max_imag:.3e})"
        )
    return pd.Series(
        np.asarray(values, dtype=float),
        index=spread.index,
        name=spread.name,
    )


def zscore_signal(
    spread: pd.Series,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Generate a stateful mean-reversion position from a spread z-score.

    A positive position means *long the spread*: hold the asset legs in the
    direction specified by the spread weights. A negative position reverses
    every leg and is therefore *short the spread*. A zero position means no
    exposure. The state is retained until the z-score returns inside the exit
    threshold.

    Parameters
    ----------
    spread
        Real-valued spread time series.
    window
        Rolling number of observations used to estimate the local mean and
        standard deviation.
    entry_z
        Absolute z-score at which a flat strategy opens a position.
    exit_z
        Absolute z-score below which an open position is closed.

    Returns
    -------
    z_score
        Rolling z-score of the spread.
    signal
        Position state: ``+1`` long spread, ``-1`` short spread, and ``0`` flat.

    Raises
    ------
    ValueError
        If thresholds or the rolling window are invalid, or if the spread has a
        material imaginary component.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if entry_z <= 0:
        raise ValueError("entry_z must be positive")
    if exit_z < 0 or exit_z >= entry_z:
        raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z")

    real_spread = _as_real_series(spread)
    rolling_mean = real_spread.rolling(window).mean()
    rolling_std = real_spread.rolling(window).std()
    displacement = real_spread - rolling_mean
    z_score = displacement / rolling_std.replace(0.0, np.nan)
    # A perfectly constant rolling window has zero displacement and zero
    # variance. Its physically meaningful standardized displacement is zero,
    # which also permits an open position to exit.
    z_score = z_score.mask((rolling_std == 0.0) & (displacement == 0.0), 0.0)
    z_score = z_score.rename("z_score")

    positions: list[int] = []
    current = 0
    for value in z_score:
        if pd.isna(value):
            positions.append(current)
            continue
        if current == 0:
            if value > entry_z:
                current = -1
            elif value < -entry_z:
                current = 1
        elif abs(value) < exit_z:
            current = 0
        positions.append(current)

    signal = pd.Series(positions, index=real_spread.index, name="signal", dtype=int)
    return z_score, signal
