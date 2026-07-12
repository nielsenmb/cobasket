"""Trading signal generation from a spread series."""

import numpy as np
import pandas as pd


def zscore_signal(spread, window=30, entry_z=2.0, exit_z=0.5):
    """
    Rolling z-score of the spread, converted into a discrete +1/0/-1 position.
    +1: spread is far below its rolling mean -> expect reversion up (long)
    -1: spread is far above its rolling mean -> expect reversion down (short)
     0: within exit_z of the mean -> flatten
    Between entry_z and exit_z, hold the existing position.
    """
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = (spread - mu) / sigma

    signal = pd.Series(0, index=spread.index)
    signal[z > entry_z] = -1
    signal[z < -entry_z] = 1
    signal[z.abs() < exit_z] = 0
    signal = signal.replace(0, np.nan).ffill().fillna(0)  # hold position until exit
    return z, signal
