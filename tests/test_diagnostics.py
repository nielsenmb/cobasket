import numpy as np
import pandas as pd

from cobasket.evidence import calibration_diagnostics, reliability_table


def test_perfect_forecasts_have_zero_brier_and_ece():
    result = calibration_diagnostics([0, 1, 0, 1], [0, 1, 0, 1], n_bins=2)
    assert result.brier_score == 0
    assert result.expected_calibration_error == 0


def test_reliability_table_preserves_all_valid_samples():
    table = reliability_table([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=4)
    assert table["sample_count"].sum() == 4


def test_diagnostics_reject_invalid_only_input():
    import pytest
    with pytest.raises(ValueError):
        calibration_diagnostics([np.nan], [1])
