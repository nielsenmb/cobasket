"""Regression tests for shared basket quality thresholds."""

from __future__ import annotations

import inspect

from cobasket.basket_calibration import fit_basket_calibrations
from cobasket.basket_validation import BasketValidationThresholds
from cobasket.discovery import discover_baskets
from cobasket.thresholds import MIN_ACCEPTED_EVALUATIONS


def test_default_minimum_evaluations_is_shared() -> None:
    """Discovery, validation, and calibration should use one common default."""
    discovery_default = inspect.signature(discover_baskets).parameters["promising_evaluations"].default
    calibration_default = inspect.signature(fit_basket_calibrations).parameters["min_evaluations"].default

    assert MIN_ACCEPTED_EVALUATIONS == 15
    assert BasketValidationThresholds().min_evaluations == MIN_ACCEPTED_EVALUATIONS
    assert discovery_default == MIN_ACCEPTED_EVALUATIONS
    assert calibration_default == MIN_ACCEPTED_EVALUATIONS
