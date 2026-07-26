"""Tests for GUI-facing report conversion and formatting helpers."""

from cobasket.gui.models import (
    portfolio_report_from_dict,
    probability_label,
    recommendation_priority,
)


def _payload():
    """Return a representative serialized portfolio report."""
    return {
        "generated_at_utc": "2026-07-26T12:00:00+00:00",
        "latest_price_date": "2026-07-25T00:00:00",
        "cash": 1000.0,
        "invested_value": 250.0,
        "total_value": 1250.0,
        "tickers": [
            {
                "ticker": "AAPL",
                "held_quantity": 1.0,
                "current_price": 250.0,
                "market_value": 250.0,
                "evidence_score": 0.7,
                "evidence_confidence": 0.8,
                "probability_outperform": 0.68,
                "probability_lower": 0.55,
                "probability_upper": 0.79,
                "calibration_sample_count": 42,
                "recommendation": "Add",
                "explanation": "Relative evidence is positive.",
                "basket_memberships": [["AAPL", "MSFT"]],
                "warnings": ["One basket only."],
            }
        ],
        "basket_diagnostics": [{"basket": "AAPL, MSFT", "trace_ratio": 1.4}],
        "warnings": ["Example warning."],
        "metadata": {"calibrated": True},
    }


def test_portfolio_report_from_dict_round_trip_fields():
    """Serialized reports should reconstruct nested immutable records."""
    report = portfolio_report_from_dict(_payload())
    assert report.total_value == 1250.0
    assert report.tickers[0].ticker == "AAPL"
    assert report.tickers[0].basket_memberships == (("AAPL", "MSFT"),)
    assert report.tickers[0].warnings == ("One basket only.",)


def test_probability_label_includes_interval():
    """Calibrated records should display mean and interval percentages."""
    report = portfolio_report_from_dict(_payload())
    assert probability_label(report.tickers[0]) == "68% [55%, 79%]"


def test_probability_label_handles_uncalibrated_record():
    """Missing calibration should be labelled explicitly."""
    payload = _payload()
    payload["tickers"][0]["probability_outperform"] = None
    payload["tickers"][0]["probability_lower"] = None
    payload["tickers"][0]["probability_upper"] = None
    report = portfolio_report_from_dict(payload)
    assert probability_label(report.tickers[0]) == "Uncalibrated"


def test_recommendation_priority_orders_buy_above_reduce():
    """Display priority should preserve the intended recommendation ordering."""
    assert recommendation_priority("Strong buy") > recommendation_priority("Hold")
    assert recommendation_priority("Hold") > recommendation_priority("Consider reducing")
