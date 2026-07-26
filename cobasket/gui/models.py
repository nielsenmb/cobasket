"""Pure display helpers shared by the Cobasket PyQt dashboard."""

from __future__ import annotations

from typing import Any, Mapping

from cobasket.workflow import PortfolioReport, TickerReport


def portfolio_report_from_dict(payload: Mapping[str, Any]) -> PortfolioReport:
    """Reconstruct a :class:`PortfolioReport` from JSON-compatible data.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`cobasket.workflow.PortfolioReport.to_dict`.

    Returns
    -------
    PortfolioReport
        Reconstructed immutable report object.
    """
    tickers = tuple(
        TickerReport(
            ticker=str(item["ticker"]),
            held_quantity=float(item["held_quantity"]),
            current_price=float(item["current_price"]),
            market_value=float(item["market_value"]),
            evidence_score=float(item["evidence_score"]),
            evidence_confidence=float(item["evidence_confidence"]),
            probability_outperform=(
                None
                if item.get("probability_outperform") is None
                else float(item["probability_outperform"])
            ),
            probability_lower=(
                None
                if item.get("probability_lower") is None
                else float(item["probability_lower"])
            ),
            probability_upper=(
                None
                if item.get("probability_upper") is None
                else float(item["probability_upper"])
            ),
            calibration_sample_count=(
                None
                if item.get("calibration_sample_count") is None
                else int(item["calibration_sample_count"])
            ),
            recommendation=str(item["recommendation"]),
            explanation=str(item["explanation"]),
            basket_memberships=tuple(
                tuple(str(ticker) for ticker in basket)
                for basket in item.get("basket_memberships", ())
            ),
            warnings=tuple(str(warning) for warning in item.get("warnings", ())),
        )
        for item in payload.get("tickers", ())
    )
    return PortfolioReport(
        generated_at_utc=str(payload["generated_at_utc"]),
        latest_price_date=str(payload["latest_price_date"]),
        cash=float(payload["cash"]),
        invested_value=float(payload["invested_value"]),
        total_value=float(payload["total_value"]),
        tickers=tickers,
        basket_diagnostics=tuple(dict(item) for item in payload.get("basket_diagnostics", ())),
        warnings=tuple(str(warning) for warning in payload.get("warnings", ())),
        metadata=dict(payload.get("metadata", {})),
    )


def probability_label(item: TickerReport) -> str:
    """Format one ticker's calibrated probability and interval.

    Parameters
    ----------
    item
        Ticker report to format.

    Returns
    -------
    str
        Percentage and credible interval, or ``"Uncalibrated"``.
    """
    if item.probability_outperform is None:
        return "Uncalibrated"
    if item.probability_lower is None or item.probability_upper is None:
        return f"{item.probability_outperform:.0%}"
    return (
        f"{item.probability_outperform:.0%} "
        f"[{item.probability_lower:.0%}, {item.probability_upper:.0%}]"
    )


def recommendation_priority(action: str) -> int:
    """Return a stable display priority for a recommendation label.

    Parameters
    ----------
    action
        Human-readable recommendation.

    Returns
    -------
    int
        Larger values indicate stronger positive recommendations.
    """
    normalized = action.strip().lower()
    ordered = {
        "strong buy": 6,
        "strong add": 6,
        "buy": 5,
        "add": 5,
        "watch": 4,
        "hold": 4,
        "wait": 3,
        "hold without adding": 3,
        "avoid buying": 2,
        "consider reducing": 1,
        "sell": 0,
    }
    return ordered.get(normalized, 3)
