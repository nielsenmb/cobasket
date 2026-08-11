"""End-to-end portfolio reporting for Cobasket decision support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cobasket.data import DataManager
from cobasket.evidence import (
    BasketWatchlist,
    ProbabilityCalibration,
    ProbabilityRecommendationPolicy,
    RecommendationPolicy,
    evaluate_watchlist,
)


@dataclass(frozen=True)
class PortfolioConfig:
    """Persistent inputs for a live portfolio analysis.

    Parameters
    ----------
    holdings
        Mapping from ticker symbol to currently owned quantity.
    cash
        Uninvested cash recorded for portfolio context.
    watchlist_path
        JSON file containing a monitored basket watchlist.
    calibration_path
        Optional global probability-calibration JSON.
    validation_path
        Optional basket-validation-profile JSON. When supplied, live actions are
        gated so baskets that are not historically validated cannot generate a
        positive or reducing recommendation by themselves.
    period
        Historical period requested from the data provider.
    z_window
        Rolling window used to standardize each current basket spread.
    min_trace_ratio
        Minimum current Johansen trace-statistic ratio accepted for a basket.
    max_price_age_days
        Maximum acceptable age of the latest price before a stale-data warning.
    """

    holdings: Mapping[str, float]
    cash: float = 0.0
    watchlist_path: str = "portfolio_watchlist.json"
    calibration_path: str | None = None
    validation_path: str | None = None
    period: str = "3y"
    z_window: int = 60
    min_trace_ratio: float = 1.0
    max_price_age_days: float = 7.0

    def __post_init__(self) -> None:
        """Normalize portfolio fields and validate configuration values."""
        holdings = {
            str(ticker).strip().upper(): float(quantity)
            for ticker, quantity in self.holdings.items()
        }
        if any(not ticker for ticker in holdings):
            raise ValueError("holding ticker symbols must not be empty")
        if any(quantity < 0.0 for quantity in holdings.values()):
            raise ValueError("holding quantities must be non-negative")
        if self.cash < 0.0:
            raise ValueError("cash must be non-negative")
        if self.z_window < 2:
            raise ValueError("z_window must be at least two")
        if self.min_trace_ratio <= 0.0:
            raise ValueError("min_trace_ratio must be positive")
        if self.max_price_age_days < 0.0:
            raise ValueError("max_price_age_days must be non-negative")
        object.__setattr__(self, "holdings", holdings)

    def save(self, path: str | Path) -> Path:
        """Write the configuration to human-readable JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioConfig":
        """Load a portfolio configuration from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass(frozen=True)
class TickerReport:
    """Current evidence and recommendation for one watched ticker."""

    ticker: str
    held_quantity: float
    current_price: float
    market_value: float
    evidence_score: float
    evidence_confidence: float
    probability_outperform: float | None
    probability_lower: float | None
    probability_upper: float | None
    calibration_sample_count: int | None
    recommendation: str
    explanation: str
    basket_memberships: tuple[tuple[str, ...], ...]
    basket_validation: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioReport:
    """Serializable result of a complete watchlist analysis."""

    generated_at_utc: str
    latest_price_date: str
    cash: float
    invested_value: float
    total_value: float
    tickers: tuple[TickerReport, ...]
    basket_diagnostics: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into JSON-compatible Python objects."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the report to JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    def table(self) -> pd.DataFrame:
        """Return a display-friendly table of ticker recommendations."""
        rows = []
        for item in self.tickers:
            row = asdict(item)
            row["basket_memberships"] = [", ".join(basket) for basket in item.basket_memberships]
            row["basket_validation"] = [
                f"{entry['basket']}: {entry['status']}" for entry in item.basket_validation
            ]
            row["warnings"] = list(item.warnings)
            rows.append(row)
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        sort_column = (
            "probability_outperform"
            if table["probability_outperform"].notna().any()
            else "evidence_score"
        )
        return table.sort_values(sort_column, ascending=False, na_position="last", ignore_index=True)


class PortfolioAnalyzer:
    """Run the complete data-to-recommendation Cobasket workflow."""

    def __init__(
        self,
        data_manager: DataManager | None = None,
        *,
        raw_policy: RecommendationPolicy | None = None,
        probability_policy: ProbabilityRecommendationPolicy | None = None,
        wide_interval_threshold: float = 0.30,
    ) -> None:
        if not 0.0 <= wide_interval_threshold <= 1.0:
            raise ValueError("wide_interval_threshold must lie in [0, 1]")
        self.data_manager = data_manager or DataManager()
        self.raw_policy = raw_policy or RecommendationPolicy()
        self.probability_policy = probability_policy or ProbabilityRecommendationPolicy()
        self.wide_interval_threshold = wide_interval_threshold

    def run(
        self,
        config: PortfolioConfig,
        *,
        force_refresh: bool = False,
    ) -> PortfolioReport:
        """Generate a current portfolio and watchlist report."""
        watchlist = BasketWatchlist.load(config.watchlist_path)
        calibration = (
            ProbabilityCalibration.load(config.calibration_path)
            if config.calibration_path is not None
            else None
        )
        validation = None
        validation_by_key: dict[str, Any] = {}
        if config.validation_path is not None:
            from cobasket.basket_validation import BasketValidationSet

            validation = BasketValidationSet.load(config.validation_path)
            validation_by_key = validation.by_key()

        prices = self.data_manager.prices(
            watchlist.tickers,
            period=config.period,
            force_refresh=force_refresh,
            min_coverage=1.0,
        )
        evaluation = evaluate_watchlist(
            prices,
            watchlist,
            holdings=config.holdings,
            window=config.z_window,
            min_trace_ratio=config.min_trace_ratio,
            policy=self.raw_policy,
            calibration=calibration,
        )

        raw_by_ticker = {item.ticker: item for item in evaluation.recommendations}
        calibrated_by_ticker = {
            item.ticker: item for item in (evaluation.calibrated_recommendations or ())
        }
        evidence_by_ticker = {item.ticker: item for item in evaluation.evidence}
        calibrated_evidence = {
            item.ticker: item for item in (evaluation.calibrated_evidence or ())
        }

        latest_date = pd.Timestamp(prices.index[-1])
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        latest_naive = latest_date.tz_localize(None) if latest_date.tzinfo else latest_date
        price_age_days = max(0.0, (now - latest_naive).total_seconds() / 86400.0)
        global_warnings: list[str] = []
        if price_age_days > config.max_price_age_days:
            global_warnings.append(
                f"Latest price is {price_age_days:.1f} days old, exceeding the "
                f"{config.max_price_age_days:.1f}-day limit."
            )
        if evaluation.failed_baskets:
            global_warnings.append(
                f"{len(evaluation.failed_baskets)} watchlist basket(s) failed current evaluation."
            )
        if calibration is None:
            global_warnings.append(
                "No probability calibration was supplied; recommendations use raw evidence thresholds."
            )
        if validation is None:
            global_warnings.append(
                "No basket validation profile was supplied; live recommendations are not reliability-gated."
            )

        ticker_reports: list[TickerReport] = []
        for ticker in watchlist.tickers:
            evidence = evidence_by_ticker.get(ticker)
            if evidence is None:
                continue
            held_quantity = float(config.holdings.get(ticker, 0.0))
            price = float(prices[ticker].iloc[-1])
            memberships = tuple(basket for basket in watchlist.baskets if ticker in basket)
            warnings: list[str] = []
            if len(memberships) == 1:
                warnings.append("Recommendation is supported by only one watchlist basket.")

            validation_entries: list[dict[str, str]] = []
            validated_memberships = 0
            if validation is not None:
                for basket in memberships:
                    key = ", ".join(basket)
                    profile = validation_by_key.get(key)
                    status = profile.status if profile is not None else "missing"
                    validation_entries.append({"basket": key, "status": status})
                    if status == "validated":
                        validated_memberships += 1
                if not validation_entries:
                    warnings.append("Ticker has no matching basket validation profile.")
                elif validated_memberships == 0:
                    statuses = sorted({entry["status"] for entry in validation_entries})
                    warnings.append(
                        "No supporting basket is historically validated "
                        f"(statuses: {', '.join(statuses)})."
                    )

            calibrated = calibrated_evidence.get(ticker)
            calibrated_recommendation = calibrated_by_ticker.get(ticker)
            if calibrated is not None and calibrated_recommendation is not None:
                probability = float(calibrated.probability_outperform)
                lower = float(calibrated.probability_lower)
                upper = float(calibrated.probability_upper)
                sample_count = int(calibrated.sample_count)
                recommendation = calibrated_recommendation.action
                explanation = calibrated_recommendation.explanation
                if sample_count < self.probability_policy.min_samples:
                    warnings.append(
                        f"Calibration bin has only {sample_count} historical examples."
                    )
                if upper - lower > self.wide_interval_threshold:
                    warnings.append("Calibrated probability interval is wide.")
            else:
                probability = lower = upper = None
                sample_count = None
                raw = raw_by_ticker[ticker]
                recommendation = raw.action
                explanation = raw.explanation
                warnings.append("Ticker has no calibrated probability.")

            if validation is not None and validated_memberships == 0:
                gated_action = "Hold" if held_quantity > 0.0 else "Wait"
                explanation = (
                    f"{gated_action}: current statistical evidence is not used for an actionable "
                    "recommendation because none of this ticker's supporting baskets currently has "
                    "validated historical reliability. "
                    + explanation
                )
                recommendation = gated_action

            ticker_reports.append(
                TickerReport(
                    ticker=ticker,
                    held_quantity=held_quantity,
                    current_price=price,
                    market_value=held_quantity * price,
                    evidence_score=float(evidence.score),
                    evidence_confidence=float(evidence.confidence),
                    probability_outperform=probability,
                    probability_lower=lower,
                    probability_upper=upper,
                    calibration_sample_count=sample_count,
                    recommendation=recommendation,
                    explanation=explanation,
                    basket_memberships=memberships,
                    basket_validation=tuple(validation_entries),
                    warnings=tuple(warnings),
                )
            )

        ticker_reports.sort(
            key=lambda item: (
                item.probability_outperform
                if item.probability_outperform is not None
                else item.evidence_score
            ),
            reverse=True,
        )
        invested_value = float(sum(item.market_value for item in ticker_reports))
        diagnostics = evaluation.basket_diagnostics.copy()
        if validation is not None and not diagnostics.empty:
            diagnostics["basket_key"] = diagnostics["basket"].apply(
                lambda value: ", ".join(value) if isinstance(value, (list, tuple)) else str(value)
            )
            diagnostics["validation_status"] = diagnostics["basket_key"].map(
                lambda key: validation_by_key[key].status if key in validation_by_key else "missing"
            )
            diagnostics = diagnostics.drop(columns=["basket_key"])
        metadata: dict[str, Any] = {
            "watchlist_name": watchlist.name,
            "period": config.period,
            "z_window": config.z_window,
            "min_trace_ratio": config.min_trace_ratio,
            "price_age_days": price_age_days,
            "calibrated": calibration is not None,
            "validation_gated": validation is not None,
        }
        if validation is not None:
            metadata["validation_generated_at_utc"] = validation.generated_at_utc
        if getattr(self.data_manager, "last_metadata", None) is not None:
            metadata["data"] = asdict(self.data_manager.last_metadata)

        return PortfolioReport(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            latest_price_date=latest_date.isoformat(),
            cash=float(config.cash),
            invested_value=invested_value,
            total_value=float(config.cash + invested_value),
            tickers=tuple(ticker_reports),
            basket_diagnostics=tuple(diagnostics.to_dict(orient="records")),
            warnings=tuple(global_warnings),
            metadata=metadata,
        )
