"""End-to-end portfolio reporting for Cobasket decision support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cobasket.data import DataManager
from cobasket.evidence import (
    BasketWatchlist,
    CalibratedAssetEvidence,
    ProbabilityCalibration,
    ProbabilityRecommendationPolicy,
    RecommendationPolicy,
    cointegration_evidence,
    evaluate_watchlist,
)
from cobasket.fx import latest_fx_quote, normalize_currency


@dataclass(frozen=True)
class PortfolioConfig:
    """Persistent inputs for a live portfolio analysis.

    Parameters
    ----------
    holdings
        Mapping from ticker symbol to currently owned quantity.
    cash
        Uninvested cash, denominated in ``base_currency``.
    watchlist_path
        JSON file containing a monitored basket watchlist.
    base_currency
        Currency used for cash, market values, and portfolio totals. When ``None``,
        Cobasket uses the watchlist's native analysis currency.
    calibration_path
        Optional global probability-calibration JSON.
    validation_path
        Optional basket-validation-profile JSON.
    basket_calibration_path
        Optional basket-specific probability-calibration JSON.
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
    base_currency: str | None = None
    calibration_path: str | None = None
    validation_path: str | None = None
    basket_calibration_path: str | None = None
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
        base_currency = None if self.base_currency is None else normalize_currency(self.base_currency)
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "base_currency", base_currency)

    def save(self, path: str | Path) -> Path:
        """Write the configuration to human-readable JSON.

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
        output.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioConfig":
        """Load a portfolio configuration from JSON.

        Parameters
        ----------
        path
            Existing JSON configuration.

        Returns
        -------
        PortfolioConfig
            Loaded configuration.
        """
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
    native_currency: str | None = None
    base_currency: str | None = None
    fx_rate_to_base: float = 1.0
    basket_validation: tuple[dict[str, str], ...] = ()
    probability_sources: tuple[str, ...] = ()
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
        """Write the report to JSON.

        Parameters
        ----------
        path
            Destination JSON path.

        Returns
        -------
        pathlib.Path
            Written path.
        """
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
            row["probability_sources"] = list(item.probability_sources)
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


def _watchlist_currency_metadata(path: str | Path) -> tuple[str | None, float]:
    """Read native analysis currency and quote scale from a watchlist JSON.

    Parameters
    ----------
    path
        Existing watchlist JSON.

    Returns
    -------
    tuple
        ``(analysis_currency, price_scale)``. Missing legacy metadata returns
        ``(None, 1.0)``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("universe_metadata") or {}
    currency = metadata.get("analysis_currency")
    price_scale = float(metadata.get("price_scale", 1.0))
    if price_scale <= 0.0:
        raise ValueError("watchlist universe price_scale must be positive")
    return (normalize_currency(currency) if currency else None, price_scale)


def _combine_basket_calibrations(
    *,
    ticker: str,
    evidence: Any,
    memberships: tuple[tuple[str, ...], ...],
    prices: pd.DataFrame,
    calibration_by_key: Mapping[str, Any],
    window: int,
    min_trace_ratio: float,
) -> tuple[CalibratedAssetEvidence, tuple[str, ...]] | None:
    """Combine basket-specific probabilities for one ticker."""
    values: list[tuple[float, float, float, int, int, int, str]] = []
    for basket in memberships:
        key = ", ".join(basket)
        stored = calibration_by_key.get(key)
        if stored is None:
            continue
        try:
            result = cointegration_evidence(
                prices.loc[:, list(basket)],
                window=window,
                min_trace_ratio=min_trace_ratio,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        item = next((candidate for candidate in result.asset_evidence if candidate.ticker == ticker), None)
        if item is None:
            continue
        calibrated = stored.calibration.calibrate(item)
        weight = max(int(stored.accepted_evaluations), 1)
        values.append(
            (
                calibrated.probability_outperform,
                calibrated.probability_lower,
                calibrated.probability_upper,
                calibrated.sample_count,
                calibrated.horizon,
                weight,
                key,
            )
        )

    if not values:
        return None
    weights = np.asarray([value[5] for value in values], dtype=float)
    combined = CalibratedAssetEvidence(
        evidence=evidence,
        probability_outperform=float(np.average([value[0] for value in values], weights=weights)),
        probability_lower=float(np.average([value[1] for value in values], weights=weights)),
        probability_upper=float(np.average([value[2] for value in values], weights=weights)),
        sample_count=int(sum(value[3] for value in values)),
        horizon=int(round(np.average([value[4] for value in values], weights=weights))),
        benchmark="equal-weight basket",
    )
    return combined, tuple(value[6] for value in values)


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

    def run(self, config: PortfolioConfig, *, force_refresh: bool = False) -> PortfolioReport:
        """Generate a current portfolio and watchlist report."""
        watchlist = BasketWatchlist.load(config.watchlist_path)
        native_currency, price_scale = _watchlist_currency_metadata(config.watchlist_path)
        base_currency = config.base_currency or native_currency or "USD"
        if native_currency is None:
            native_currency = base_currency

        global_calibration = (
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

        basket_calibration = None
        basket_calibration_by_key: dict[str, Any] = {}
        if config.basket_calibration_path is not None:
            from cobasket.basket_calibration import BasketCalibrationSet

            basket_calibration = BasketCalibrationSet.load(config.basket_calibration_path)
            basket_calibration_by_key = basket_calibration.by_key()

        prices = self.data_manager.prices(
            watchlist.tickers,
            period=config.period,
            force_refresh=force_refresh,
            min_coverage=1.0,
        )
        equity_metadata = getattr(self.data_manager, "last_metadata", None)
        evaluation = evaluate_watchlist(
            prices,
            watchlist,
            holdings=config.holdings,
            window=config.z_window,
            min_trace_ratio=config.min_trace_ratio,
            policy=self.raw_policy,
            calibration=(global_calibration if basket_calibration is None else None),
        )

        raw_by_ticker = {item.ticker: item for item in evaluation.recommendations}
        globally_calibrated_by_ticker = {
            item.ticker: item for item in (evaluation.calibrated_recommendations or ())
        }
        evidence_by_ticker = {item.ticker: item for item in evaluation.evidence}
        globally_calibrated_evidence = {
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
        if basket_calibration is None and global_calibration is None:
            global_warnings.append(
                "No probability calibration was supplied; recommendations use raw evidence thresholds."
            )
        elif basket_calibration is not None:
            global_warnings.append(
                "Basket-specific calibration is active; pooled calibration is not used for missing baskets."
            )
        if validation is None:
            global_warnings.append(
                "No basket validation profile was supplied; live recommendations are not reliability-gated."
            )

        held_total = sum(float(config.holdings.get(ticker, 0.0)) for ticker in watchlist.tickers)
        fx_quote = latest_fx_quote(
            self.data_manager,
            native_currency,
            base_currency,
            force_refresh=force_refresh,
        ) if held_total > 0.0 else None
        fx_rate = 1.0 if fx_quote is None else fx_quote.rate

        ticker_reports: list[TickerReport] = []
        for ticker in watchlist.tickers:
            evidence = evidence_by_ticker.get(ticker)
            if evidence is None:
                continue
            held_quantity = float(config.holdings.get(ticker, 0.0))
            native_price = float(prices[ticker].iloc[-1]) * price_scale
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

            probability_sources: tuple[str, ...] = ()
            if basket_calibration is not None:
                combined = _combine_basket_calibrations(
                    ticker=ticker,
                    evidence=evidence,
                    memberships=memberships,
                    prices=prices,
                    calibration_by_key=basket_calibration_by_key,
                    window=config.z_window,
                    min_trace_ratio=config.min_trace_ratio,
                )
                if combined is not None:
                    calibrated, probability_sources = combined
                    classified = self.probability_policy.classify(
                        calibrated,
                        currently_held=held_quantity > 0.0,
                    )
                    probability = float(calibrated.probability_outperform)
                    lower = float(calibrated.probability_lower)
                    upper = float(calibrated.probability_upper)
                    sample_count = int(calibrated.sample_count)
                    recommendation = classified.action
                    explanation = (
                        f"Basket-specific calibration from {len(probability_sources)} validated "
                        f"basket(s). {classified.explanation}"
                    )
                    if sample_count < self.probability_policy.min_samples:
                        warnings.append(
                            f"Basket-specific calibration has only {sample_count} score-bin examples."
                        )
                    if upper - lower > self.wide_interval_threshold:
                        warnings.append("Basket-specific probability interval is wide.")
                else:
                    probability = lower = upper = None
                    sample_count = None
                    raw = raw_by_ticker[ticker]
                    recommendation = raw.action
                    explanation = raw.explanation
                    warnings.append("Ticker has no eligible basket-specific probability calibration.")
            else:
                calibrated = globally_calibrated_evidence.get(ticker)
                calibrated_recommendation = globally_calibrated_by_ticker.get(ticker)
                if calibrated is not None and calibrated_recommendation is not None:
                    probability = float(calibrated.probability_outperform)
                    lower = float(calibrated.probability_lower)
                    upper = float(calibrated.probability_upper)
                    sample_count = int(calibrated.sample_count)
                    recommendation = calibrated_recommendation.action
                    explanation = calibrated_recommendation.explanation
                    probability_sources = ("pooled calibration",)
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

            reliability_gate = validation is not None and validated_memberships == 0
            calibration_gate = basket_calibration is not None and not probability_sources
            if reliability_gate or calibration_gate:
                gated_action = "Hold" if held_quantity > 0.0 else "Wait"
                reason = (
                    "none of this ticker's supporting baskets currently has validated historical reliability"
                    if reliability_gate
                    else "no validated supporting basket has enough independent history for basket-specific calibration"
                )
                explanation = (
                    f"{gated_action}: current statistical evidence is not used for an actionable "
                    f"recommendation because {reason}. " + explanation
                )
                recommendation = gated_action

            ticker_reports.append(
                TickerReport(
                    ticker=ticker,
                    held_quantity=held_quantity,
                    current_price=native_price,
                    market_value=held_quantity * native_price * fx_rate,
                    evidence_score=float(evidence.score),
                    evidence_confidence=float(evidence.confidence),
                    probability_outperform=probability,
                    probability_lower=lower,
                    probability_upper=upper,
                    calibration_sample_count=sample_count,
                    recommendation=recommendation,
                    explanation=explanation,
                    basket_memberships=memberships,
                    native_currency=native_currency,
                    base_currency=base_currency,
                    fx_rate_to_base=fx_rate,
                    basket_validation=tuple(validation_entries),
                    probability_sources=probability_sources,
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
            if basket_calibration is not None:
                diagnostics["basket_calibrated"] = diagnostics["basket_key"].isin(
                    basket_calibration_by_key
                )
            diagnostics = diagnostics.drop(columns=["basket_key"])

        metadata: dict[str, Any] = {
            "watchlist_name": watchlist.name,
            "period": config.period,
            "z_window": config.z_window,
            "min_trace_ratio": config.min_trace_ratio,
            "price_age_days": price_age_days,
            "calibrated": basket_calibration is not None or global_calibration is not None,
            "calibration_mode": (
                "basket_specific"
                if basket_calibration is not None
                else "pooled"
                if global_calibration is not None
                else "none"
            ),
            "validation_gated": validation is not None,
            "native_currency": native_currency,
            "base_currency": base_currency,
            "price_scale": price_scale,
            "fx_rate_to_base": fx_rate,
            "fx_ticker": None if fx_quote is None else fx_quote.ticker,
        }
        if validation is not None:
            metadata["validation_generated_at_utc"] = validation.generated_at_utc
        if basket_calibration is not None:
            metadata["basket_calibration_generated_at_utc"] = basket_calibration.generated_at_utc
            metadata["basket_calibration_count"] = len(basket_calibration.calibrations)
        if equity_metadata is not None:
            metadata["data"] = asdict(equity_metadata)

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
