"""Command-line interface for live Cobasket portfolio reports."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np

from cobasket.config_paths import resolve_portfolio_config_paths
from cobasket.data import DataManager
from cobasket.evidence import BasketWatchlist, cointegration_evidence
from cobasket.workflow import PortfolioAnalyzer, PortfolioConfig


def _diagnose_failed_baskets(config: PortfolioConfig, manager: DataManager) -> tuple[str, ...]:
    """Return readable reasons for baskets that cannot produce current evidence."""
    watchlist = BasketWatchlist.load(config.watchlist_path)
    prices = manager.prices(watchlist.tickers, period=config.period, min_coverage=1.0)
    failures: list[str] = []
    for basket in watchlist.baskets:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Casting complex values to real discards the imaginary part",
                )
                cointegration_evidence(
                    prices.loc[:, list(basket)],
                    window=config.z_window,
                    min_trace_ratio=config.min_trace_ratio,
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append(f"{', '.join(basket)}: {type(exc).__name__}: {exc}")
    return tuple(failures)


def main() -> None:
    """Generate and save a current portfolio report."""
    parser = argparse.ArgumentParser(
        description="Generate a current long-only Cobasket portfolio report."
    )
    parser.add_argument("--portfolio", required=True, help="Portfolio configuration JSON")
    parser.add_argument("--watchlist", help="Override the watchlist path in the configuration")
    parser.add_argument("--calibration", help="Override the pooled calibration path")
    parser.add_argument("--validation", help="Override the basket-validation path")
    parser.add_argument("--basket-calibration", help="Override the basket-specific calibration path")
    parser.add_argument("--period", help="Override the historical download period")
    parser.add_argument("--output", default="cobasket_report.json", help="Output report JSON")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass valid price caches")
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio).expanduser().resolve()
    config = resolve_portfolio_config_paths(PortfolioConfig.load(portfolio_path), portfolio_path)
    payload = {
        **config.__dict__,
        "watchlist_path": args.watchlist or config.watchlist_path,
        "calibration_path": args.calibration or config.calibration_path,
        "validation_path": args.validation or config.validation_path,
        "basket_calibration_path": args.basket_calibration or config.basket_calibration_path,
        "period": args.period or config.period,
    }
    resolved = PortfolioConfig(**payload)
    manager = DataManager()
    analyzer = PortfolioAnalyzer(data_manager=manager)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
        report = analyzer.run(resolved, force_refresh=args.force_refresh)

    if any("failed current evaluation" in item for item in report.warnings):
        failures = _diagnose_failed_baskets(resolved, manager)
        if failures:
            metadata = dict(report.metadata)
            metadata["failed_basket_diagnostics"] = list(failures)
            report = replace(
                report,
                warnings=report.warnings + tuple(f"Basket failure: {item}" for item in failures),
                metadata=metadata,
            )

    report.save(args.output)
    print(report.table().to_string(index=False))
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"\nSaved report to {args.output}")
