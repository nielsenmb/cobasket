"""Command-line interface for live Cobasket portfolio reports."""

from __future__ import annotations

import argparse

from cobasket.workflow import PortfolioAnalyzer, PortfolioConfig


def main() -> None:
    """Generate and save a current portfolio report.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Generate a current long-only Cobasket portfolio report."
    )
    parser.add_argument("--portfolio", required=True, help="Portfolio configuration JSON")
    parser.add_argument("--watchlist", help="Override the watchlist path in the configuration")
    parser.add_argument("--calibration", help="Override the calibration path in the configuration")
    parser.add_argument("--period", help="Override the historical download period")
    parser.add_argument("--output", default="cobasket_report.json", help="Output report JSON")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass valid price caches")
    args = parser.parse_args()

    config = PortfolioConfig.load(args.portfolio)
    payload = {
        **config.__dict__,
        "watchlist_path": args.watchlist or config.watchlist_path,
        "calibration_path": args.calibration or config.calibration_path,
        "period": args.period or config.period,
    }
    report = PortfolioAnalyzer().run(
        PortfolioConfig(**payload),
        force_refresh=args.force_refresh,
    )
    report.save(args.output)
    print(report.table().to_string(index=False))
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"\nSaved report to {args.output}")
