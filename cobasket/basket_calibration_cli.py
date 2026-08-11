"""Command-line interface for basket-specific probability calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cobasket.basket_calibration import fit_basket_calibrations
from cobasket.basket_validation import BasketValidationSet
from cobasket.config_paths import resolve_portfolio_config_paths
from cobasket.workflow import PortfolioConfig


def _update_portfolio_path(portfolio_path: Path, calibration_path: Path) -> None:
    """Store a basket-calibration path in a portfolio configuration.

    Parameters
    ----------
    portfolio_path
        Existing portfolio configuration JSON.
    calibration_path
        Basket-calibration JSON written by the command.

    Returns
    -------
    None
    """
    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    try:
        stored = str(calibration_path.relative_to(portfolio_path.parent))
    except ValueError:
        stored = str(calibration_path)
    payload["basket_calibration_path"] = stored
    portfolio_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Fit and save independent probability calibrations for validated baskets."""
    parser = argparse.ArgumentParser(
        description="Fit basket-specific Cobasket probability calibrations for historically validated baskets."
    )
    parser.add_argument("--portfolio", required=True, help="Portfolio configuration JSON")
    parser.add_argument("--validation", help="Basket validation JSON; defaults to validation_path in portfolio")
    parser.add_argument("--output", default="basket_calibration.json", help="Basket calibration JSON output")
    parser.add_argument("--min-evaluations", type=int, default=20, help="Minimum independent evaluation dates")
    parser.add_argument("--train-window", type=int, default=252, help="Trailing fit window in trading days")
    parser.add_argument("--horizon", type=int, default=20, help="Forward outcome horizon in trading days")
    parser.add_argument("--step", type=int, default=20, help="Spacing between historical evaluations")
    parser.add_argument("--z-window", type=int, help="Override the portfolio z-score window")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass valid price caches")
    parser.add_argument(
        "--update-portfolio",
        action="store_true",
        help="Write basket_calibration_path into the portfolio configuration",
    )
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio).expanduser().resolve()
    config = resolve_portfolio_config_paths(PortfolioConfig.load(portfolio_path), portfolio_path)
    raw_validation = args.validation or config.validation_path
    if raw_validation is None:
        raise SystemExit("No basket validation file supplied. Run cobasket-validate first or pass --validation.")
    validation_path = Path(raw_validation).expanduser().resolve()
    validation = BasketValidationSet.load(validation_path)

    output = Path(args.output).expanduser().resolve()
    result = fit_basket_calibrations(
        portfolio_path,
        validation_path,
        min_evaluations=args.min_evaluations,
        train_window=args.train_window,
        z_window=args.z_window,
        horizon=args.horizon,
        step=args.step,
        force_refresh=args.force_refresh,
    )
    result.save(output)

    print("\nBasket-specific calibration summary:")
    rows = []
    calibrated = result.by_key()
    for profile in validation.profiles:
        key = profile.key
        if key in calibrated:
            item = calibrated[key]
            rows.append((key, "calibrated", item.accepted_evaluations, ""))
        else:
            rows.append((key, "skipped", profile.accepted_evaluations, result.skipped.get(key, "not eligible")))
    width = max(len(row[0]) for row in rows) if rows else 6
    for basket, status, evaluations, reason in rows:
        suffix = f" - {reason}" if reason else ""
        print(f"{basket:<{width}}  {status:<10} evaluations={evaluations:>3}{suffix}")

    print(f"\nSaved basket calibrations to {output}")
    if args.update_portfolio:
        _update_portfolio_path(portfolio_path, output)
        print(f"Updated basket_calibration_path in {portfolio_path}")


if __name__ == "__main__":
    main()
