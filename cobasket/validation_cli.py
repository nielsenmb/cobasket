"""Command-line interface for persistent basket validation profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cobasket.basket_validation import validate_watchlist_baskets, validation_table


def _update_portfolio_validation(portfolio_path: Path, validation_path: Path) -> None:
    """Store a validation-profile path in an existing portfolio configuration.

    Parameters
    ----------
    portfolio_path
        Existing portfolio configuration JSON.
    validation_path
        Validation JSON written by this command.
    """
    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    try:
        stored = validation_path.resolve().relative_to(portfolio_path.resolve().parent)
        payload["validation_path"] = str(stored)
    except ValueError:
        payload["validation_path"] = str(validation_path.resolve())
    portfolio_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Validate every monitored basket and save persistent reliability profiles."""
    parser = argparse.ArgumentParser(
        description="Build historical validation profiles for Cobasket watchlist baskets."
    )
    parser.add_argument("--portfolio", required=True, help="Portfolio configuration JSON")
    parser.add_argument("--output", default="basket_validation.json", help="Validation JSON output")
    parser.add_argument("--train-window", type=int, default=252, help="Trailing fit window in trading days")
    parser.add_argument("--horizon", type=int, default=20, help="Forward relative-performance horizon")
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="Spacing between historical evaluations; default avoids overlapping outcomes",
    )
    parser.add_argument("--z-window", type=int, help="Override the portfolio z-score window")
    parser.add_argument("--min-trace-ratio", type=float, help="Override the Johansen threshold")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass valid market-data caches")
    parser.add_argument(
        "--update-portfolio",
        action="store_true",
        help="Write the validation file path into portfolio.json",
    )
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    result = validate_watchlist_baskets(
        portfolio_path,
        train_window=args.train_window,
        z_window=args.z_window,
        horizon=args.horizon,
        step=args.step,
        min_trace_ratio=args.min_trace_ratio,
        force_refresh=args.force_refresh,
    )
    result.save(output)

    table = validation_table(result)
    display_columns = [
        "basket",
        "status",
        "current_trace_ratio",
        "accepted_evaluations",
        "acceptance_rate",
        "weight_stability",
        "score_return_correlation",
        "calibration_contrast",
        "reasons",
    ]
    print("\nBasket validation summary:")
    print(table.loc[:, display_columns].to_string(index=False))
    print(f"\nSaved validation profiles to {output}")

    if args.update_portfolio:
        _update_portfolio_validation(portfolio_path, output)
        print(f"Updated validation_path in {portfolio_path}")


if __name__ == "__main__":
    main()
