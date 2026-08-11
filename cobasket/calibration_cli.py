"""Command-line interface for watchlist probability calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cobasket.calibration_workflow import calibrate_watchlist
from cobasket.evidence import save_probability_calibration


def _update_portfolio_calibration(portfolio_path: Path, calibration_path: Path) -> None:
    """Store a calibration path in an existing portfolio configuration.

    Parameters
    ----------
    portfolio_path
        Existing portfolio configuration JSON.
    calibration_path
        Calibration file written by the command.

    Returns
    -------
    None
    """
    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    payload["calibration_path"] = str(calibration_path.resolve())
    portfolio_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Fit and save a leakage-safe watchlist probability calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate Cobasket evidence scores using historical walk-forward outcomes."
    )
    parser.add_argument("--portfolio", required=True, help="Portfolio configuration JSON")
    parser.add_argument("--output", default="probability_calibration.json", help="Calibration JSON output")
    parser.add_argument("--records-out", help="Optional CSV or Parquet file for pooled historical outcomes")
    parser.add_argument("--train-window", type=int, default=252, help="Trailing fit window in trading days")
    parser.add_argument("--horizon", type=int, default=20, help="Forward outcome horizon in trading days")
    parser.add_argument("--step", type=int, default=5, help="Spacing between historical evaluations")
    parser.add_argument("--z-window", type=int, help="Override the portfolio z-score window")
    parser.add_argument("--min-trace-ratio", type=float, help="Override the historical cointegration threshold")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass valid market-data caches")
    parser.add_argument(
        "--update-portfolio",
        action="store_true",
        help="Write the saved calibration path into the portfolio configuration",
    )
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    result = calibrate_watchlist(
        portfolio_path,
        train_window=args.train_window,
        z_window=args.z_window,
        horizon=args.horizon,
        step=args.step,
        min_trace_ratio=args.min_trace_ratio,
        force_refresh=args.force_refresh,
    )
    save_probability_calibration(result.calibration, output)

    print("\nBasket calibration summary:")
    print(result.basket_summary.to_string(index=False))
    print("\nProbability calibration:")
    print(result.calibration.table.to_string(index=False))
    print(f"\nPooled historical asset outcomes: {len(result.records)}")
    print(f"Saved calibration to {output}")

    if args.records_out:
        records_path = Path(args.records_out).expanduser().resolve()
        records_path.parent.mkdir(parents=True, exist_ok=True)
        if records_path.suffix.lower() in {".parquet", ".pq"}:
            result.records.to_parquet(records_path, index=False)
        else:
            result.records.to_csv(records_path, index=False)
        print(f"Saved calibration records to {records_path}")

    if args.update_portfolio:
        _update_portfolio_calibration(portfolio_path, output)
        print(f"Updated calibration_path in {portfolio_path}")


if __name__ == "__main__":
    main()
