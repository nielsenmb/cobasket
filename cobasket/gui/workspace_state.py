"""Inspect a Cobasket workspace and determine the next useful action."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceState:
    """Summary of files and dependencies in a Cobasket workspace.

    Parameters
    ----------
    name
        Short machine-readable state name.
    summary
        Human-readable explanation of the current workspace state.
    next_stage
        Recommended next workflow stage.
    next_label
        Label for the primary workflow button.
    update_stages
        Stages required to bring the workspace through a fresh report.
    stage_statuses
        Ordered ``(stage, status)`` pairs for display in the GUI.
    """

    name: str
    summary: str
    next_stage: str
    next_label: str
    update_stages: tuple[str, ...]
    stage_statuses: tuple[tuple[str, str], ...]


def _mtime(path: Path) -> float:
    """Return the modification time for an existing path.

    Parameters
    ----------
    path
        Existing filesystem path.

    Returns
    -------
    float
        Modification timestamp in seconds since the epoch.
    """
    return path.stat().st_mtime


def _resolve_path(root: Path, value: object, fallback: str) -> Path:
    """Resolve a configured artifact path relative to its workspace.

    Parameters
    ----------
    root
        Workspace directory containing ``portfolio.json``.
    value
        Configured path value, which may be missing or ``None``.
    fallback
        Standard workflow filename used when no path is configured.

    Returns
    -------
    pathlib.Path
        Resolved artifact path.
    """
    if not value:
        return root / fallback
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _workspace_paths(root: Path) -> dict[str, Path]:
    """Return workflow artifact paths, following portfolio configuration when possible.

    Parameters
    ----------
    root
        Workspace directory.

    Returns
    -------
    dict of str to pathlib.Path
        Paths for portfolio, watchlist, validation, calibration, discovery table,
        and report.
    """
    portfolio = root / "portfolio.json"
    payload: dict[str, object] = {}
    if portfolio.exists():
        try:
            loaded = json.loads(portfolio.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}

    return {
        "portfolio": portfolio,
        "watchlist": _resolve_path(root, payload.get("watchlist_path"), "discovered_watchlist.json"),
        "validation": _resolve_path(root, payload.get("validation_path"), "basket_validation.json"),
        "calibration": _resolve_path(
            root, payload.get("basket_calibration_path"), "basket_calibration.json"
        ),
        "discovery": root / "discovery_results.csv",
        "report": root / "report.json",
    }


def inspect_workspace(workspace: str | Path) -> WorkspaceState:
    """Determine the current guided-workflow state of a workspace.

    Parameters
    ----------
    workspace
        Directory containing Cobasket workflow artifacts.

    Returns
    -------
    WorkspaceState
        Recommended next action and per-stage status.

    Notes
    -----
    Existing workspaces are inspected using the artifact paths stored in
    ``portfolio.json`` rather than assuming the newest default filenames.
    Validation is considered stale when the configured watchlist is newer than
    validation. Calibration is stale when validation is newer than calibration.
    A report is stale when the portfolio or any upstream model artifact changed
    after the report was written.
    """
    root = Path(workspace).expanduser().resolve()
    paths = _workspace_paths(root)
    portfolio = paths["portfolio"]
    watchlist = paths["watchlist"]
    discovery = paths["discovery"]
    validation = paths["validation"]
    calibration = paths["calibration"]
    report = paths["report"]

    has_portfolio = portfolio.exists()
    has_watchlist = watchlist.exists()
    has_validation = validation.exists()
    has_calibration = calibration.exists()
    has_report = report.exists()

    if not (has_portfolio and has_watchlist):
        discovery_status = "ran, but no promising watchlist was produced" if discovery.exists() else "not run"
        summary = (
            "Discovery has run, but no promising basket produced a usable workspace. "
            "Try discovery again, possibly with another universe or period."
            if discovery.exists()
            else "This is an empty or incomplete workspace. Start by discovering persistent candidate baskets."
        )
        return WorkspaceState(
            name="empty" if not discovery.exists() else "no_candidates",
            summary=summary,
            next_stage="discover",
            next_label="Discover baskets",
            update_stages=("discover",),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", "waiting for discovery"),
                ("Calibration", "waiting for validation"),
                ("Live report", "waiting for calibration"),
            ),
        )

    validation_stale = has_validation and _mtime(validation) < _mtime(watchlist)
    calibration_stale = has_calibration and (
        not has_validation or _mtime(calibration) < _mtime(validation)
    )
    report_dependencies = [portfolio, watchlist]
    if has_validation:
        report_dependencies.append(validation)
    if has_calibration:
        report_dependencies.append(calibration)
    report_stale = has_report and any(_mtime(report) < _mtime(path) for path in report_dependencies)

    discovery_status = "ready"
    validation_status = "needs update" if validation_stale else ("ready" if has_validation else "not run")
    calibration_status = "needs update" if calibration_stale else ("ready" if has_calibration else "not run")
    report_status = "needs refresh" if report_stale else ("ready" if has_report else "not run")

    if not has_validation or validation_stale:
        return WorkspaceState(
            name="needs_validation",
            summary="Candidate baskets are available. Validate their historical persistence next.",
            next_stage="validate",
            next_label="Validate baskets",
            update_stages=("validate", "calibrate", "report"),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", validation_status),
                ("Calibration", "waiting for validation" if not has_calibration else calibration_status),
                ("Live report", report_status),
            ),
        )

    if not has_calibration or calibration_stale:
        return WorkspaceState(
            name="needs_calibration",
            summary="Validation is available. Fit basket-specific probability calibration next.",
            next_stage="calibrate",
            next_label="Calibrate probabilities",
            update_stages=("calibrate", "report"),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", validation_status),
                ("Calibration", calibration_status),
                ("Live report", report_status),
            ),
        )

    if not has_report or report_stale:
        return WorkspaceState(
            name="needs_report",
            summary="The model state is ready. Generate a current report using the latest holdings and prices.",
            next_stage="report",
            next_label="Generate live report",
            update_stages=("report",),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", validation_status),
                ("Calibration", calibration_status),
                ("Live report", report_status),
            ),
        )

    return WorkspaceState(
        name="ready",
        summary=(
            "This workspace is complete. Use Refresh recommendations whenever holdings or market prices "
            "need updating; re-run discovery separately when you want to reconsider the basket universe."
        ),
        next_stage="report",
        next_label="Refresh recommendations",
        update_stages=("report",),
        stage_statuses=(
            ("Discovery", discovery_status),
            ("Validation", validation_status),
            ("Calibration", calibration_status),
            ("Live report", report_status),
        ),
    )
