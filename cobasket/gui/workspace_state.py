"""Inspect a Cobasket workspace and determine the next useful action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time


_SECONDS_PER_DAY = 86400.0
_FRESHNESS_FILENAME = "workspace_freshness.json"


@dataclass(frozen=True)
class FreshnessPolicy:
    """Recommended maximum ages for normal Cobasket workflow artifacts.

    Parameters
    ----------
    report_days
        Age after which a live report refresh is recommended.
    validation_days
        Age after which historical basket validation should be refreshed.
    calibration_days
        Age after which basket-specific probability calibration should be refreshed.
    discovery_days
        Age after which re-running basket discovery is recommended. Discovery is
        advisory only because it can replace the watchlist and invalidate downstream
        analysis.
    """

    report_days: float = 3.0
    validation_days: float = 90.0
    calibration_days: float = 90.0
    discovery_days: float = 180.0

    def __post_init__(self) -> None:
        """Validate freshness intervals.

        Raises
        ------
        ValueError
            If any refresh interval is not strictly positive.
        """
        values = {
            "report_days": self.report_days,
            "validation_days": self.validation_days,
            "calibration_days": self.calibration_days,
            "discovery_days": self.discovery_days,
        }
        for name, value in values.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def save(self, path: str | Path) -> Path:
        """Save the freshness policy as JSON.

        Parameters
        ----------
        path
            Destination JSON path.

        Returns
        -------
        pathlib.Path
            Written path.
        """
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FreshnessPolicy":
        """Load a freshness policy, using defaults when the file is absent.

        Parameters
        ----------
        path
            Policy JSON path.

        Returns
        -------
        FreshnessPolicy
            Loaded policy, or the default policy when ``path`` does not exist.
        """
        source = Path(path).expanduser().resolve()
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(**payload)


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


def freshness_policy_path(workspace: str | Path) -> Path:
    """Return the persistent freshness-policy path for a workspace.

    Parameters
    ----------
    workspace
        Cobasket workspace directory.

    Returns
    -------
    pathlib.Path
        ``workspace_freshness.json`` inside the workspace.
    """
    return Path(workspace).expanduser().resolve() / _FRESHNESS_FILENAME


def load_freshness_policy(workspace: str | Path) -> FreshnessPolicy:
    """Load the freshness policy associated with a workspace.

    Parameters
    ----------
    workspace
        Cobasket workspace directory.

    Returns
    -------
    FreshnessPolicy
        Stored policy or the default policy when no settings file exists.
    """
    return FreshnessPolicy.load(freshness_policy_path(workspace))


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


def _age_days(path: Path, now: float) -> float:
    """Return the non-negative age of a file in days.

    Parameters
    ----------
    path
        Existing artifact path.
    now
        Reference Unix timestamp.

    Returns
    -------
    float
        File age in days.
    """
    return max(0.0, (now - _mtime(path)) / _SECONDS_PER_DAY)


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


def _age_status(label: str, age_days: float, due: bool) -> str:
    """Format a compact artifact-age status.

    Parameters
    ----------
    label
        Base status label.
    age_days
        Artifact age in days.
    due
        Whether a freshness refresh is recommended.

    Returns
    -------
    str
        Human-readable status.
    """
    if due:
        return f"refresh recommended ({age_days:.0f} days old)"
    return label


def inspect_workspace(
    workspace: str | Path,
    *,
    now: float | None = None,
    policy: FreshnessPolicy | None = None,
) -> WorkspaceState:
    """Determine the current guided-workflow state of a workspace.

    Parameters
    ----------
    workspace
        Directory containing Cobasket workflow artifacts.
    now
        Optional Unix timestamp used as the reference time. Primarily useful for
        deterministic tests.
    policy
        Optional freshness policy. When omitted, the workspace policy file is
        loaded, falling back to :class:`FreshnessPolicy` defaults.

    Returns
    -------
    WorkspaceState
        Recommended next action and per-stage status.

    Notes
    -----
    Existing workspaces are inspected using the artifact paths stored in
    ``portfolio.json`` rather than assuming the newest default filenames.
    Dependency changes always invalidate downstream artifacts. Age-based freshness
    then recommends periodic validation, calibration, and live-report refreshes.
    Discovery age is advisory because re-discovery can replace the watchlist.
    """
    root = Path(workspace).expanduser().resolve()
    current_time = time.time() if now is None else float(now)
    freshness = policy or load_freshness_policy(root)
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

    discovery_age = _age_days(watchlist, current_time)
    validation_age = _age_days(validation, current_time) if has_validation else None
    calibration_age = _age_days(calibration, current_time) if has_calibration else None
    report_age = _age_days(report, current_time) if has_report else None

    validation_dependency_stale = has_validation and _mtime(validation) < _mtime(watchlist)
    calibration_dependency_stale = has_calibration and (
        not has_validation or _mtime(calibration) < _mtime(validation)
    )
    report_dependencies = [portfolio, watchlist]
    if has_validation:
        report_dependencies.append(validation)
    if has_calibration:
        report_dependencies.append(calibration)
    report_dependency_stale = has_report and any(
        _mtime(report) < _mtime(path) for path in report_dependencies
    )

    discovery_due = discovery_age > freshness.discovery_days
    validation_due = has_validation and validation_age is not None and validation_age > freshness.validation_days
    calibration_due = (
        has_calibration and calibration_age is not None and calibration_age > freshness.calibration_days
    )
    report_due = has_report and report_age is not None and report_age > freshness.report_days

    discovery_status = _age_status("ready", discovery_age, discovery_due)
    if validation_dependency_stale:
        validation_status = "needs update (watchlist changed)"
    elif has_validation and validation_age is not None:
        validation_status = _age_status("ready", validation_age, validation_due)
    else:
        validation_status = "not run"

    if calibration_dependency_stale:
        calibration_status = "needs update (validation changed)"
    elif has_calibration and calibration_age is not None:
        calibration_status = _age_status("ready", calibration_age, calibration_due)
    else:
        calibration_status = "not run"

    if report_dependency_stale:
        report_status = "needs refresh (workspace changed)"
    elif has_report and report_age is not None:
        report_status = _age_status("ready", report_age, report_due)
    else:
        report_status = "not run"

    discovery_advice = (
        f" Discovery is {discovery_age:.0f} days old; consider re-running it when convenient."
        if discovery_due
        else ""
    )

    if not has_validation or validation_dependency_stale or validation_due:
        if validation_due and not validation_dependency_stale:
            summary = (
                f"Historical basket validation is {validation_age:.0f} days old. Refresh validation, "
                "then calibration and the live report."
            )
        else:
            summary = "Candidate baskets are available. Validate their historical persistence next."
        return WorkspaceState(
            name="needs_validation",
            summary=summary + discovery_advice,
            next_stage="validate",
            next_label="Refresh validation" if has_validation else "Validate baskets",
            update_stages=("validate", "calibrate", "report"),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", validation_status),
                ("Calibration", "waiting for validation" if not has_calibration else calibration_status),
                ("Live report", report_status),
            ),
        )

    if not has_calibration or calibration_dependency_stale or calibration_due:
        if calibration_due and not calibration_dependency_stale:
            summary = (
                f"Basket-specific calibration is {calibration_age:.0f} days old. Refresh calibration "
                "before generating a new live report."
            )
        else:
            summary = "Validation is available. Fit basket-specific probability calibration next."
        return WorkspaceState(
            name="needs_calibration",
            summary=summary + discovery_advice,
            next_stage="calibrate",
            next_label="Refresh calibration" if has_calibration else "Calibrate probabilities",
            update_stages=("calibrate", "report"),
            stage_statuses=(
                ("Discovery", discovery_status),
                ("Validation", validation_status),
                ("Calibration", calibration_status),
                ("Live report", report_status),
            ),
        )

    if not has_report or report_dependency_stale or report_due:
        if report_due and not report_dependency_stale:
            summary = f"The live report is {report_age:.0f} days old. Refresh current prices and recommendations."
        else:
            summary = "The model state is ready. Generate a current report using the latest holdings and prices."
        return WorkspaceState(
            name="needs_report",
            summary=summary + discovery_advice,
            next_stage="report",
            next_label="Refresh recommendations" if has_report else "Generate live report",
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
            "This workspace is current. Refresh recommendations whenever holdings or market prices need updating."
            + discovery_advice
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
