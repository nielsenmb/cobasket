"""Path helpers for persistent Cobasket portfolio configurations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cobasket.workflow import PortfolioConfig


def resolve_portfolio_config_paths(
    config: PortfolioConfig,
    config_path: str | Path,
) -> PortfolioConfig:
    """Resolve configuration file references relative to ``portfolio.json``.

    Parameters
    ----------
    config
        Loaded portfolio configuration.
    config_path
        Path from which the configuration was loaded.

    Returns
    -------
    PortfolioConfig
        Equivalent immutable configuration with absolute watchlist and optional
        calibration paths.
    """
    base = Path(config_path).expanduser().resolve().parent

    def resolve(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base / path).resolve())

    return replace(
        config,
        watchlist_path=resolve(config.watchlist_path),
        calibration_path=resolve(config.calibration_path) if config.calibration_path else None,
    )
