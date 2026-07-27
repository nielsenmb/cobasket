"""Release, public API, example, and notebook smoke tests."""

from __future__ import annotations

import ast
from importlib.metadata import entry_points, version
import json
from pathlib import Path

import nbformat

import cobasket
from cobasket.evidence import BasketWatchlist
from cobasket.workflow import PortfolioConfig


ROOT = Path(__file__).resolve().parents[1]


def test_installed_version_matches_public_version() -> None:
    """Ensure package metadata and the imported version cannot diverge."""
    assert version("cobasket") == cobasket.__version__


def test_supported_public_api_is_importable() -> None:
    """Check the documented top-level research interfaces."""
    expected = {
        "DataManager",
        "PortfolioAnalyzer",
        "PortfolioConfig",
        "StrategyRules",
        "StrategyRule",
        "MetricCondition",
        "PriceMetricConfig",
        "WalkForwardConfig",
        "ContinuousDeploymentConfig",
        "run_strategy_experiment",
        "run_repeated_walk_forward",
        "run_continuous_walk_forward",
    }
    assert expected.issubset(set(cobasket.__all__))
    assert all(hasattr(cobasket, name) for name in expected)


def test_console_entry_points_are_registered() -> None:
    """Verify every documented command is included in package metadata."""
    names = {item.name for item in entry_points(group="console_scripts")}
    assert {
        "cobasket-backtest",
        "cobasket-screen",
        "cobasket-pca-screen",
        "cobasket-report",
        "cobasket-gui",
        "cobasket-edit",
    }.issubset(names)


def test_example_configuration_files_load() -> None:
    """Keep the shipped example state synchronized with backend schemas."""
    portfolio = PortfolioConfig.load(ROOT / "examples" / "portfolio.json")
    watchlist = BasketWatchlist.load(ROOT / "examples" / "portfolio_watchlist.json")
    assert portfolio.holdings
    assert watchlist.baskets
    assert set(portfolio.holdings).issubset(set(watchlist.tickers))


def test_all_notebooks_are_valid_and_code_cells_parse() -> None:
    """Parse every notebook and syntax-check ordinary Python code cells."""
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells, f"{path.name} has no cells"
        for number, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code":
                continue
            lines = [
                line
                for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            source = "\n".join(lines).strip()
            if source:
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    raise AssertionError(
                        f"{path.name} code cell {number} is not valid Python: {exc}"
                    ) from exc


def test_example_json_is_human_readable() -> None:
    """Ensure example files remain conventional formatted JSON."""
    for path in sorted((ROOT / "examples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload
        assert path.read_text(encoding="utf-8").endswith("\n")
