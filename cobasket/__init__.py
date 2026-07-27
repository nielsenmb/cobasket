"""Transparent cointegration research and long-only decision support."""

from cobasket._version import __version__
from cobasket.continuous_walk_forward import (
    ContinuousDeploymentConfig,
    ContinuousWalkForwardResult,
    run_continuous_walk_forward,
)
from cobasket.data import DataManager
from cobasket.price_metrics import PriceMetricConfig, build_price_metrics
from cobasket.repeated_walk_forward import (
    RepeatedWalkForwardResult,
    WalkForwardConfig,
    run_repeated_walk_forward,
)
from cobasket.strategy_experiments import (
    ExperimentSplit,
    StrategyExperimentConfig,
    StrategyExperimentResult,
    run_strategy_experiment,
)
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules
from cobasket.workflow import PortfolioAnalyzer, PortfolioConfig, PortfolioReport, TickerReport

__all__ = [
    "__version__",
    "ContinuousDeploymentConfig",
    "ContinuousWalkForwardResult",
    "DataManager",
    "ExperimentSplit",
    "MetricCondition",
    "PortfolioAnalyzer",
    "PortfolioConfig",
    "PortfolioReport",
    "PriceMetricConfig",
    "RepeatedWalkForwardResult",
    "StrategyExperimentConfig",
    "StrategyExperimentResult",
    "StrategyRule",
    "StrategyRules",
    "TickerReport",
    "WalkForwardConfig",
    "build_price_metrics",
    "run_continuous_walk_forward",
    "run_repeated_walk_forward",
    "run_strategy_experiment",
]
