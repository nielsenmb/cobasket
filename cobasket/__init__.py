"""Transparent cointegration research and long-only decision support."""

from cobasket._version import __version__
from cobasket.basket_validation import (
    BasketValidationProfile,
    BasketValidationSet,
    BasketValidationThresholds,
    validate_watchlist_baskets,
    validation_table,
)
from cobasket.calibration_workflow import WatchlistCalibrationResult, calibrate_watchlist
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
    "BasketValidationProfile",
    "BasketValidationSet",
    "BasketValidationThresholds",
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
    "WatchlistCalibrationResult",
    "build_price_metrics",
    "calibrate_watchlist",
    "run_continuous_walk_forward",
    "run_repeated_walk_forward",
    "run_strategy_experiment",
    "validate_watchlist_baskets",
    "validation_table",
]
