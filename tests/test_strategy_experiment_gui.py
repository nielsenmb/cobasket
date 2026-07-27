"""Headless tests for strategy editing and controlled experiment rendering."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from cobasket.gui.strategy_experiment_dialog import (
    StrategyEditorDialog,
    StrategyExperimentDialog,
    parse_conditions,
)
from cobasket.strategy_experiments import ExperimentSplit, StrategyExperimentResult
from cobasket.strategy_rules import MetricCondition, StrategyRule, StrategyRules


def _strategy(name: str = "probability") -> StrategyRules:
    """Return a small ordered strategy for GUI tests."""
    return StrategyRules(
        name=name,
        rules=(
            StrategyRule(
                "sell",
                (MetricCondition("probability", "<=", 0.30),),
                0.0,
            ),
            StrategyRule(
                "buy",
                (
                    MetricCondition("probability", ">=", 0.60),
                    MetricCondition("high_volatility", "==", False),
                ),
                0.10,
            ),
        ),
    )


def test_parse_conditions_supports_numeric_and_boolean_thresholds():
    """The compact condition syntax should preserve types and order."""
    conditions = parse_conditions(
        "probability >= 0.6; high_volatility == False"
    )
    assert [item.metric for item in conditions] == ["probability", "high_volatility"]
    assert conditions[0].threshold == pytest.approx(0.6)
    assert conditions[1].threshold is False


def test_strategy_editor_round_trips_ordered_rules():
    """Populating and reading the editor should preserve rule priority."""
    app = QApplication.instance() or QApplication([])
    original = _strategy()
    dialog = StrategyEditorDialog(original)
    rebuilt = dialog.strategy_from_widgets()
    assert rebuilt.name == original.name
    assert [item.action for item in rebuilt.rules] == ["sell", "buy"]
    assert rebuilt.rules[1].match == "all"
    assert len(rebuilt.rules[1].conditions) == 2
    dialog.close()
    app.processEvents()


def test_strategy_editor_move_changes_first_match_priority():
    """Moving a rule should change the serialized backend order."""
    app = QApplication.instance() or QApplication([])
    dialog = StrategyEditorDialog(_strategy())
    dialog.rules.selectRow(1)
    dialog.move_rule(-1)
    rebuilt = dialog.strategy_from_widgets()
    assert [item.action for item in rebuilt.rules] == ["buy", "sell"]
    dialog.close()
    app.processEvents()


def test_experiment_dialog_renders_train_validation_and_test_tables():
    """A completed experiment should populate all result tabs and warnings."""
    app = QApplication.instance() or QApplication([])
    dialog = StrategyExperimentDialog()
    strategy = _strategy()
    index = pd.date_range("2024-01-01", periods=90, freq="B")
    split = ExperimentSplit.from_fractions(index)
    train = pd.DataFrame(
        {"total_return": [0.10], "sharpe_ratio": [1.0]}, index=[strategy.name]
    )
    validation = pd.DataFrame(
        {"total_return": [0.08], "sharpe_ratio": [0.8]}, index=[strategy.name]
    )
    test = pd.DataFrame(
        {"total_return": [0.04, 0.03, 0.0], "sharpe_ratio": [0.5, 0.4, 0.0]},
        index=[strategy.name, "benchmark_equal_weight", "benchmark_cash"],
    )
    result = StrategyExperimentResult(
        split=split,
        train_table=train,
        validation_table=validation,
        test_table=test,
        selected_strategy=strategy,
        interval_results={"train": {}, "validation": {}, "test": {}},
        warnings=("Example warning",),
    )
    dialog._show_result(result)
    assert dialog.train_table.rowCount() == 1
    assert dialog.validation_table.rowCount() == 1
    assert dialog.test_table.rowCount() == 3
    assert "probability" in dialog.selected_label.text()
    assert dialog.warning_list.count() == 1
    dialog.close()
    app.processEvents()
