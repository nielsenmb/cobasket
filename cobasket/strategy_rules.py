"""Declarative trading rules and leakage-aware historical simulation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cobasket.evidence.policy_backtest import PolicyBacktestResult, _performance_metrics


_OPERATORS = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}


@dataclass(frozen=True)
class MetricCondition:
    """One threshold condition applied to a named strategy metric.

    Parameters
    ----------
    metric
        Metric name in the evaluation context.
    operator
        Comparison operator: ``>``, ``>=``, ``<``, ``<=``, ``==``, or ``!=``.
    threshold
        Numeric or boolean comparison value.
    """

    metric: str
    operator: str
    threshold: float | bool

    def __post_init__(self) -> None:
        """Validate the metric name and comparison operator."""
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.operator not in _OPERATORS:
            raise ValueError(f"unsupported operator: {self.operator}")

    def evaluate(self, context: Mapping[str, float | bool]) -> bool:
        """Evaluate the condition against one ticker-date context.

        Parameters
        ----------
        context
            Mapping of metric names to values available at the decision date.

        Returns
        -------
        bool
            ``True`` when the metric is present, finite, and passes the comparison.
        """
        if self.metric not in context:
            return False
        value = context[self.metric]
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return False
        return bool(_OPERATORS[self.operator](value, self.threshold))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible condition representation."""
        return {"metric": self.metric, "operator": self.operator, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetricCondition":
        """Construct a condition from a JSON-compatible mapping."""
        return cls(
            metric=str(payload["metric"]),
            operator=str(payload["operator"]),
            threshold=payload["threshold"],
        )


@dataclass(frozen=True)
class StrategyRule:
    """An ordered action triggered by one or more metric conditions.

    Parameters
    ----------
    action
        Human-readable action such as ``buy``, ``reduce``, or ``sell``.
    conditions
        Conditions tested for the rule.
    target_weight
        Desired portfolio fraction. ``None`` preserves the current weight.
    match
        ``all`` requires every condition; ``any`` requires at least one.
    """

    action: str
    conditions: tuple[MetricCondition, ...]
    target_weight: float | None
    match: str = "all"

    def __post_init__(self) -> None:
        """Validate action, condition matching, and target allocation."""
        if not self.action.strip():
            raise ValueError("action must not be empty")
        if not self.conditions:
            raise ValueError("a strategy rule requires at least one condition")
        if self.match not in {"all", "any"}:
            raise ValueError("match must be 'all' or 'any'")
        if self.target_weight is not None and not 0.0 <= self.target_weight <= 1.0:
            raise ValueError("target_weight must lie in [0, 1]")

    def matches(self, context: Mapping[str, float | bool]) -> bool:
        """Return whether this rule matches one metric context."""
        outcomes = [condition.evaluate(context) for condition in self.conditions]
        return all(outcomes) if self.match == "all" else any(outcomes)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible rule representation."""
        return {
            "action": self.action,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "target_weight": self.target_weight,
            "match": self.match,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StrategyRule":
        """Construct a rule from a JSON-compatible mapping."""
        return cls(
            action=str(payload["action"]),
            conditions=tuple(MetricCondition.from_dict(item) for item in payload["conditions"]),
            target_weight=(
                None if payload.get("target_weight") is None else float(payload["target_weight"])
            ),
            match=str(payload.get("match", "all")),
        )


@dataclass(frozen=True)
class StrategyRules:
    """Ordered declarative strategy with first-match priority.

    Parameters
    ----------
    name
        Human-readable strategy identifier.
    rules
        Rules in descending priority. The first matching rule is applied.
    default_action
        Action label used when no rule matches.
    transaction_cost_bps
        One-way transaction cost in basis points of traded notional.
    minimum_trade_value
        Trades below this cash value are ignored.
    """

    name: str
    rules: tuple[StrategyRule, ...]
    default_action: str = "hold"
    transaction_cost_bps: float = 10.0
    minimum_trade_value: float = 1.0

    def __post_init__(self) -> None:
        """Validate strategy identity and trading-cost assumptions."""
        if not self.name.strip():
            raise ValueError("strategy name must not be empty")
        if not self.rules:
            raise ValueError("strategy must contain at least one rule")
        if self.transaction_cost_bps < 0 or self.minimum_trade_value < 0:
            raise ValueError("cost settings must be non-negative")

    def decide(
        self,
        context: Mapping[str, float | bool],
        current_weight: float,
    ) -> tuple[str, float]:
        """Return the first matching action and desired asset weight.

        Parameters
        ----------
        context
            Metric values available for one ticker at one decision date.
        current_weight
            Current fraction of portfolio value allocated to the ticker.

        Returns
        -------
        tuple of str and float
            Action label and desired portfolio weight.
        """
        for rule in self.rules:
            if rule.matches(context):
                target = current_weight if rule.target_weight is None else rule.target_weight
                return rule.action, float(target)
        return self.default_action, float(current_weight)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible strategy representation."""
        return {
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
            "default_action": self.default_action,
            "transaction_cost_bps": self.transaction_cost_bps,
            "minimum_trade_value": self.minimum_trade_value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StrategyRules":
        """Construct a strategy from a JSON-compatible mapping."""
        return cls(
            name=str(payload["name"]),
            rules=tuple(StrategyRule.from_dict(item) for item in payload["rules"]),
            default_action=str(payload.get("default_action", "hold")),
            transaction_cost_bps=float(payload.get("transaction_cost_bps", 10.0)),
            minimum_trade_value=float(payload.get("minimum_trade_value", 1.0)),
        )

    def save(self, path: str | Path) -> Path:
        """Write the strategy to a human-readable JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "StrategyRules":
        """Load a strategy from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RuleBacktestResult:
    """Outputs from a declarative-rule portfolio simulation.

    Parameters
    ----------
    backtest
        Portfolio accounting result.
    decisions
        Rule decisions made at each historical evaluation date.
    strategy_name
        Name of the evaluated strategy.
    """

    backtest: PolicyBacktestResult
    decisions: pd.DataFrame
    strategy_name: str


def _prepare_metrics(
    metrics: Mapping[str, pd.DataFrame],
    tickers: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Validate and align metric tables without filling future information."""
    if not metrics:
        raise ValueError("at least one metric table is required")
    prepared: dict[str, pd.DataFrame] = {}
    dates = pd.DatetimeIndex([])
    for name, table in metrics.items():
        if not name.strip():
            raise ValueError("metric names must not be empty")
        frame = table.astype(float).sort_index().reindex(columns=tickers)
        prepared[name] = frame
        dates = dates.union(pd.DatetimeIndex(frame.index))
    return prepared, dates.sort_values()


def run_rule_strategy_backtest(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategy: StrategyRules,
    *,
    initial_cash: float = 10_000.0,
    periods_per_year: int = 252,
) -> RuleBacktestResult:
    """Simulate an ordered rule strategy using next-observation execution.

    Metric tables are read only on their indexed decision dates. Decisions are
    executed at the next available price observation, preventing same-bar
    look-ahead. Missing metric values make their conditions fail.

    Parameters
    ----------
    prices
        Positive aligned prices with dates in rows and tickers in columns.
    metrics
        Mapping from metric names to ticker-by-date tables.
    strategy
        Ordered declarative strategy.
    initial_cash
        Starting uninvested capital.
    periods_per_year
        Number of observations used for annualized metrics.

    Returns
    -------
    RuleBacktestResult
        Portfolio accounting and the complete decision history.
    """
    clean = prices.astype(float).sort_index().dropna(how="any")
    if clean.empty or (clean <= 0.0).any().any():
        raise ValueError("prices must contain positive aligned observations")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    tickers = list(clean.columns)
    prepared, evaluation_dates = _prepare_metrics(metrics, tickers)
    evaluation_set = set(evaluation_dates)
    shares = pd.Series(0.0, index=tickers)
    cash_value = float(initial_cash)
    pending: pd.DataFrame | None = None
    equity_rows: list[float] = []
    cash_rows: list[float] = []
    position_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for date, price in clean.iterrows():
        if pending is not None:
            portfolio_before = cash_value + float((shares * price).sum())
            current_values = shares * price
            desired = pending["target_weight"].astype(float).copy()
            if float(desired.sum()) > 1.0:
                desired /= float(desired.sum())
            trade_values = desired * portfolio_before - current_values
            for ticker in sorted(tickers, key=lambda item: trade_values.loc[item]):
                trade_value = float(trade_values.loc[ticker])
                if abs(trade_value) < strategy.minimum_trade_value:
                    continue
                if trade_value > 0.0:
                    denominator = 1.0 + strategy.transaction_cost_bps / 10_000.0
                    trade_value = min(trade_value, cash_value / denominator)
                quantity = trade_value / float(price.loc[ticker])
                cost = abs(trade_value) * strategy.transaction_cost_bps / 10_000.0
                shares.loc[ticker] += quantity
                cash_value -= trade_value + cost
                trades.append({
                    "date": date,
                    "ticker": ticker,
                    "side": "buy" if trade_value > 0 else "sell",
                    "quantity": abs(quantity),
                    "price": float(price.loc[ticker]),
                    "notional": abs(trade_value),
                    "cost": cost,
                    "action": str(pending.loc[ticker, "action"]),
                })
            pending = None

        equity_value = cash_value + float((shares * price).sum())
        current_weights = shares * price / equity_value
        equity_rows.append(equity_value)
        cash_rows.append(cash_value)
        position_rows.append(shares.copy())
        weight_rows.append(current_weights.copy())

        if date in evaluation_set:
            rows: list[dict[str, object]] = []
            for ticker in tickers:
                context: dict[str, float | bool] = {
                    name: float(frame.loc[date, ticker])
                    for name, frame in prepared.items()
                    if date in frame.index and pd.notna(frame.loc[date, ticker])
                }
                context["current_weight"] = float(current_weights.loc[ticker])
                context["is_held"] = bool(current_weights.loc[ticker] > 0.0)
                action, target = strategy.decide(context, float(current_weights.loc[ticker]))
                rows.append({"ticker": ticker, "action": action, "target_weight": target})
                decisions.append({
                    "date": date,
                    "ticker": ticker,
                    "action": action,
                    "target_weight": target,
                    **context,
                })
            pending = pd.DataFrame(rows).set_index("ticker").reindex(tickers)

    index = clean.index
    equity = pd.Series(equity_rows, index=index, name="equity")
    cash = pd.Series(cash_rows, index=index, name="cash")
    positions = pd.DataFrame(position_rows, index=index)
    weights = pd.DataFrame(weight_rows, index=index)
    trade_table = pd.DataFrame(
        trades,
        columns=["date", "ticker", "side", "quantity", "price", "notional", "cost", "action"],
    )
    metric_summary = _performance_metrics(equity, periods_per_year)
    metric_summary["transaction_costs"] = (
        float(trade_table["cost"].sum()) if not trade_table.empty else 0.0
    )
    metric_summary["trade_count"] = float(len(trade_table))
    backtest = PolicyBacktestResult(equity, cash, positions, weights, trade_table, metric_summary)
    return RuleBacktestResult(backtest, pd.DataFrame(decisions), strategy.name)


def compare_rule_strategies(
    prices: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategyRules],
    *,
    initial_cash: float = 10_000.0,
) -> tuple[pd.DataFrame, dict[str, RuleBacktestResult]]:
    """Run several rule sets under identical data and capital assumptions.

    Parameters
    ----------
    prices
        Historical price table.
    metrics
        Historical metric tables shared by all strategies.
    strategies
        Distinct rule sets to compare.
    initial_cash
        Starting capital supplied independently to each simulation.

    Returns
    -------
    pandas.DataFrame
        One performance-summary row per strategy.
    dict
        Detailed results keyed by strategy name.
    """
    if not strategies:
        raise ValueError("at least one strategy is required")
    if len({strategy.name for strategy in strategies}) != len(strategies):
        raise ValueError("strategy names must be unique")
    results = {
        strategy.name: run_rule_strategy_backtest(
            prices, metrics, strategy, initial_cash=initial_cash
        )
        for strategy in strategies
    }
    rows = []
    for name, result in results.items():
        rows.append({"strategy": name, **dict(result.backtest.metrics)})
    return pd.DataFrame(rows).set_index("strategy"), results
