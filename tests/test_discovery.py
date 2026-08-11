"""Tests for persistence-aware basket discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

import cobasket.discovery as discovery


def test_discovery_rejects_transient_high_sharpe_candidate(monkeypatch):
    """A high preliminary Sharpe should not rescue a historically transient basket."""
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 130, len(index)),
            "BBB": np.linspace(80, 110, len(index)),
            "CCC": np.linspace(50, 75, len(index)),
            "DDD": np.linspace(60, 90, len(index)),
            "SPY": np.linspace(200, 260, len(index)),
        },
        index=index,
    )
    confirmed = [
        (["AAA", "BBB"], object(), 20.0, 10.0),
        (["CCC", "DDD"], object(), 20.0, 10.0),
    ]
    monkeypatch.setattr(discovery, "screen_universe", lambda *args, **kwargs: (confirmed, prices, None, None))
    monkeypatch.setattr(
        discovery,
        "rank_confirmed_baskets",
        lambda *args, **kwargs: [
            {
                "basket": ["AAA", "BBB"], "sharpe": 2.0, "total_return": 0.5,
                "max_drawdown": -0.3, "n_trades": 20, "johansen_stat": 20.0, "johansen_crit": 10.0,
            },
            {
                "basket": ["CCC", "DDD"], "sharpe": 0.8, "total_return": 0.2,
                "max_drawdown": -0.1, "n_trades": 15, "johansen_stat": 20.0, "johansen_crit": 10.0,
            },
        ],
    )

    def fake_metrics(prices, basket, **kwargs):
        if tuple(basket) == ("AAA", "BBB"):
            return {"accepted_evaluations": 2, "possible_evaluations": 20, "persistence": 0.10, "weight_stability": 0.95}
        return {"accepted_evaluations": 12, "possible_evaluations": 20, "persistence": 0.60, "weight_stability": 0.90}

    monkeypatch.setattr(discovery, "_persistence_metrics", fake_metrics)
    result = discovery.discover_baskets(["AAA", "BBB", "CCC", "DDD"])
    assert tuple(result.table.iloc[0]["basket"]) == ("CCC", "DDD")
    assert bool(result.table.iloc[0]["usable"])
    transient = result.table[result.table["basket"].apply(tuple) == ("AAA", "BBB")].iloc[0]
    assert not bool(transient["usable"])
