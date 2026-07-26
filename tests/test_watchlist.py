"""Tests for stock selection and persistent watchlist evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobasket.evidence import (
    BasketCandidate,
    BasketWatchlist,
    evaluate_watchlist,
    select_candidate_baskets,
    watchlist_from_candidates,
)


def synthetic_universe(n: int = 420, seed: int = 91) -> pd.DataFrame:
    """Return a small universe containing one cointegrated group."""
    rng = np.random.default_rng(seed)
    common = 80.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    for index in range(1, n):
        e1[index] = 0.7 * e1[index - 1] + rng.normal(0.0, 0.35)
        e2[index] = 0.6 * e2[index - 1] + rng.normal(0.0, 0.40)
    independent = 60.0 + np.cumsum(rng.normal(0.04, 0.9, n))
    return pd.DataFrame(
        {
            "AAA": common + e1 + 20.0,
            "BBB": 1.1 * common + e2 + 15.0,
            "CCC": independent + 20.0,
        },
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )


def test_watchlist_round_trip_and_zero_holding_reentry(tmp_path) -> None:
    watchlist = BasketWatchlist(baskets=(("aaa", "bbb"),), name="test")
    path = watchlist.save(tmp_path / "watchlist.json")
    loaded = BasketWatchlist.load(path)
    assert loaded.tickers == ("AAA", "BBB")

    prices = synthetic_universe().loc[:, ["AAA", "BBB"]]
    evaluation = evaluate_watchlist(
        prices,
        loaded,
        holdings={"AAA": 0.0, "BBB": 1.0},
        window=40,
        min_trace_ratio=0.5,
    )
    by_ticker = {item.ticker: item for item in evaluation.recommendations}
    assert set(by_ticker) == {"AAA", "BBB"}
    assert by_ticker["AAA"].currently_held is False
    assert by_ticker["BBB"].currently_held is True


def test_watchlist_from_candidates() -> None:
    candidates = (
        BasketCandidate(("AAA", "BBB"), 2.0, 2),
        BasketCandidate(("CCC", "DDD"), 1.5, 2),
    )
    watchlist = watchlist_from_candidates(candidates, top_n=1)
    assert watchlist.baskets == (("AAA", "BBB"),)


def test_initial_candidate_selection_finds_a_basket() -> None:
    candidates = select_candidate_baskets(
        synthetic_universe(),
        distance_threshold=1.5,
        min_trace_ratio=0.5,
        max_basket_size=3,
    )
    assert candidates
    assert all(len(candidate.tickers) >= 2 for candidate in candidates)
