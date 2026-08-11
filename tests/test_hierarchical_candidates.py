"""Tests for hierarchical candidate extraction in broad-universe discovery."""

import numpy as np
import pandas as pd
import pytest

from cobasket.cointegration import _hierarchical_subclusters, cluster_candidates


def test_large_neighborhood_retains_nested_small_candidates() -> None:
    """A neighborhood above the size cap should yield its nested subclusters."""
    linkage_matrix = np.array(
        [
            [0, 1, 0.10, 2],
            [2, 3, 0.10, 2],
            [6, 7, 0.20, 4],
            [4, 5, 0.10, 2],
            [8, 9, 0.30, 6],
        ],
        dtype=float,
    )
    candidates = _hierarchical_subclusters(
        linkage_matrix,
        ("A", "B", "C", "D", "E", "F"),
        distance_threshold=0.5,
        max_basket_size=3,
    )
    candidate_sets = {frozenset(candidate) for candidate in candidates}

    assert candidate_sets == {
        frozenset(("A", "B")),
        frozenset(("C", "D")),
        frozenset(("E", "F")),
    }


def test_cap_controls_which_nested_hierarchy_levels_are_retained() -> None:
    """Increasing the cap should add larger nested candidates without losing pairs."""
    linkage_matrix = np.array(
        [
            [0, 1, 0.10, 2],
            [2, 3, 0.10, 2],
            [6, 7, 0.20, 4],
            [4, 5, 0.10, 2],
            [8, 9, 0.30, 6],
        ],
        dtype=float,
    )
    candidates = _hierarchical_subclusters(
        linkage_matrix,
        ("A", "B", "C", "D", "E", "F"),
        distance_threshold=0.5,
        max_basket_size=4,
    )
    candidate_sets = {frozenset(candidate) for candidate in candidates}

    assert frozenset(("A", "B")) in candidate_sets
    assert frozenset(("A", "B", "C", "D")) in candidate_sets
    assert frozenset(("A", "B", "C", "D", "E", "F")) not in candidate_sets


def test_distance_cut_still_limits_candidate_neighborhoods() -> None:
    """Nested extraction should not bridge linkage nodes above the distance cut."""
    linkage_matrix = np.array(
        [
            [0, 1, 0.10, 2],
            [2, 3, 0.10, 2],
            [4, 5, 0.60, 4],
        ],
        dtype=float,
    )
    candidates = _hierarchical_subclusters(
        linkage_matrix,
        ("A", "B", "C", "D"),
        distance_threshold=0.2,
        max_basket_size=4,
    )
    candidate_sets = {frozenset(candidate) for candidate in candidates}

    assert candidate_sets == {frozenset(("A", "B")), frozenset(("C", "D"))}


def test_cluster_candidates_rejects_invalid_size_cap() -> None:
    """The candidate size cap must allow at least a pair."""
    residuals = pd.DataFrame({"A": [0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="at least 2"):
        cluster_candidates(residuals, max_basket_size=1)
