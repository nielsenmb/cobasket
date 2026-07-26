"""Reusable plotting helpers for PCA and basket diagnostics."""

from __future__ import annotations

from os import PathLike
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def plot_scree(pca, path: str | PathLike[str] = "pca_scree.png"):
    """Save a scree plot of per-component and cumulative PCA variance.

    Parameters
    ----------
    pca
        Fitted scikit-learn PCA estimator.
    path
        Output image path.

    Returns
    -------
    str or os.PathLike
        The supplied output path.
    """
    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)

    figure, primary_axis = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(variance_ratio) + 1)
    primary_axis.bar(x, variance_ratio, alpha=0.7, label="Per-PC variance")
    primary_axis.set_xlabel("Principal component")
    primary_axis.set_ylabel("Explained variance ratio")
    primary_axis.set_xticks(x)

    secondary_axis = primary_axis.twinx()
    secondary_axis.plot(x, cumulative, marker="o", label="Cumulative")
    secondary_axis.set_ylabel("Cumulative explained variance")
    secondary_axis.set_ylim(0, 1.05)

    primary_axis.set_title("PCA scree plot")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_loadings_2d(
    loadings: pd.DataFrame,
    pc_x: str = "PC2",
    pc_y: str = "PC3",
    basket_labels: Mapping[str, int] | None = None,
    path: str | PathLike[str] = "pca_loadings.png",
):
    """Save a two-dimensional scatter plot of PCA asset loadings.

    Parameters
    ----------
    loadings
        Assets in rows and principal-component loadings in columns.
    pc_x, pc_y
        Component names plotted on the horizontal and vertical axes.
    basket_labels
        Optional mapping from ticker symbol to cluster identifier.
    path
        Output image path.

    Returns
    -------
    str or os.PathLike
        The supplied output path.
    """
    figure, axis = plt.subplots(figsize=(9, 9))

    if basket_labels:
        cluster_ids = [basket_labels.get(str(ticker), -1) for ticker in loadings.index]
        axis.scatter(loadings[pc_x], loadings[pc_y], c=cluster_ids, s=40)
    else:
        axis.scatter(loadings[pc_x], loadings[pc_y], s=40)

    if len(loadings) <= 60:
        labels = loadings.index
    else:
        extremes = pd.concat(
            [
                loadings[pc_x].nlargest(10),
                loadings[pc_x].nsmallest(10),
                loadings[pc_y].nlargest(10),
                loadings[pc_y].nsmallest(10),
            ]
        )
        labels = extremes.index.unique()

    for ticker in labels:
        axis.annotate(
            ticker,
            (loadings.loc[ticker, pc_x], loadings.loc[ticker, pc_y]),
            fontsize=7,
            alpha=0.8,
        )

    axis.axhline(0, linewidth=0.5)
    axis.axvline(0, linewidth=0.5)
    axis.set_xlabel(f"{pc_x} loading")
    axis.set_ylabel(f"{pc_y} loading")
    axis.set_title(f"Stock loadings: {pc_x} vs {pc_y}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: Sequence[str],
    path: str | PathLike[str] = "cluster_dendrogram.png",
    distance_threshold: float | None = None,
):
    """Save a hierarchical-clustering dendrogram.

    Parameters
    ----------
    linkage_matrix
        SciPy linkage matrix.
    labels
        Labels associated with the original observations.
    path
        Output image path.
    distance_threshold
        Optional clustering cut height drawn as a horizontal line.

    Returns
    -------
    str or os.PathLike
        The supplied output path.
    """
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.15), 6))
    dendrogram(
        linkage_matrix,
        labels=list(labels),
        ax=axis,
        leaf_rotation=90,
        leaf_font_size=6,
    )
    if distance_threshold is not None:
        axis.axhline(
            distance_threshold,
            linestyle="--",
            label=f"cut = {distance_threshold}",
        )
        axis.legend()
    axis.set_title("Hierarchical clustering dendrogram")
    axis.set_ylabel("Distance")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_basket_diagnostics(
    basket_result: Mapping[str, object],
    path: str | PathLike[str] = "basket_diagnostics.png",
):
    """Save prices, spread, z-score, and equity diagnostics for a basket.

    Parameters
    ----------
    basket_result
        Dictionary returned by ``backtest_single_basket``.
    path
        Output image path.

    Returns
    -------
    str or os.PathLike
        The supplied output path.
    """
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    basket_result["trading_prices"].plot(ax=axes[0], title="Prices (trading window)")
    basket_result["trading_spread"].plot(
        ax=axes[1],
        title="Spread (frozen weights)",
    )
    basket_result["z"].plot(ax=axes[2], title="Z-score / signal")
    axes[2].axhline(2, linestyle="--")
    axes[2].axhline(-2, linestyle="--")
    basket_result["equity"].plot(
        ax=axes[3],
        title="Strategy equity curve (starting at 1.0)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
