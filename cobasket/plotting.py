"""
Plotting helpers. All functions save a PNG and return the path, so they
can be called standalone or chained in a CLI command.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def plot_scree(pca, path="pca_scree.png"):
    """
    Scree plot: variance explained per PC, plus cumulative. The same plot
    you'd make after PCA on any dataset -- look for the 'elbow' to judge
    how many PCs represent real structure vs noise.
    """
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(var_ratio) + 1)
    ax1.bar(x, var_ratio, color="steelblue", alpha=0.7, label="Per-PC variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio", color="steelblue")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="darkorange", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance", color="darkorange")
    ax2.set_ylim(0, 1.05)

    plt.title("PCA scree plot")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loadings_2d(loadings, pc_x="PC2", pc_y="PC3", basket_labels=None, path="pca_loadings.png"):
    """
    Scatter of stock loadings on two PCs (skip PC1 by default -- it's
    almost always the market and just spreads everything along one
    direction; PC2 vs PC3 usually shows more interesting structure).

    basket_labels: optional dict {ticker: cluster_id} to color points by
    their assigned cluster, so you can see visually whether the clusters
    found downstream actually look like sensible groups in loading-space.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    if basket_labels:
        cluster_ids = [basket_labels.get(t, -1) for t in loadings.index]
        scatter = ax.scatter(
            loadings[pc_x], loadings[pc_y], c=cluster_ids, cmap="tab20", s=40
        )
    else:
        ax.scatter(loadings[pc_x], loadings[pc_y], s=40, color="steelblue")

    # label a subset of points so it doesn't get unreadable -- all points
    # if the universe is small, otherwise just the extremes on either axis
    if len(loadings) <= 60:
        to_label = loadings.index
    else:
        extremes = pd.concat([
            loadings[pc_x].nlargest(10), loadings[pc_x].nsmallest(10),
            loadings[pc_y].nlargest(10), loadings[pc_y].nsmallest(10),
        ])
        to_label = extremes.index.unique()

    for ticker in to_label:
        ax.annotate(ticker, (loadings.loc[ticker, pc_x], loadings.loc[ticker, pc_y]),
                    fontsize=7, alpha=0.8)

    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"{pc_x} loading")
    ax.set_ylabel(f"{pc_y} loading")
    ax.set_title(f"Stock loadings: {pc_x} vs {pc_y}")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dendrogram(Z, labels, path="cluster_dendrogram.png", distance_threshold=None):
    """
    Dendrogram of the clustering step (works for either the correlation-
    distance linkage from cointegration.py or the loading-distance linkage
    from pca.py). Draw the cut line if a distance_threshold is given, so
    you can see exactly which clusters resulted from that threshold.
    """
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.15), 6))
    dendrogram(Z, labels=list(labels), ax=ax, leaf_rotation=90, leaf_font_size=6)
    if distance_threshold is not None:
        ax.axhline(distance_threshold, color="r", ls="--", label=f"cut = {distance_threshold}")
        ax.legend()
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_basket_diagnostics(basket_result, path="basket_diagnostics.png"):
    """
    The 4-panel view already used in the single-basket backtest CLI
    command (prices / spread / z-score / equity curve), pulled out here
    so it can be reused from anywhere.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    basket_result["trading_prices"].plot(ax=axes[0], title="Prices (trading window)")
    basket_result["trading_spread"].plot(ax=axes[1], title="Spread (frozen weights)")
    basket_result["z"].plot(ax=axes[2], title="Z-score / signal")
    axes[2].axhline(2, color="r", ls="--")
    axes[2].axhline(-2, color="r", ls="--")
    basket_result["equity"].plot(ax=axes[3], title="Strategy equity curve (starting at 1.0)")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path
