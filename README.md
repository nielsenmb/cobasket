# cobasket

Find cointegrated stock baskets and backtest a mean-reverting spread strategy
against them.

## Install

```bash
pip install -e .
```

(editable install, so code changes take effect immediately -- good for local dev)

## Usage

Backtest a basket you already have in mind:

```bash
cobasket-backtest XOM CVX COP OXY --period 2y
```

Screen the S&P 500 for candidate baskets via correlation clustering (fetches ~500 tickers -- slow the first time, cached afterward):

```bash
cobasket-screen --period 2y --distance-threshold 0.8
```

Screen via PCA instead -- decomposes returns into latent factors, removes the market-wide component(s), clusters on factor loadings, and saves diagnostic plots so you can actually see what's going on:

```bash
cobasket-pca-screen --period 2y --n-remove 1 --distance-threshold 1.5
```

This produces three PNGs each run:
- `pca_scree.png` -- variance explained per PC; the "elbow" tells you how many components carry real structure
- `pca_loadings.png` -- stocks plotted by PC2 vs PC3 loading, colored by cluster; tight separated blobs are good basket candidates, a shapeless cloud means clustering isn't separating well
- `cluster_dendrogram.png` -- the clustering tree with your `--distance-threshold` cut drawn as a red line, so you can see exactly why stocks ended up in the clusters they did

Both commands cache downloaded price data to `price_cache/` in the current
directory. Pass `--force-refresh` to `cobasket-screen` to bypass it, or just
delete the directory.

## Pipeline

Two screening approaches, same downstream confirmation + backtest step:

**Correlation-based (`cobasket-screen`)**
1. Fetch S&P 500 tickers + SPY as a market proxy
2. Regress out the SPY return from each stock (removes the common market
   factor so clustering reflects idiosyncratic co-movement)
3. Hierarchical clustering on 1-correlation distance of the residuals

**PCA-based (`cobasket-pca-screen`)**
1. Fetch S&P 500 tickers, standardize returns to unit variance
2. Run PCA -- lets the data itself define the dominant shared factors,
   rather than assuming SPY is the right proxy (PC1 is almost always
   "the market"; later PCs often correspond to sector/rate/commodity
   exposure)
3. Remove the top `--n-remove` PCs, leaving idiosyncratic residuals
4. Cluster stocks by their loadings (coordinates in PC-space) rather
   than by raw pairwise correlation

**Both then:**
5. Johansen cointegration test on each cluster (candidates capped at 8
   members -- Johansen gets unstable with more series than that)
6. For each confirmed basket: refit weights on the first half of history,
   backtest on the second half, rank by Sharpe ratio

## Caveats

- This is a research/learning tool, not a production trading system: no
  slippage modeling beyond a flat bps cost, no position sizing, no risk
  limits.
- Cointegration relationships can break down; nothing here re-validates a
  basket over time (no walk-forward re-estimation yet).
- `yfinance` can be rate-limited or flaky for large universes -- if
  `cobasket-screen` fails partway through, try a smaller ticker slice first.
