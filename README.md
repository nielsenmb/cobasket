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

## Data layer

Stage 1 introduces a validated adjusted-price data interface:

```python
from cobasket.data import DataManager

manager = DataManager(cache_dir="price_cache")
prices = manager.prices(
    ["AAPL", "MSFT", "GOOG"],
    period="2y",
)

print(prices.head())
print(manager.last_metadata)
```

The manager downloads adjusted closing prices in batches, stores each ticker in
its own parquet cache file, removes unusable symbols without discarding valid
ones, aligns trading dates, and validates the returned table. Adjusted prices
account for stock splits and dividends, making historical price changes more
comparable through time.

For explicit dates, set `period=None`:

```python
prices = manager.prices(
    ["AAPL", "MSFT"],
    period=None,
    start="2023-01-01",
    end="2025-01-01",
)
```

Run the offline test suite with:

```bash
pip install -e ".[test]"
pytest
```

## Stage 2: spread accounting

CoBasket now treats Johansen weights as a direction rather than as directly
tradable portfolio weights. The weights are normalized by their absolute sum,
then converted into fixed share units whose initial gross exposure is one
capital unit. This makes the backtest dimensionally consistent:

```python
from cobasket.backtest import run_backtest
from cobasket.cointegration import build_spread, johansen_test
from cobasket.signals import zscore_signal

result = johansen_test(prices, verbose=False)
spread, weights = build_spread(prices, result)
z_score, signal = zscore_signal(spread)
backtest = run_backtest(prices, weights, signal, cost_bps=10)

print(backtest.units)
print(backtest.sharpe)
print(backtest.equity.tail())
```

The spread-position convention is:

- `+1`: hold the asset legs in the direction of the fitted weight vector
  (long the spread);
- `-1`: reverse every leg (short the spread);
- `0`: hold no spread position.

The executable notebook in `notebooks/01_data_and_basic_workflow.ipynb` explains
these terms and works through the current data, cointegration, signal, and
backtest APIs.

## Long-only decision support

Cobasket 0.3 begins separating statistical evidence from investment actions.
Cointegration is now treated as one relative-value evidence source rather than a
direct instruction to execute a long/short trade.

```python
from cobasket.data import DataManager
from cobasket.evidence import (
    cointegration_evidence,
    recommendation_table,
    recommend_assets,
)

prices = DataManager().prices(["AAPL", "MSFT"], period="2y")
result = cointegration_evidence(prices, window=60)

recommendations = recommend_assets(
    result.asset_evidence,
    holdings={"AAPL": 2.0},
)
print(recommendation_table(recommendations))
```

A positive evidence score means an asset appears relatively cheap within the
fitted basket. A negative score means relatively expensive. These scores are
bounded diagnostics, **not calibrated probabilities** and not estimates of
intrinsic company value. Recommendation thresholds are configurable decision
rules and should eventually be calibrated using walk-forward, out-of-sample
results.

## Walk-forward calibration and persistent watchlists

Cobasket keeps the **selection universe** separate from the **current portfolio**.
A ticker remains in a `BasketWatchlist` even when its held quantity becomes zero,
so it continues to receive fresh `Buy`, `Watch`, `Wait`, or `Avoid buying`
recommendations and can later become a re-entry candidate.

```python
from cobasket.evidence import (
    BasketWatchlist,
    evaluate_watchlist,
    select_candidate_baskets,
    watchlist_from_candidates,
)

# `universe_prices` contains the stocks considered during initial selection.
candidates = select_candidate_baskets(
    universe_prices,
    market_ticker="SPY",
)
watchlist = watchlist_from_candidates(candidates, top_n=5)
watchlist.save("portfolio_watchlist.json")

# A sold stock remains in the watchlist when its quantity is set to zero.
evaluation = evaluate_watchlist(
    current_prices,
    watchlist,
    holdings={"AAPL": 0.0, "MSFT": 3.0},
)
```

Walk-forward calibration repeatedly fits the model using only information that
would have been available on each historical date, then measures subsequent
relative performance:

```python
from cobasket.evidence import (
    calibrate_evidence,
    fit_probability_calibration,
    walk_forward_evidence,
)

records = walk_forward_evidence(
    basket_prices,
    train_window=252,
    z_window=60,
    horizon=20,
    step=5,
)
calibration = fit_probability_calibration(records, horizon=20)
```

The resulting probability answers a specific relative question: whether the
stock outperforms the equal-weight return of its basket over the selected
horizon. It is not the probability that the stock price rises in absolute terms.
