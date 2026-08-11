# Persistence-aware basket discovery

`cobasket-discover` is the recommended basket-discovery command when runtime of several minutes or longer is acceptable. It first finds currently related clusters, then checks whether the fitted relationship survives independent historical windows before ranking candidates.

```bash
cobasket-discover \
    --universe sp500 \
    --period 5y \
    --top-n 20 \
    --watchlist-out screened_watchlist.json \
    --table-out discovery_results.csv
```

Built-in universes are `sp500`, `nasdaq100`, `ftse100`, and `eurostoxx50`. A custom single-currency universe can be supplied with `--universe custom --tickers-file ... --market-ticker ... --currency ...`.

The detailed output records current Johansen strength, historical persistence, sign-invariant weight stability, preliminary Sharpe ratio, return, maximum drawdown, and trade count. A basket must pass the persistence and weight-stability thresholds before it is written to the watchlist.

## Currency conventions

Discovery is deliberately single-currency. S&P 500 and Nasdaq-100 use USD. EURO STOXX 50 uses EUR. FTSE 100 equities are normally quoted by Yahoo in GBp (pence); the generated watchlist metadata records a scale of `0.01` to GBP for later monetary reporting.

Cross-currency custom discovery is not inferred automatically. Run separate single-currency universes or pre-convert the price series to one currency before performing cross-market research. This avoids allowing FX variation to masquerade as a stock-spread relationship.

Cobasket's statistical screening primarily uses returns and normalized spread weights, so absolute currency units are not themselves trading signals. Portfolio monetary totals are a separate concern: values in USD, GBP, and EUR must not be added without an explicit FX conversion policy.
