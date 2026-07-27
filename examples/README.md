# Example configuration

`portfolio.json` and `portfolio_watchlist.json` form a minimal starting point for the live reporting and GUI workflows.

Copy both files into the same working directory:

```bash
cp examples/portfolio.json .
cp examples/portfolio_watchlist.json .
```

Then edit:

- `holdings`: the number of shares currently owned; use `0.0` to keep monitoring a ticker without holding it;
- `cash`: uninvested cash used for portfolio context;
- `baskets`: groups of at least two tickers that Cobasket should evaluate together;
- analysis windows and warning thresholds as needed.

Generate a report with:

```bash
cobasket-report \
    --portfolio portfolio.json \
    --watchlist portfolio_watchlist.json \
    --output report.json
```

The included ticker group is illustrative, not a claim that the assets are currently cointegrated or suitable investments.
