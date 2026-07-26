# Cobasket dashboard

Stage 7A adds a read-only PyQt dashboard for the live portfolio-report backend.
It does not place orders or alter a Trading 212 account.

## Install

```bash
pip install -e ".[gui]"
```

## Launch

```bash
cobasket-gui
```

The dashboard accepts either of two JSON files:

1. A `PortfolioConfig` file. Choose **Run analysis** to download/update prices and
   generate a fresh report in a background thread.
2. A previously exported `PortfolioReport`. Choose **Load report** to inspect it
   without downloading market data.

The main table shows current holdings, prices, market value, calibrated
probability, recommendation, and warning count. Selecting a row displays the
plain-language explanation, basket memberships, credible interval, and
per-ticker warnings.

Probabilities remain relative: they estimate whether a ticker will outperform
its associated basket over the calibrated horizon. They are not probabilities
that the price rises in absolute terms.

## Scope

This first dashboard intentionally remains read-only. Editing holdings,
investigating basket plots, calibration charts, and backtest controls belong to
later GUI stages. Keeping those features separate prevents financial logic from
being embedded directly in widget callbacks.
