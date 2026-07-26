# Portfolio and watchlist editor

Install the optional GUI dependencies:

```bash
pip install -e ".[gui]"
```

Launch the main dashboard:

```bash
cobasket-gui
```

After choosing a portfolio configuration JSON file, use **Portfolio → Edit portfolio and watchlist…** to change:

- held quantities;
- available cash;
- historical price period;
- rolling z-score window;
- minimum Johansen trace ratio;
- stale-price warning threshold;
- calibration file;
- monitored baskets.

The editor can also be launched independently:

```bash
cobasket-edit portfolio.json
```

A holding quantity of zero means that the stock is not currently owned. It does not remove the ticker from its basket watchlist. Cobasket will continue evaluating it and may later issue a new buy recommendation.

Removing a holding row changes the portfolio configuration only. Removing a basket from the watchlist stops monitoring that relationship. The editor validates both files before saving, including duplicate holdings, negative quantities, and baskets with fewer than two unique ticker symbols.
