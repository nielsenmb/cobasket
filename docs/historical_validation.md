# Historical policy and calibration validation

Open **Portfolio → Historical validation…** from `cobasket-gui`.

The dialog accepts:

- a price table indexed by date with one column per ticker;
- a probability table indexed by evaluation date with the same ticker columns;
- an optional walk-forward outcomes table containing `probability_outperform` and `outperformed`.

CSV and Parquet are supported for the indexed price and probability tables. The optional outcomes table is CSV.

The performance view shows:

- policy equity against an equal-weight buy-and-hold benchmark;
- drawdown from the previous equity peak;
- the fraction of the portfolio invested rather than held as cash;
- a reliability diagram comparing predicted probabilities with observed frequencies.

The summary bar reports total return, annualized return, Sharpe ratio, maximum drawdown, trade count, Brier score, and expected calibration error. The trade-history tab lists every simulated buy and sell, including the probability that triggered it and the transaction cost.

The backtest acts on each recommendation at the next available price observation. This avoids using a closing price to trade on information that was only calculated at that same close.

The equal-weight curve is a simple reference, not a claim that it is the optimal benchmark. A policy should be judged on several properties at once: return, drawdown, turnover, calibration, and robustness to different historical windows.
