# Ticker and basket investigation

The dashboard can inspect the monitored baskets supporting a ticker recommendation.

1. Load a portfolio configuration and run an analysis.
2. Select a ticker in the recommendation table.
3. Double-click the row, or choose **Portfolio → Investigate selected ticker…**.
4. Select one of the ticker's monitored baskets and click **Load diagnostics**.

Cobasket downloads the configured price history and shows four views:

- **Normalized prices** compare relative price changes after dividing each series by its first value. They show co-movement without the original currency scales.
- **Fitted spread** is the weighted combination estimated by the Johansen test. A stable relation should fluctuate rather than drift persistently.
- **Rolling spread z-score** expresses the current displacement in local standard deviations. It is the quantity used by the relative-value evidence model.
- **Cointegration weights** show how each ticker contributes to the fitted spread. Positive and negative signs describe opposite directions in the synthetic relation; they are not long-only portfolio allocations.

The dialog also reports the latest z-score and the Johansen trace ratio. A trace ratio above one means the rank-zero statistic exceeds its 95% critical value for the current sample. This is evidence for cointegration, not a guarantee that the relation will persist or be profitable.

Price downloads and fitting run in a background thread so the dashboard remains responsive. The investigation view does not submit trades.
