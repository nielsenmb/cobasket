# Recommendation and decision history

Cobasket stores each successful live analysis in a local SQLite database named
`cobasket_history.sqlite` beside the selected portfolio configuration file.

The history database keeps three distinct records:

1. **Model snapshots** — probabilities, credible intervals, evidence scores,
   holdings, prices, and recommendations from each report.
2. **User actions** — what you actually did, including optional quantity, price,
   and a note explaining the decision.
3. **Later outcomes** — forward returns measured after configurable trading-day
   horizons.

Keeping these separate avoids rewriting history. A model recommendation remains
what it was even when you choose not to act on it.

## Using the GUI

Run a fresh analysis from the dashboard to create a snapshot. Then open:

```
Portfolio -> Recommendation history...
```

The selected ticker is used automatically when possible. The window shows:

- probability and credible-interval history;
- evidence-score history;
- recommendation transitions such as `Hold -> Buy`;
- quantities held at each report;
- your recorded actions and notes;
- later forward returns when outcome data have been added.

A zero holding remains valid. It allows the history to show a later re-entry
recommendation after a complete sale.

## Updating outcomes

Outcome calculation is deliberately separate from report generation because the
future prices do not yet exist when a recommendation is made.

```python
from cobasket.data import DataManager
from cobasket.history import RecommendationHistoryStore

prices = DataManager().prices(["AAPL", "MSFT"], period="5y")
store = RecommendationHistoryStore("cobasket_history.sqlite")
store.update_outcomes(prices, horizons=(5, 20, 60))
```

The horizons count trading observations, not calendar days. Repeating the update
is safe: existing outcomes are not duplicated.

The SQLite file contains personal portfolio history and is excluded by the
repository `.gitignore`.
