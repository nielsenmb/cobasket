# Changelog

Cobasket follows semantic versioning while it remains a research-oriented beta.

## 0.5.0 - Unreleased

### Added

- Live portfolio and persistent watchlist reporting.
- Calibrated long-only recommendations with uncertainty and warnings.
- PyQt portfolio, investigation, history, validation, and strategy interfaces.
- Declarative trading rules with explicit first-match priority.
- Momentum, trend, volatility, and basket-robustness metrics.
- Controlled train/validation/test strategy experiments.
- Repeated and continuous walk-forward evaluation.
- Continuous account simulation with retained or liquidated fold boundaries.
- Example portfolio and watchlist configuration files.
- Stable top-level public imports and release smoke tests.

### Changed

- Package version is now defined in one source file and read by the build system.
- Documentation is organized around live use, investigation, and historical validation.

### Notes

Cobasket remains a research and learning tool. The 0.5.0 release does not place orders, connect to a brokerage account, or provide financial advice.

## 0.4.0

- Added the end-to-end GUI and historical validation workflows developed through Stages 7-10.

## 0.3.0

- Added long-only evidence, persistent watchlists, probability calibration, and live reports.

## 0.2.0

- Added dimensionally consistent spread accounting and statistical tests.

## 0.1.0

- Initial data, screening, cointegration, and spread-backtesting prototype.
