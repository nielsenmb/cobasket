"""Foreign-exchange helpers for portfolio valuation.

FX conversion is deliberately kept outside the statistical basket analysis. Basket
prices stay in their native analysis currency; these helpers are used only when a
monetary value must be reported in a common portfolio currency.
"""

from __future__ import annotations

from dataclasses import dataclass

from cobasket.data import DataManager


@dataclass(frozen=True)
class FXQuote:
    """Latest conversion from one currency into another.

    Parameters
    ----------
    source_currency
        Currency of the value being converted.
    target_currency
        Portfolio/base currency.
    rate
        Units of ``target_currency`` per one unit of ``source_currency``.
    ticker
        Yahoo Finance FX symbol used to obtain the rate, or ``None`` when no
        conversion is required.
    """

    source_currency: str
    target_currency: str
    rate: float
    ticker: str | None = None


def normalize_currency(currency: str) -> str:
    """Normalize and validate a three-letter ISO-style currency code.

    Parameters
    ----------
    currency
        Currency code such as ``USD``, ``GBP``, or ``EUR``.

    Returns
    -------
    str
        Upper-case currency code.

    Raises
    ------
    ValueError
        If the supplied code is not exactly three alphabetic characters.
    """
    code = str(currency).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError("currency must be a three-letter alphabetic code")
    return code


def yahoo_fx_ticker(source_currency: str, target_currency: str) -> str:
    """Return Yahoo Finance's direct FX symbol for a currency pair.

    Parameters
    ----------
    source_currency
        Currency being converted from.
    target_currency
        Currency being converted to.

    Returns
    -------
    str
        Yahoo Finance symbol such as ``USDGBP=X``.
    """
    source = normalize_currency(source_currency)
    target = normalize_currency(target_currency)
    return f"{source}{target}=X"


def latest_fx_quote(
    data_manager: DataManager,
    source_currency: str,
    target_currency: str,
    *,
    force_refresh: bool = False,
) -> FXQuote:
    """Fetch the latest FX conversion needed for portfolio valuation.

    Parameters
    ----------
    data_manager
        Cobasket data manager used for cached Yahoo Finance retrieval.
    source_currency
        Currency of the native asset value.
    target_currency
        Desired portfolio/base currency.
    force_refresh
        Bypass reusable FX price caches when ``True``.

    Returns
    -------
    FXQuote
        Latest direct conversion rate. The rate is one when source and target
        currencies are identical.

    Raises
    ------
    ValueError
        If the returned FX rate is non-positive.
    """
    source = normalize_currency(source_currency)
    target = normalize_currency(target_currency)
    if source == target:
        return FXQuote(source, target, 1.0, None)

    ticker = yahoo_fx_ticker(source, target)
    prices = data_manager.prices(
        [ticker],
        period="5d",
        force_refresh=force_refresh,
        min_coverage=1.0,
    )
    rate = float(prices[ticker].iloc[-1])
    if rate <= 0.0:
        raise ValueError(f"non-positive FX rate returned for {ticker}")
    return FXQuote(source, target, rate, ticker)
