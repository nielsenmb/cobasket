"""Trading 212 instrument metadata helpers for tradeability filtering."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from .exceptions import DownloadError


_LIVE_URL = "https://live.trading212.com/api/v0/equity/metadata/instruments"
_DEMO_URL = "https://demo.trading212.com/api/v0/equity/metadata/instruments"


def _normalized_symbol(value: str) -> str:
    """Normalize Yahoo or broker short symbols for conservative matching.

    Parameters
    ----------
    value
        Yahoo ticker or Trading 212 ``shortName``.

    Returns
    -------
    str
        Upper-case alphanumeric symbol with Yahoo exchange suffixes removed.
    """
    symbol = str(value).strip().upper()
    if "." in symbol:
        head, tail = symbol.rsplit(".", 1)
        if tail in {"L", "DE", "PA", "AS", "MI", "MC", "BR", "HE", "VI", "LS", "IR", "SW", "ST", "CO", "OL", "WA"}:
            symbol = head
    return re.sub(r"[^A-Z0-9]", "", symbol)


def _currency_family(value: str) -> str:
    """Normalize quote-currency aliases used by brokers and Yahoo.

    Parameters
    ----------
    value
        Currency or quote-unit label.

    Returns
    -------
    str
        Normalized currency family.
    """
    currency = str(value).strip().upper()
    return "GBP" if currency in {"GBP", "GBX", "GBPENCE", "GBPENNY"} else currency


def load_trading212_instruments(
    cache_dir: str | Path = "price_cache",
    *,
    force_refresh: bool = False,
    environment: str = "live",
    max_cache_age_seconds: float = 3600.0,
) -> list[dict[str, object]]:
    """Retrieve account-accessible Trading 212 instruments.

    Credentials are read from ``TRADING212_API_KEY`` and
    ``TRADING212_API_SECRET``. The response is cached because the metadata
    endpoint is rate limited.

    Parameters
    ----------
    cache_dir
        Directory used for the local metadata cache.
    force_refresh
        Ignore a fresh cache and query Trading 212 again.
    environment
        ``"live"`` or ``"demo"``.
    max_cache_age_seconds
        Maximum cache age before a refresh is attempted.

    Returns
    -------
    list of dict
        Trading 212 instrument metadata records.

    Raises
    ------
    DownloadError
        If credentials are unavailable and no usable cache exists, or if the
        metadata request fails.
    """
    env = str(environment).strip().lower()
    if env not in {"live", "demo"}:
        raise ValueError("Trading 212 environment must be 'live' or 'demo'")
    cache_path = Path(cache_dir) / f"trading212_{env}_instruments.json"
    if cache_path.exists() and not force_refresh:
        age = time.time() - cache_path.stat().st_mtime
        if age <= max_cache_age_seconds:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(payload)

    api_key = os.getenv("TRADING212_API_KEY")
    api_secret = os.getenv("TRADING212_API_SECRET")
    if not api_key or not api_secret:
        if cache_path.exists():
            return list(json.loads(cache_path.read_text(encoding="utf-8")))
        raise DownloadError(
            "Trading 212 filtering requires TRADING212_API_KEY and TRADING212_API_SECRET "
            "environment variables (or an existing instrument cache)."
        )

    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
    request = Request(
        _LIVE_URL if env == "live" else _DEMO_URL,
        headers={"Authorization": f"Basic {token}", "User-Agent": "Cobasket/0.5"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if cache_path.exists():
            return list(json.loads(cache_path.read_text(encoding="utf-8")))
        raise DownloadError(f"failed to retrieve Trading 212 instruments: {exc}") from exc
    if not isinstance(payload, list):
        raise DownloadError("Trading 212 instruments endpoint returned an unexpected payload")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def filter_trading212_tickers(
    tickers: tuple[str, ...] | list[str],
    *,
    analysis_currency: str,
    instruments: list[dict[str, object]],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Keep Yahoo tickers that match accessible Trading 212 stocks.

    Parameters
    ----------
    tickers
        Yahoo-compatible symbols.
    analysis_currency
        Currency family expected for the universe.
    instruments
        Records returned by :func:`load_trading212_instruments`.

    Returns
    -------
    filtered
        Input tickers with an accessible Trading 212 stock match.
    broker_tickers
        Mapping from Yahoo ticker to Trading 212's unique ticker identifier.
    """
    currency = _currency_family(analysis_currency)
    matches: dict[str, list[str]] = {}
    for item in instruments:
        if str(item.get("type", "")).upper() != "STOCK":
            continue
        if _currency_family(str(item.get("currencyCode", ""))) != currency:
            continue
        short_name = item.get("shortName")
        broker_ticker = item.get("ticker")
        if not short_name or not broker_ticker:
            continue
        matches.setdefault(_normalized_symbol(str(short_name)), []).append(str(broker_ticker))

    kept: list[str] = []
    mapping: dict[str, str] = {}
    for ticker in tickers:
        candidates = matches.get(_normalized_symbol(ticker), [])
        if not candidates:
            continue
        kept.append(ticker)
        mapping[ticker] = candidates[0]
    return tuple(kept), mapping
