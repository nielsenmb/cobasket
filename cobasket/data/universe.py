"""Ticker-universe retrieval helpers with explicit market/currency metadata."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from urllib.request import Request, urlopen
import warnings

import pandas as pd

from .exceptions import DownloadError


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
FTSE100_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
FTSE250_URL = "https://www.lse.co.uk/indices/ftse-250/constituents.html"
EUROSTOXX50_URL = "https://en.wikipedia.org/wiki/EURO_STOXX_50"


@dataclass(frozen=True)
class UniverseSpec:
    """Resolved ticker universe and its quote conventions.

    Parameters
    ----------
    name
        Stable universe identifier.
    tickers
        Yahoo-compatible ticker symbols.
    market_ticker
        Market proxy used when removing common-market returns.
    quote_currency
        Currency/unit used by downloaded quotes, for example ``USD`` or ``GBp``.
    analysis_currency
        Monetary unit used for portfolio values after ``price_scale`` is applied.
    price_scale
        Factor converting quote values to ``analysis_currency``.
    """

    name: str
    tickers: tuple[str, ...]
    market_ticker: str
    quote_currency: str
    analysis_currency: str
    price_scale: float = 1.0


def _download_tables(url: str, label: str) -> list[pd.DataFrame]:
    """Download HTML tables with a browser-like user agent.

    Parameters
    ----------
    url
        Source webpage containing constituent tables.
    label
        Human-readable universe label used in error messages.

    Returns
    -------
    list of pandas.DataFrame
        Parsed HTML tables.

    Raises
    ------
    DownloadError
        If the source cannot be retrieved or parsed.
    """
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/145.0 Safari/537.36 Cobasket/0.5"
            )
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
        return pd.read_html(StringIO(html))
    except Exception as exc:
        raise DownloadError(f"failed to retrieve {label} constituents: {exc}") from exc


def _download_sp500_table() -> pd.DataFrame:
    """Download the S&P 500 constituent table.

    Returns
    -------
    pandas.DataFrame
        Constituent table containing a ``Symbol`` column.

    Raises
    ------
    DownloadError
        If no downloaded table contains a ``Symbol`` column.
    """
    for table in _download_tables(SP500_URL, "sp500"):
        if "Symbol" in table.columns:
            return table
    raise DownloadError("sp500 constituent page did not contain a Symbol column")


def _normalize_cached_tickers(cache_path: Path, transform) -> list[str]:
    """Normalize cached ticker symbols and update stale cache entries in place.

    Parameters
    ----------
    cache_path
        CSV cache containing a ``ticker`` column.
    transform
        Callable converting a cached symbol to current Yahoo-compatible notation.

    Returns
    -------
    list of str
        De-duplicated normalized ticker symbols.
    """
    raw = pd.read_csv(cache_path)["ticker"].astype(str).tolist()
    tickers = list(dict.fromkeys(transform(value.strip()) for value in raw if value.strip()))
    if tickers != raw:
        pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers


def _cached_tickers(
    *,
    name: str,
    url: str,
    column_candidates: tuple[str, ...],
    cache_dir: str | Path,
    force_refresh: bool,
    transform,
) -> list[str]:
    """Retrieve one HTML-table universe with a normalized local constituent cache."""
    cache_path = Path(cache_dir) / f"{name}_tickers.csv"
    if cache_path.exists() and not force_refresh:
        return _normalize_cached_tickers(cache_path, transform)

    try:
        tables = _download_tables(url, name)
        series = None
        for table in tables:
            for column in column_candidates:
                if column in table.columns:
                    series = table[column]
                    break
            if series is not None:
                break
        if series is None:
            raise DownloadError(f"{name} constituent page did not contain a ticker column")
        tickers = [transform(str(value).strip()) for value in series if str(value).strip()]
        tickers = list(dict.fromkeys(ticker for ticker in tickers if ticker))
    except DownloadError:
        if cache_path.exists():
            warnings.warn(
                f"Could not refresh {name} constituents; using the existing cached universe.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _normalize_cached_tickers(cache_path, transform)
        raise

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers


def _yahoo_us_ticker(value: str) -> str:
    """Normalize a US ticker to Yahoo share-class notation.

    Parameters
    ----------
    value
        Exchange ticker symbol.

    Returns
    -------
    str
        Yahoo-compatible ticker symbol.
    """
    return str(value).strip().upper().replace(".", "-")


def _yahoo_london_ticker(value: str) -> str:
    """Convert an LSE EPIC symbol or stale cached ticker to Yahoo notation.

    Parameters
    ----------
    value
        LSE ticker/EPIC symbol or an existing Yahoo-style London symbol.

    Returns
    -------
    str
        Yahoo-compatible London ticker.
    """
    symbol = str(value).strip().upper()
    base = symbol[:-2] if symbol.endswith(".L") else symbol
    base = base.rstrip(".")
    return f"{base.replace('.', '-')}.L"


def _yahoo_ftse250_share(value: str) -> str:
    """Extract an EPIC code from a FTSE 250 share label and convert it for Yahoo.

    Parameters
    ----------
    value
        Constituent label such as ``"3i Infrastructure (3IN)"`` or a bare EPIC.

    Returns
    -------
    str
        Yahoo-compatible London ticker.
    """
    text = str(value).strip()
    match = re.search(r"\(([^()]+)\)\s*$", text)
    return _yahoo_london_ticker(match.group(1) if match else text)


def get_sp500_tickers(
    cache_dir: str | Path = "price_cache",
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Return current S&P 500 Yahoo-compatible symbols."""
    cache_path = Path(cache_dir) / "sp500_tickers.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)["ticker"].astype(str).tolist()
    try:
        table = _download_sp500_table()
        tickers = table["Symbol"].astype(str).map(_yahoo_us_ticker).tolist()
    except DownloadError:
        if cache_path.exists():
            warnings.warn(
                "Could not refresh sp500 constituents; using the existing cached universe.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.read_csv(cache_path)["ticker"].astype(str).tolist()
        raise
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers


def get_sp400_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return current S&P MidCap 400 Yahoo-compatible symbols."""
    return _cached_tickers(
        name="sp400",
        url=SP400_URL,
        column_candidates=("Symbol", "Ticker"),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=_yahoo_us_ticker,
    )


def get_sp600_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return current S&P SmallCap 600 Yahoo-compatible symbols."""
    return _cached_tickers(
        name="sp600",
        url=SP600_URL,
        column_candidates=("Symbol", "Ticker"),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=_yahoo_us_ticker,
    )


def get_sp1500_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return the S&P Composite 1500 as the union of its 500/400/600 components."""
    groups = (
        get_sp500_tickers(cache_dir, force_refresh=force_refresh),
        get_sp400_tickers(cache_dir, force_refresh=force_refresh),
        get_sp600_tickers(cache_dir, force_refresh=force_refresh),
    )
    return list(dict.fromkeys(ticker for group in groups for ticker in group))


def get_nasdaq100_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return current Nasdaq-100 Yahoo-compatible symbols."""
    return _cached_tickers(
        name="nasdaq100",
        url=NASDAQ100_URL,
        column_candidates=("Ticker", "Symbol"),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=_yahoo_us_ticker,
    )


def get_ftse100_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return current FTSE 100 symbols using Yahoo's ``.L`` notation."""
    return _cached_tickers(
        name="ftse100",
        url=FTSE100_URL,
        column_candidates=("Ticker", "EPIC"),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=_yahoo_london_ticker,
    )


def get_ftse250_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return current FTSE 250 symbols using Yahoo's ``.L`` notation."""
    return _cached_tickers(
        name="ftse250",
        url=FTSE250_URL,
        column_candidates=("Ticker", "EPIC", "Code", "Share"),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=_yahoo_ftse250_share,
    )


def get_ftse350_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return the FTSE 350 as the union of FTSE 100 and FTSE 250 constituents."""
    groups = (
        get_ftse100_tickers(cache_dir, force_refresh=force_refresh),
        get_ftse250_tickers(cache_dir, force_refresh=force_refresh),
    )
    return list(dict.fromkeys(ticker for group in groups for ticker in group))


def get_eurostoxx50_tickers(cache_dir: str | Path = "price_cache", *, force_refresh: bool = False) -> list[str]:
    """Return EURO STOXX 50 main-listing Yahoo symbols."""
    return _cached_tickers(
        name="eurostoxx50",
        url=EUROSTOXX50_URL,
        column_candidates=("Ticker",),
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        transform=lambda value: value,
    )


def load_custom_tickers(path: str | Path) -> list[str]:
    """Load a custom ticker universe from CSV or newline-delimited text.

    Parameters
    ----------
    path
        CSV containing a ``ticker`` column, or a text file with one ticker per line.

    Returns
    -------
    list of str
        Normalized ticker symbols.
    """
    source = Path(path).expanduser()
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
        if "ticker" not in table.columns:
            raise ValueError("custom universe CSV must contain a 'ticker' column")
        values = table["ticker"].astype(str)
    else:
        values = pd.Series(source.read_text(encoding="utf-8").splitlines(), dtype=str)
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))


def get_universe(
    name: str,
    *,
    cache_dir: str | Path = "price_cache",
    force_refresh: bool = False,
    custom_path: str | Path | None = None,
    custom_market_ticker: str | None = None,
    custom_currency: str | None = None,
    custom_price_scale: float = 1.0,
) -> UniverseSpec:
    """Resolve a built-in or custom single-currency discovery universe.

    Parameters
    ----------
    name
        Built-in universe name, or ``custom``.
    cache_dir
        Constituent-cache directory.
    force_refresh
        Refresh online constituent lists.
    custom_path
        Required ticker file for ``custom``.
    custom_market_ticker
        Required market proxy for ``custom``.
    custom_currency
        Required quote/analysis currency label for ``custom``.
    custom_price_scale
        Quote-to-analysis currency scale for ``custom``.

    Returns
    -------
    UniverseSpec
        Resolved ticker list and market/currency conventions.
    """
    key = str(name).strip().lower()
    if key == "sp500":
        return UniverseSpec(key, tuple(get_sp500_tickers(cache_dir, force_refresh=force_refresh)), "SPY", "USD", "USD")
    if key == "sp400":
        return UniverseSpec(key, tuple(get_sp400_tickers(cache_dir, force_refresh=force_refresh)), "^SP400", "USD", "USD")
    if key == "sp600":
        return UniverseSpec(key, tuple(get_sp600_tickers(cache_dir, force_refresh=force_refresh)), "^SP600", "USD", "USD")
    if key == "sp1500":
        return UniverseSpec(key, tuple(get_sp1500_tickers(cache_dir, force_refresh=force_refresh)), "^SP1500", "USD", "USD")
    if key == "nasdaq100":
        return UniverseSpec(key, tuple(get_nasdaq100_tickers(cache_dir, force_refresh=force_refresh)), "QQQ", "USD", "USD")
    if key == "ftse100":
        return UniverseSpec(key, tuple(get_ftse100_tickers(cache_dir, force_refresh=force_refresh)), "^FTSE", "GBp", "GBP", 0.01)
    if key == "ftse250":
        return UniverseSpec(key, tuple(get_ftse250_tickers(cache_dir, force_refresh=force_refresh)), "^FTMC", "GBp", "GBP", 0.01)
    if key == "ftse350":
        return UniverseSpec(key, tuple(get_ftse350_tickers(cache_dir, force_refresh=force_refresh)), "^FTLC", "GBp", "GBP", 0.01)
    if key == "eurostoxx50":
        return UniverseSpec(key, tuple(get_eurostoxx50_tickers(cache_dir, force_refresh=force_refresh)), "^STOXX50E", "EUR", "EUR")
    if key == "custom":
        if custom_path is None or custom_market_ticker is None or custom_currency is None:
            raise ValueError("custom universe requires --tickers-file, --market-ticker, and --currency")
        currency = str(custom_currency).strip().upper()
        return UniverseSpec(
            key,
            tuple(load_custom_tickers(custom_path)),
            str(custom_market_ticker).strip().upper(),
            currency,
            currency,
            float(custom_price_scale),
        )
    raise ValueError(f"unknown universe: {name}")
