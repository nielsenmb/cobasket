"""High-level adjusted-price download and cache management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .cache import PriceCache
from .cleaning import align_prices, clean_prices
from .exceptions import DownloadError, ValidationError
from .validation import validate_prices

Downloader = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class PriceMetadata:
    """Provenance information for the most recent request."""

    requested_tickers: tuple[str, ...]
    returned_tickers: tuple[str, ...]
    failed_tickers: tuple[str, ...]
    cache_hits: tuple[str, ...]
    downloaded_tickers: tuple[str, ...]
    requested_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "Yahoo Finance via yfinance"
    adjusted: bool = True


class DataManager:
    """Fetch, cache, clean, align, and validate adjusted closing prices."""

    def __init__(
        self,
        cache_dir: str | Path = "price_cache",
        *,
        downloader: Downloader | None = None,
        cache_max_age_days: float | None = 1.0,
        download_batch_size: int = 100,
    ) -> None:
        """Initialize a data manager.

        Parameters
        ----------
        cache_dir
            Root directory used for per-ticker cache files.
        downloader
            Optional function compatible with ``yfinance.download``. This is
            primarily useful for tests and alternative data providers.
        cache_max_age_days
            Maximum reusable cache age. ``None`` disables age expiry.
        download_batch_size
            Maximum number of uncached tickers requested in one call.

        Raises
        ------
        ValueError
            If ``download_batch_size`` is less than one.
        ImportError
            If live downloading is requested but ``yfinance`` is unavailable.
        """
        if download_batch_size < 1:
            raise ValueError("download_batch_size must be at least 1")

        self.cache = PriceCache(cache_dir)
        if downloader is None:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise ImportError(
                    "yfinance is required for live downloads; install cobasket dependencies"
                ) from exc
            downloader = yf.download
        self.downloader = downloader
        self.cache_max_age_days = cache_max_age_days
        self.download_batch_size = download_batch_size
        self.last_metadata: PriceMetadata | None = None

    def prices(
        self,
        tickers: Sequence[str],
        period: str | None = "2y",
        *,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
        min_coverage: float = 1.0,
    ) -> pd.DataFrame:
        """Return an aligned adjusted-close table for a set of tickers.

        Parameters
        ----------
        tickers
            Asset symbols to retrieve.
        period
            Relative history specification accepted by ``yfinance``. Set to
            ``None`` when using explicit dates.
        start, end
            Optional explicit date bounds.
        force_refresh
            Ignore reusable cache files when ``True``.
        min_coverage
            Minimum fraction of dates required for each retained ticker.

        Returns
        -------
        pandas.DataFrame
            Clean, aligned, validated adjusted closing prices.

        Raises
        ------
        ValueError
            If tickers or date-range arguments are invalid.
        DownloadError
            If no usable symbols can be retrieved.
        ValidationError
            If downloaded data fail validation or coverage requirements.

        Notes
        -----
        Cached symbols are loaded independently. Uncached symbols are downloaded
        in batches and then split into one cache file per symbol. If a provider
        rejects an entire multi-symbol batch, Cobasket retries those symbols one
        at a time so one stale constituent cannot discard the rest of the batch.
        """
        normalized = self._normalize_tickers(tickers)
        self._validate_range(period=period, start=start, end=end)

        frames: dict[str, pd.DataFrame] = {}
        cache_hits: list[str] = []
        downloaded: list[str] = []
        failed: list[str] = []
        pending: list[str] = []

        for ticker in normalized:
            path = self.cache.path_for(ticker, period=period, start=start, end=end)
            if not force_refresh and self.cache.is_fresh(
                path, max_age_days=self.cache_max_age_days
            ):
                frame = clean_prices(self.cache.read(path))
                if ticker in frame.columns and not frame[[ticker]].dropna().empty:
                    frames[ticker] = frame[[ticker]]
                    cache_hits.append(ticker)
                    continue
            pending.append(ticker)

        for batch in self._batches(pending, self.download_batch_size):
            try:
                raw = self._download_batch(batch, period=period, start=start, end=end)
            except DownloadError:
                if len(batch) == 1:
                    failed.extend(batch)
                    continue
                self._retry_individually(
                    batch,
                    frames=frames,
                    downloaded=downloaded,
                    failed=failed,
                    period=period,
                    start=start,
                    end=end,
                )
                continue

            for ticker in batch:
                frame = self._validated_ticker_frame(raw, ticker)
                if frame is None:
                    failed.append(ticker)
                    continue
                path = self.cache.path_for(ticker, period=period, start=start, end=end)
                self.cache.write(path, frame)
                frames[ticker] = frame[[ticker]]
                downloaded.append(ticker)

        ordered_frames = [frames[ticker] for ticker in normalized if ticker in frames]
        if not ordered_frames:
            self._set_metadata(normalized, (), failed, cache_hits, downloaded)
            raise DownloadError(
                "no usable price data were returned for: " + ", ".join(normalized)
            )

        combined = clean_prices(pd.concat(ordered_frames, axis=1, join="outer"))
        combined = align_prices(combined, min_coverage=min_coverage)
        if combined.shape[1] == 0:
            self._set_metadata(normalized, (), failed, cache_hits, downloaded)
            raise ValidationError("all downloaded tickers failed the coverage requirement")

        validate_prices(combined)
        returned = tuple(combined.columns)
        failed = list(dict.fromkeys([*failed, *(t for t in normalized if t not in returned)]))
        self._set_metadata(normalized, returned, failed, cache_hits, downloaded)
        return combined

    def _retry_individually(
        self,
        tickers: Sequence[str],
        *,
        frames: dict[str, pd.DataFrame],
        downloaded: list[str],
        failed: list[str],
        period: str | None,
        start: str | None,
        end: str | None,
    ) -> None:
        """Retry a provider-rejected batch one ticker at a time.

        Parameters
        ----------
        tickers
            Symbols from the rejected batch.
        frames
            Mutable mapping receiving successful price frames.
        downloaded
            Mutable list receiving successful downloaded symbols.
        failed
            Mutable list receiving unavailable or invalid symbols.
        period
            Relative history specification.
        start, end
            Optional explicit date bounds.
        """
        for ticker in tickers:
            try:
                raw = self._download_batch((ticker,), period=period, start=start, end=end)
            except DownloadError:
                failed.append(ticker)
                continue
            frame = self._validated_ticker_frame(raw, ticker)
            if frame is None:
                failed.append(ticker)
                continue
            path = self.cache.path_for(ticker, period=period, start=start, end=end)
            self.cache.write(path, frame)
            frames[ticker] = frame[[ticker]]
            downloaded.append(ticker)

    @staticmethod
    def _validated_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        """Extract and validate one downloaded ticker frame.

        Parameters
        ----------
        raw
            Provider response containing the requested ticker.
        ticker
            Symbol to extract.

        Returns
        -------
        pandas.DataFrame or None
            Valid single-ticker frame, or ``None`` if the symbol is unavailable
            or fails price validation.
        """
        try:
            frame = clean_prices(DataManager._extract_close(raw, ticker))
            if frame.empty or ticker not in frame.columns:
                raise DownloadError(
                    f"download returned no adjusted closing prices for {ticker}"
                )
            validate_prices(frame, allow_missing=True)
        except (DownloadError, ValidationError):
            return None
        return frame

    def _download_batch(
        self,
        tickers: Sequence[str],
        *,
        period: str | None,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Download one batch of symbols from the configured provider.

        Parameters
        ----------
        tickers
            Symbols included in the request.
        period
            Relative history specification.
        start, end
            Optional explicit date bounds.

        Returns
        -------
        pandas.DataFrame
            Raw provider response.

        Raises
        ------
        DownloadError
            If the provider raises an exception or returns an empty table.
        """
        kwargs: dict[str, object] = {
            "tickers": list(tickers) if len(tickers) > 1 else tickers[0],
            "auto_adjust": True,
            "progress": False,
            "group_by": "column",
            "threads": True,
        }
        if period is not None:
            kwargs["period"] = period
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end

        try:
            raw = self.downloader(**kwargs)
        except Exception as exc:
            joined = ", ".join(tickers)
            raise DownloadError(f"download failed for {joined}: {exc}") from exc
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise DownloadError("download returned an empty table")
        return raw

    @staticmethod
    def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract one symbol's closing-price series from provider layouts.

        Parameters
        ----------
        raw
            Raw table returned by ``yfinance.download`` or a compatible provider.
        ticker
            Symbol to extract.

        Returns
        -------
        pandas.DataFrame
            Single-column price table named after ``ticker``.

        Raises
        ------
        DownloadError
            If a closing-price field or requested ticker cannot be identified.
        """
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise DownloadError(f"download returned an empty table for {ticker}")

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            level1 = raw.columns.get_level_values(1)
            if "Close" in level0:
                close = raw["Close"]
            elif "Close" in level1:
                close = raw.xs("Close", axis=1, level=1)
            else:
                raise DownloadError(f"download did not contain a Close field for {ticker}")
        elif "Close" in raw.columns:
            close = raw[["Close"]].rename(columns={"Close": ticker})
        elif len(raw.columns) == 1:
            close = raw.copy()
        else:
            raise DownloadError(f"download did not contain a Close field for {ticker}")

        if isinstance(close, pd.Series):
            close = close.to_frame(name=ticker)
        elif ticker in close.columns:
            close = close[[ticker]]
        elif len(close.columns) == 1:
            close = close.copy()
            close.columns = [ticker]
        else:
            raise DownloadError(f"could not identify {ticker} in downloaded Close data")
        return close

    @staticmethod
    def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
        """Normalize, deduplicate, and validate ticker symbols.

        Parameters
        ----------
        tickers
            User-supplied ticker sequence.

        Returns
        -------
        tuple of str
            Uppercase symbols in first-occurrence order.

        Raises
        ------
        ValueError
            If no non-empty symbols are supplied.
        """
        normalized = tuple(
            dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip())
        )
        if not normalized:
            raise ValueError("at least one ticker is required")
        return normalized

    @staticmethod
    def _validate_range(
        *, period: str | None, start: str | None, end: str | None
    ) -> None:
        """Validate mutually exclusive relative and explicit date ranges.

        Parameters
        ----------
        period
            Relative history specification.
        start, end
            Optional explicit date bounds.

        Raises
        ------
        ValueError
            If both range styles are mixed or an explicit range lacks ``start``.
        """
        if period is not None and (start is not None or end is not None):
            raise ValueError("specify either period or start/end, not both")
        if period is None and start is None:
            raise ValueError("start is required when period is None")

    @staticmethod
    def _batches(items: Sequence[str], size: int):
        """Yield fixed-size tuples from a sequence.

        Parameters
        ----------
        items
            Symbols to divide into batches.
        size
            Maximum batch length.

        Yields
        ------
        tuple of str
            Consecutive symbol batches.
        """
        for start in range(0, len(items), size):
            yield tuple(items[start : start + size])

    def _set_metadata(
        self,
        requested: Sequence[str],
        returned: Sequence[str],
        failed: Sequence[str],
        cache_hits: Sequence[str],
        downloaded: Sequence[str],
    ) -> None:
        """Store provenance metadata for the most recent request.

        Parameters
        ----------
        requested
            Requested symbols.
        returned
            Symbols present in the final aligned table.
        failed
            Symbols that failed download, validation, or alignment.
        cache_hits
            Symbols loaded from cache.
        downloaded
            Symbols retrieved from the provider.
        """
        self.last_metadata = PriceMetadata(
            requested_tickers=tuple(requested),
            returned_tickers=tuple(returned),
            failed_tickers=tuple(dict.fromkeys(failed)),
            cache_hits=tuple(cache_hits),
            downloaded_tickers=tuple(downloaded),
        )
