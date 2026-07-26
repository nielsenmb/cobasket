"""Per-ticker Parquet caching for adjusted price data."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .exceptions import CacheError


def request_key(*, period: str | None, start: str | None, end: str | None) -> str:
    """Construct a stable short identifier for a date-range request.

    Parameters
    ----------
    period
        Relative history specification accepted by ``yfinance``.
    start, end
        Optional explicit date bounds.

    Returns
    -------
    str
        Twelve-character SHA-256 prefix for the request parameters.
    """
    payload = json.dumps({"period": period, "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class PriceCache:
    """Read and write one Parquet file per ticker and request range."""

    def __init__(self, directory: str | Path = "price_cache") -> None:
        """Initialize a price cache.

        Parameters
        ----------
        directory
            Root directory for cache files.
        """
        self.directory = Path(directory)

    def path_for(
        self,
        ticker: str,
        *,
        period: str | None,
        start: str | None,
        end: str | None,
    ) -> Path:
        """Return the cache path for one ticker and date request.

        Parameters
        ----------
        ticker
            Asset symbol.
        period
            Relative history specification.
        start, end
            Optional explicit date bounds.

        Returns
        -------
        pathlib.Path
            Deterministic Parquet-file path.
        """
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.upper())
        key = request_key(period=period, start=start, end=end)
        return self.directory / "prices" / safe_ticker / f"{key}.parquet"

    def is_fresh(self, path: Path, *, max_age_days: float | None) -> bool:
        """Check whether a cache file exists and satisfies an age limit.

        Parameters
        ----------
        path
            Candidate cache file.
        max_age_days
            Maximum permitted age in days. ``None`` accepts any existing file.

        Returns
        -------
        bool
            ``True`` when the file can be reused.
        """
        if not path.exists():
            return False
        if max_age_days is None:
            return True
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return age_seconds <= max_age_days * 86400

    def read(self, path: Path) -> pd.DataFrame:
        """Read a cached price table.

        Parameters
        ----------
        path
            Parquet file to read.

        Returns
        -------
        pandas.DataFrame
            Cached price table.

        Raises
        ------
        CacheError
            If the file cannot be decoded.
        """
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise CacheError(f"failed to read cache file {path}: {exc}") from exc

    def write(self, path: Path, data: pd.DataFrame) -> None:
        """Write a price table to a Parquet cache file.

        Parameters
        ----------
        path
            Destination file.
        data
            Price table to serialize.

        Raises
        ------
        CacheError
            If the destination cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path)
        except Exception as exc:
            raise CacheError(f"failed to write cache file {path}: {exc}") from exc
