"""Per-ticker parquet caching."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .exceptions import CacheError


def request_key(*, period: str | None, start: str | None, end: str | None) -> str:
    """Return a short stable identifier for a requested time range."""
    payload = json.dumps({"period": period, "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class PriceCache:
    """Read and write one parquet file per ticker and request range."""

    def __init__(self, directory: str | Path = "price_cache") -> None:
        self.directory = Path(directory)

    def path_for(
        self,
        ticker: str,
        *,
        period: str | None,
        start: str | None,
        end: str | None,
    ) -> Path:
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.upper())
        key = request_key(period=period, start=start, end=end)
        return self.directory / "prices" / safe_ticker / f"{key}.parquet"

    def is_fresh(self, path: Path, *, max_age_days: float | None) -> bool:
        if not path.exists():
            return False
        if max_age_days is None:
            return True
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return age_seconds <= max_age_days * 86400

    def read(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pandas/pyarrow expose several exception types
            raise CacheError(f"failed to read cache file {path}: {exc}") from exc

    def write(self, path: Path, data: pd.DataFrame) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path)
        except Exception as exc:
            raise CacheError(f"failed to write cache file {path}: {exc}") from exc
