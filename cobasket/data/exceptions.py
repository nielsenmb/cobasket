"""Exceptions raised by the :mod:`cobasket.data` package."""


class CobasketDataError(Exception):
    """Base class for data-layer failures."""


class DownloadError(CobasketDataError):
    """Raised when price data cannot be downloaded or interpreted."""


class CacheError(CobasketDataError):
    """Raised when cached data cannot be read or written."""


class ValidationError(CobasketDataError):
    """Raised when a price table violates the expected data contract."""
