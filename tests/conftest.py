"""Test helpers that keep the suite independent of optional parquet engines."""

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def parquet_backend_for_tests(monkeypatch):
    """Use pickle serialization behind the parquet API in minimal CI images.

    Production installations include pyarrow through the project's declared
    dependencies. This shim tests cobasket's cache behaviour without requiring
    a compiled parquet dependency in the test runner.
    """
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, *a, **k: self.to_pickle(path))
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **k: pd.read_pickle(path))
