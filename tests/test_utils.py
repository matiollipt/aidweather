# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

from aidweather.utils import ensure_date_column


def test_ensure_date_column_already_present():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
    res = ensure_date_column(df, name="date")
    assert "date" in res.columns
    assert pd.api.types.is_datetime64_any_dtype(res["date"])


def test_ensure_date_column_from_candidate():
    df = pd.DataFrame({"obs_date": ["2023-01-01", "2023-01-02"]})
    res = ensure_date_column(df, name="date", candidates=["obs_date"])
    assert "date" in res.columns
    assert "obs_date" not in res.columns
    assert pd.api.types.is_datetime64_any_dtype(res["date"])


def test_ensure_date_column_from_datetimeindex():
    idx = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame({"val": [1, 2]}, index=idx)
    res = ensure_date_column(df, name="date", index_fallback=True)
    assert "date" in res.columns
    assert pd.api.types.is_datetime64_any_dtype(res["date"])


def test_ensure_date_column_missing_raises():
    df = pd.DataFrame({"val": [1, 2]})
    with pytest.raises(ValueError, match="Could not ensure date column"):
        ensure_date_column(df, name="date", index_fallback=False)
