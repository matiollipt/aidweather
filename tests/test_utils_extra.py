# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from aidweather.utils import ensure_date_column


def test_ensure_date_column_inplace_true():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
    # mutate in place
    ensure_date_column(df, name="date", inplace=True)
    # original df should have datetime dtype
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_ensure_date_column_normalization():
    df = pd.DataFrame({"date": ["2023-01-01 12:30:00", "2023-01-02 18:45:00"]})
    
    # 1. Default should NOT normalize (should keep hours/minutes)
    res_default = ensure_date_column(df, name="date")
    assert res_default["date"].iloc[0].hour == 12
    assert res_default["date"].iloc[0].minute == 30
    
    # 2. normalize=True should strip hours/minutes (to midnight)
    res_normalized = ensure_date_column(df, name="date", normalize=True)
    assert res_normalized["date"].iloc[0].hour == 0
    assert res_normalized["date"].iloc[0].minute == 0
