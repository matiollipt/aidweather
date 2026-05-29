# SPDX-License-Identifier: Apache-2.0
import pandas as pd
from aidweather.utils import ensure_date_column


def test_ensure_date_column_inplace_true():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
    # mutate in place
    ensure_date_column(df, name="date", inplace=True)
    # original df should have datetime dtype
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
