# SPDX-License-Identifier: Apache-2.0
import os

import pandas as pd
import pytest

from aidweather.client import PowerClient


@pytest.mark.skipif(
    os.environ.get("AIDWEATHER_RUN_LIVE_TESTS") != "1",
    reason="AIDWEATHER_RUN_LIVE_TESTS=1 is not set"
)
def test_live_nasa_power_connection():
    """Verify that a live query to NASA POWER API returns valid weather data."""
    client = PowerClient(temporal_api="daily")
    client.cache_cfg["enabled"] = False  # Ensure we bypass local cache for live test

    # Fetch daily temperature (T2M) for one coordinate over a 2-day range
    df = client.get_point_data(
        lat=-21.77,
        lon=-48.82,
        start="20230101",
        end="20230102",
        params=["T2M"],
    )

    assert not df.empty, "Response DataFrame should not be empty"
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert "T2M" in df.columns, "T2M parameter column should be present"
    assert len(df) == 2, "Should return exactly 2 days of data"
