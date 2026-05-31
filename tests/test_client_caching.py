# SPDX-License-Identifier: Apache-2.0
import os

import pandas as pd
import pytest
import requests_mock
from aidweather.config import cfg
from aidweather.client import PowerClient


@pytest.fixture
def mock_cache_config(tmp_path, monkeypatch):
    """
    Mocks the configuration to use a temporary directory for the SQLite cache
    and ensures caching is enabled.
    """

    def mock_return():
        return {"enabled": True, "path": str(tmp_path), "ttl_seconds": 3600}

    monkeypatch.setattr(cfg, "cache_config", mock_return)
    return tmp_path


def test_caching_logic(mock_cache_config):
    """
    Verifies that:
    1. The first request hits the API and saves to the DB.
    2. The second request hits the DB (cache) and does NOT hit the API.
    """
    lat, lon = 10.0, 20.0
    start, end = "20230101", "20230102"
    params = ["T2M"]

    # Mock API response structure
    api_response = {"properties": {"parameter": {"T2M": {"20230101": 25.0, "20230102": 26.0}}}}

    with requests_mock.Mocker() as m:
        # Match any URL since we are testing the client logic, not URL construction
        m.get(requests_mock.ANY, json=api_response)

        client = PowerClient(temporal_api="daily")

        # 1. First Call - Should hit API
        df1 = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=params)
        assert m.call_count == 1, "API should be called once on cache miss"
        assert not df1.empty
        assert df1.loc["2023-01-01", "T2M"] == 25.0

        # 2. Second Call - Should hit Cache (API count stays 1)
        df2 = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=params)
        assert m.call_count == 1, "API should not be called on cache hit"
        pd.testing.assert_frame_equal(df1, df2)

        # 3. Verify DB persistence
        db_path = os.path.join(mock_cache_config, "aidweather_cache.db")
        assert os.path.exists(db_path)


def test_cache_fetches_and_merges_missing_date_ranges(mock_cache_config):
    """A broader request should reuse cached dates and fetch only missing edges."""
    params = ["T2M"]
    cached_response = {"properties": {"parameter": {"T2M": {"20230102": 26.0}}}}
    start_response = {"properties": {"parameter": {"T2M": {"20230101": 25.0}}}}
    end_response = {"properties": {"parameter": {"T2M": {"20230103": 27.0}}}}

    with requests_mock.Mocker() as m:
        m.get(
            requests_mock.ANY,
            [
                {"json": cached_response},
                {"json": start_response},
                {"json": end_response},
            ],
        )

        client = PowerClient(temporal_api="daily")
        first = client.get_point_data(
            lat=10.0, lon=20.0, start="20230102", end="20230102", params=params
        )
        expanded = client.get_point_data(
            lat=10.0, lon=20.0, start="20230101", end="20230103", params=params
        )

    assert m.call_count == 3
    assert list(first.index) == [pd.Timestamp("2023-01-02")]
    assert list(expanded.index) == list(pd.date_range("2023-01-01", "2023-01-03"))
    assert expanded["T2M"].tolist() == [25.0, 26.0, 27.0]


def test_stale_cache_returned_when_missing_range_fetch_fails(mock_cache_config):
    """If a cache entry exists, a failed refresh should return available stale data."""
    params = ["T2M"]
    cached_response = {
        "properties": {
            "parameter": {"T2M": {"20230101": 25.0, "20230102": 26.0}}
        }
    }

    with requests_mock.Mocker() as m:
        m.get(
            requests_mock.ANY,
            [
                {"json": cached_response},
                {"status_code": 500, "text": "Server Error"},
            ],
        )

        client = PowerClient(temporal_api="daily")
        client.get_point_data(
            lat=10.0, lon=20.0, start="20230101", end="20230102", params=params
        )
        stale = client.get_point_data(
            lat=10.0, lon=20.0, start="20230101", end="20230103", params=params
        )

    assert m.call_count == 2
    assert list(stale.index) == list(pd.date_range("2023-01-01", "2023-01-02"))
    assert stale["T2M"].tolist() == [25.0, 26.0]
