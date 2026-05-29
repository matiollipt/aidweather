# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest
from aidweather.client import PowerClient, _safe_payload_repr


def test_safe_payload_repr():
    payload = {"parameters": "T2M", "api_key": "1234567890abcdef"}
    repr_str = _safe_payload_repr(payload)
    assert "1234567890" not in repr_str
    assert "***abcdef" in repr_str

    payload_demo = {"parameters": "T2M", "api_key": "DEMO_KEY"}
    assert "DEMO_KEY" in _safe_payload_repr(payload_demo)

    payload_short = {"parameters": "T2M", "api_key": "123"}
    assert "***" in _safe_payload_repr(payload_short)
    assert "123" not in _safe_payload_repr(payload_short)


def test_hourly_point_limit():
    client = PowerClient(temporal_api="hourly")
    params = [f"P{i}" for i in range(16)]
    with pytest.raises(ValueError, match="maximum of 15 parameters"):
        client._validate_request(params, is_regional=False)


def test_daily_point_limit():
    client = PowerClient(temporal_api="daily")
    params = [f"P{i}" for i in range(21)]
    with pytest.raises(ValueError, match="maximum of 20 parameters"):
        client._validate_request(params, is_regional=False)


def test_regional_limit():
    client = PowerClient(temporal_api="daily")
    params = ["T2M", "RH2M"]
    with pytest.raises(ValueError, match="maximum of 1 parameter"):
        client._validate_request(params, is_regional=True)


def test_hourly_regional_rejected():
    client = PowerClient(temporal_api="hourly")
    params = ["T2M"]
    with pytest.raises(ValueError, match="does not support regional"):
        client._validate_request(params, is_regional=True)


def test_wind_elevation_validation():
    client = PowerClient(temporal_api="daily")
    with pytest.raises(ValueError, match="wind-elevation must be between 10 and 300"):
        client._build_point_payload(
            params=["T2M"],
            start="20230101",
            end="20230101",
            lon=0,
            lat=0,
            wind_elevation=5,
        )

    with pytest.raises(ValueError, match="wind-elevation must be between 10 and 300"):
        client._build_point_payload(
            params=["T2M"],
            start="20230101",
            end="20230101",
            lon=0,
            lat=0,
            wind_elevation=400,
        )

    # Valid range should not raise
    payload = client._build_point_payload(
        params=["T2M"],
        start="20230101",
        end="20230101",
        lon=0,
        lat=0,
        wind_elevation=10,
    )
    assert payload["wind-elevation"] == 10


@patch.dict(os.environ, {"NASA_POWER_API_KEY": "SECRET_KEY"})
def test_api_key_loaded_from_env():
    client = PowerClient(temporal_api="daily")
    assert client.api_key == "SECRET_KEY"

    payload = client._build_point_payload(
        params=["T2M"], start="20230101", end="20230101", lon=0, lat=0
    )
    assert payload["api_key"] == "SECRET_KEY"
