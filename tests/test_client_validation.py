# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from aidweather.client import PowerClient
from aidweather.geo import GeoCoordinate


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


def test_max_workers_enforcement():
    # Enforces max workers clamping
    from aidweather.config import cfg

    original_workers = cfg.get("api_limits.max_workers")
    try:
        cfg.set("api_limits.max_workers", 3)
        client = PowerClient(temporal_api="daily")
        assert client.max_workers_limit == 3

        # Test get_multi_point_data clamping
        with patch.object(client, "_parse_points_input", return_value=[]):
            with patch("aidweather.client.ThreadPoolExecutor") as mock_executor:
                client.get_multi_point_data(
                    points=[],
                    start="20230101",
                    end="20230101",
                    params=["T2M"],
                    max_workers=10,
                )
                mock_executor.assert_called_once_with(max_workers=3)
    finally:
        cfg.set("api_limits.max_workers", original_workers)


def test_rate_limiting_throttling():
    import time

    client = PowerClient(temporal_api="daily")
    # Set the rate limit to 2 calls per 1 second
    client.rate_limiter.max_calls = 2
    client.rate_limiter.period = 1.0
    client.rate_limiter.calls = []

    t0 = time.time()
    client.rate_limiter.acquire()
    client.rate_limiter.acquire()
    # Third call should block until 1 second has elapsed since the first call
    client.rate_limiter.acquire()
    t1 = time.time()

    assert t1 - t0 >= 0.8  # Should be close to 1.0 second delay


def test_regional_bbox_too_large():
    """Bounding boxes exceeding 4.5° on either axis should be rejected."""
    client = PowerClient(temporal_api="daily")
    with pytest.raises(ValueError, match="Bounding box too large"):
        client._build_regional_payload(
            params=["T2M"],
            start="20230101",
            end="20230131",
            lat_min=0.0,
            lat_max=5.0,  # 5° span > 4.5° limit
            lon_min=0.0,
            lon_max=4.0,
        )

    with pytest.raises(ValueError, match="Bounding box too large"):
        client._build_regional_payload(
            params=["T2M"],
            start="20230101",
            end="20230131",
            lat_min=0.0,
            lat_max=4.0,
            lon_min=0.0,
            lon_max=5.0,  # 5° span > 4.5° limit
        )


def test_regional_bbox_lat_min_exceeds_max():
    """lat_min >= lat_max should be rejected."""
    client = PowerClient(temporal_api="daily")
    with pytest.raises(ValueError, match="lat_min.*must be less than lat_max"):
        client._build_regional_payload(
            params=["T2M"],
            start="20230101",
            end="20230131",
            lat_min=10.0,
            lat_max=5.0,
            lon_min=0.0,
            lon_max=4.0,
        )


def test_regional_bbox_lon_min_exceeds_max():
    """lon_min >= lon_max should be rejected."""
    client = PowerClient(temporal_api="daily")
    with pytest.raises(ValueError, match="lon_min.*must be less than lon_max"):
        client._build_regional_payload(
            params=["T2M"],
            start="20230101",
            end="20230131",
            lat_min=0.0,
            lat_max=4.0,
            lon_min=10.0,
            lon_max=5.0,
        )


def test_regional_bbox_valid():
    """A valid bounding box within 4.5° should produce correct payload."""
    client = PowerClient(temporal_api="daily")
    payload = client._build_regional_payload(
        params=["T2M"],
        start="20230101",
        end="20230131",
        lat_min=-23.5,
        lat_max=-20.0,
        lon_min=-47.0,
        lon_max=-44.0,
    )
    assert payload["latitude-min"] == -23.5
    assert payload["latitude-max"] == -20.0
    assert payload["longitude-min"] == -47.0
    assert payload["longitude-max"] == -44.0
    assert payload["parameters"] == "T2M"
    assert "lonlat" not in payload  # Old field must not be present


def test_date_range_validation():
    """Verify start > end raises ValueError across different query methods."""
    client = PowerClient(temporal_api="daily")
    client.cache_cfg["enabled"] = False

    # point
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        client.get_point_data_from_coordinate(
            coord=GeoCoordinate.from_decimal(0, 0),
            start="20230102",
            end="20230101",
            params=["T2M"],
        )

    # multi-point
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        client.get_multi_point_data(
            points=[{"lat": 0, "lon": 0}],
            start="20230102",
            end="20230101",
            params=["T2M"],
        )

    # regional
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        client.get_regional_data(
            lat_min=0,
            lat_max=1,
            lon_min=0,
            lon_max=1,
            start="20230102",
            end="20230101",
            params=["T2M"],
        )

    # transect
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        client.get_transect_data_from_coordinates(
            coord_a=GeoCoordinate.from_decimal(0, 0),
            coord_b=GeoCoordinate.from_decimal(1, 1),
            start="20230102",
            end="20230101",
            params=["T2M"],
            num_points=3,
        )


def test_unknown_parameter_warning():
    """Verify that using an unknown parameter raises a warning but doesn't block the request."""
    client = PowerClient(temporal_api="daily")
    client.cache_cfg["enabled"] = False

    # We mock self.session.get to avoid actually hitting the API or requiring mocks for everything
    with patch.object(client.session, "get") as mock_get:
        # Mock a minimal response
        from unittest.mock import Mock
        mock_resp = Mock()
        mock_resp.content = b"{}"
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        # Warns on unknown params
        with pytest.warns(UserWarning, match="Unknown parameter.*INVALID_PARAM"):
            try:
                client.get_point_data_from_coordinate(
                    coord=GeoCoordinate.from_decimal(0, 0),
                    start="20230101",
                    end="20230101",
                    params=["INVALID_PARAM"],
                )
            except Exception:
                # We don't care if it fails on JSON parsing, we just want to verify the warning
                pass

        # Does not warn on valid parameter
        import warnings
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                client.get_point_data_from_coordinate(
                    coord=GeoCoordinate.from_decimal(0, 0),
                    start="20230101",
                    end="20230101",
                    params=["T2M"],
                )
            except Exception:
                pass
        # Assert no warnings were issued
        user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0
