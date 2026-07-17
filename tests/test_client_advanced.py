# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pandas as pd
import pytest

from aidweather.client import PowerClient
from aidweather.geo import GeoCoordinate

SAMPLE_POINT_RESPONSE = {
    "header": {
        "title": "NASA POWER Single Point Daily Data",
        "api": "https://power.larc.nasa.gov/api/temporal/daily/point",
    },
    "properties": {
        "parameter": {
            "T2M": {
                "20230101": 15.0,
                "20230102": 16.5,
            },
            "RH2M": {
                "20230101": 60.0,
                "20230102": 62.0,
            },
        }
    },
    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
}

SAMPLE_REGIONAL_RESPONSE = {
    "type": "FeatureCollection",
    "header": {
        "title": "NASA POWER Regional Daily Data",
        "api": "https://power.larc.nasa.gov/api/temporal/daily/regional",
    },
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [10.0, 20.0, 50.5]},
            "properties": {
                "parameter": {
                    "T2M": {
                        "20230101": 14.5,
                        "20230102": 15.2,
                    }
                }
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [10.5, 20.0, 52.0]},
            "properties": {
                "parameter": {
                    "T2M": {
                        "20230101": 13.8,
                        "20230102": 14.6,
                    }
                }
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [10.0, 20.5, 48.0]},
            "properties": {
                "parameter": {
                    "T2M": {
                        "20230101": 15.1,
                        "20230102": 15.9,
                    }
                }
            },
        },
    ],
}


@pytest.fixture
def mock_session(requests_mock):
    """Provides a mocked requests session."""
    return requests_mock


def test_get_multi_point_data_success(mock_session):
    """Verify concurrent fetching across multiple locations successfully aggregates output."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        json=SAMPLE_POINT_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    points = [{"lat": 10.0, "lon": 20.0}, {"lat": -10.0, "lon": -20.0}]
    df, failed = client.get_multi_point_data(
        points=points,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M", "RH2M"],
        max_workers=2,
    )

    assert not df.empty
    assert len(failed) == 0
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "T2M" in df.columns
    # Verify both points got fetched and combined
    assert df["lat"].nunique() == 2


def test_collect_futures_results_reports_failure_reason():
    """Failed points must carry the actual error, not just the point identity, so
    callers (and the CLI) can tell a real failure apart from "no data available"."""
    from concurrent.futures import Future

    ok_df = pd.DataFrame({"T2M": [1.0]}, index=pd.to_datetime(["2023-01-01"]))
    ok_df.index.name = "date"
    ok_future: Future = Future()
    ok_future.set_result(ok_df)

    bad_future: Future = Future()
    bad_future.set_exception(RuntimeError("simulated API failure"))

    ok_point = {"lat": 1.0, "lon": 2.0}
    bad_point = {"lat": 3.0, "lon": 4.0}

    results, failed = PowerClient._collect_futures_results(
        {ok_future: ok_point, bad_future: bad_point}
    )

    assert len(results) == 1
    assert len(failed) == 1
    point, error = failed[0]
    assert point == bad_point
    assert "simulated API failure" in error


def test_get_regional_data_success(mock_session):
    """Verify regional bounding box requests are correctly dispatched and parsed."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/regional",
        json=SAMPLE_REGIONAL_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    df = client.get_regional_data(
        lat_min=20.0,
        lat_max=21.0,
        lon_min=10.0,
        lon_max=11.0,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M"],
    )

    assert not df.empty
    # Regional endpoint parses the GeoJSON FeatureCollection
    assert "T2M" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    # Should have 3 grid cells × 2 dates = 6 rows
    assert len(df) == 6
    # Verify bounding box params are sent correctly
    assert mock_session.last_request.qs["latitude-min"] == ["20.0"]
    assert mock_session.last_request.qs["latitude-max"] == ["21.0"]
    assert mock_session.last_request.qs["longitude-min"] == ["10.0"]
    assert mock_session.last_request.qs["longitude-max"] == ["11.0"]


def test_get_regional_data_from_coordinates_success(mock_session):
    """Verify regional bounding box queries with GeoCoordinate convenience wrapper."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/regional",
        json=SAMPLE_REGIONAL_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    coord_sw = GeoCoordinate(lat=20.0, lon=10.0)
    coord_ne = GeoCoordinate(lat=21.0, lon=11.0)
    df = client.get_regional_data_from_coordinates(
        coord_sw=coord_sw,
        coord_ne=coord_ne,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M"],
    )

    assert not df.empty
    assert "T2M" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert mock_session.last_request.qs["latitude-min"] == ["20.0"]
    assert mock_session.last_request.qs["latitude-max"] == ["21.0"]


def test_get_transect_data_success(mock_session):
    """Verify get_transect_data generates transect coordinates and fetches concurrently."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        json=SAMPLE_POINT_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    coord_a = GeoCoordinate.from_decimal(15.0, -40.0)
    # ~555 km north along the same meridian – guarantees > 0.5° spacing per point
    coord_b = GeoCoordinate.from_decimal(20.0, -40.0)

    df = client.get_transect_data(
        start_coord=coord_a,
        end_coord=coord_b,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M"],
        num_points=3,
        max_workers=2,
    )

    assert not df.empty
    assert "lat" in df.columns
    assert "lon" in df.columns
    # 3 unique locations along the latitude axis
    assert df["lat"].nunique() == 3


def test_get_transect_data_from_coordinates_success(mock_session):
    """Verify get_transect_data_from_coordinates convenience wrapper dispatches correctly."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        json=SAMPLE_POINT_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    coord_a = GeoCoordinate.from_decimal(15.0, -40.0)
    coord_b = GeoCoordinate.from_decimal(20.0, -40.0)

    df = client.get_transect_data_from_coordinates(
        coord_a=coord_a,
        coord_b=coord_b,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M"],
        num_points=3,
        max_workers=2,
    )

    assert not df.empty
    assert "lat" in df.columns
    assert df["lat"].nunique() == 3


def test_get_transect_data_spacing_clamped():
    """Verify that sub-0.5° spacing is clamped to avoid redundant API calls."""
    # A very short transect (~5 km) where 20 points would be way under the 0.5° minimum
    coord_a = GeoCoordinate.from_decimal(15.0, -40.0)
    coord_b = GeoCoordinate.from_decimal(15.045, -40.0)  # ~5 km north

    resolved = PowerClient._resolve_transect_num_points(
        start_coord=coord_a,
        end_coord=coord_b,
        num_points=20,
        spacing_km=None,
    )
    # Minimum is 2 (the two endpoints)
    assert resolved == 2


def test_get_transect_data_spacing_km_derives_num_points():
    """Verify spacing_km is converted to num_points correctly."""
    coord_a = GeoCoordinate.from_decimal(0.0, 0.0)
    # ~555 km north
    coord_b = GeoCoordinate.from_decimal(5.0, 0.0)

    resolved = PowerClient._resolve_transect_num_points(
        start_coord=coord_a,
        end_coord=coord_b,
        num_points=None,
        spacing_km=111.0,  # ~1° spacing => expect ~6 points
    )
    # total ~555 km / 111 km + 1 = 6
    assert resolved >= 2


def test_summarize_console_logging(capsys):
    """Verify that summarize successfully prints the complex Rich tables without raising exceptions."""
    client = PowerClient()
    # Mock database session metrics for printout
    client._metrics["total_requests"] = 5
    client._metrics["api_calls"] = 2
    client._metrics["cache_hits"] = 3
    client._metrics["total_downloaded_bytes"] = 1024
    client._metrics["fetch_duration"] = 1.5

    dummy_df = pd.DataFrame(
        {"T2M": [15.0, 16.0], "RH2M": [60.0, 65.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    # Call summarize
    client.summarize(dummy_df)

    captured = capsys.readouterr()
    # Check that key headers or metrics are written to std out
    assert "Data Insight" in captured.out
    assert "Performance" in captured.out
    assert "Efficiency" in captured.out
    assert "API Connection" in captured.out


def test_regional_request_failure_logging(mock_session):
    """Verify that when a regional request fails, it logs using _safe_payload_repr without raising NameError."""
    # Register an error status code to trigger RequestException
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/regional",
        status_code=500,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    with patch("aidweather.client.logger") as mock_logger:
        with pytest.raises(OSError, match="Regional API request failed"):
            client.get_regional_data(
                lat_min=20.0,
                lat_max=21.0,
                lon_min=10.0,
                lon_max=11.0,
                start="2023-01-01",
                end="2023-01-02",
                params=["T2M"],
            )

        # Verify that logger.error was called and did not raise NameError
        mock_logger.error.assert_called()
        # Find the log call that contains "Regional API request failed for payload"
        failed_log_found = False
        for call in mock_logger.error.call_args_list:
            if "Regional API request failed for payload" in call[0][0]:
                failed_log_found = True
                # Payload should be serialized as JSON/compact string representation
                assert '"parameters": "T2M"' in call[0][1] or "'parameters': 'T2M'" in call[0][1]
        assert failed_log_found, "Could not find payload representation log message"



def test_parameter_specific_transect_clamping():
    """Verify sub-grid transect clamping uses requested parameter native grid resolution."""
    coord_a = GeoCoordinate.from_decimal(0.0, 0.0)
    coord_b = GeoCoordinate.from_decimal(0.8, 0.0)  # ~88.8 km distance

    # For MERRA-2 (0.5° lat step ~55.55 km): max points over 88.8 km is 88.8/55.55 + 1 = 2 points
    res_merra = PowerClient._resolve_transect_num_points(
        start_coord=coord_a,
        end_coord=coord_b,
        num_points=10,
        spacing_km=None,
        params=["T2M"],
    )
    assert res_merra == 2

    # For CERES solar radiation (1.0° lat step ~111.1 km): max points over 88.8 km is 2 (endpoints clamped)
    res_solar = PowerClient._resolve_transect_num_points(
        start_coord=coord_a,
        end_coord=coord_b,
        num_points=10,
        spacing_km=None,
        params=["ALLSKY_SFC_SW_DWN"],
    )
    assert res_solar == 2


def test_cache_key_versioning():
    """Verify cache keys contain version prefix v1_."""
    from aidweather.client import _make_cache_key
    payload = {"parameters": "T2M", "latitude": 10.0, "longitude": 20.0}
    key = _make_cache_key(payload, temporal_api="daily")
    assert key.startswith("v1_")

