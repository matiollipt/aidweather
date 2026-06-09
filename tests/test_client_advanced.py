# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from aidweather.geo import GeoCoordinate
from aidweather.client import PowerClient, ExpandedPointRequest

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


def test_get_expanded_point_data_success(mock_session):
    """Verify get_expanded_point_data correctly generates transect coordinates and fetches concurrently."""
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        json=SAMPLE_POINT_RESPONSE,
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False

    # Perform a minimal transect search with 3 generated points
    df = client.get_expanded_point_data(
        lat=15.0,
        lon=-40.0,
        start="2023-01-01",
        end="2023-01-02",
        params=["T2M"],
        axis="lat",
        distance_km=5.0,
        num_points=3,
        max_workers=2,
    )

    assert not df.empty
    assert "lat" in df.columns
    assert "lon" in df.columns
    # Check that it generated 3 unique locations along the latitude axis
    assert df["lat"].nunique() == 3


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
