# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest
from aidweather.client import PowerClient

# A sample JSON response mimicking the NASA POWER daily API for a single point
SAMPLE_DAILY_RESPONSE = {
    "header": {
        "title": "NASA POWER Single Point Daily Data",
        "api": "https://power.larc.nasa.gov/api/temporal/daily/point",
    },
    "properties": {
        "parameter": {
            "T2M": {
                "20230101": 10.5,
                "20230102": 11.2,
                "20230103": -999,  # Missing value
            },
            "RH2M": {"20230101": 60.1, "20230102": 62.5, "20230103": 61.8},
        }
    },
    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
}

# A sample JSON response for the hourly API
SAMPLE_HOURLY_RESPONSE = {
    "properties": {"parameter": {"T2M": {"2023010100": 8.1, "2023010101": 8.5, "2023010102": -999}}}
}


@pytest.fixture
def mock_session(requests_mock):
    """Provides a requests session that is mocked."""
    return requests_mock


def test_get_daily_point_data_success(mock_session):
    """
    Tests successful fetching and parsing of daily data for a single point.
    """
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        json=SAMPLE_DAILY_RESPONSE,
    )
    client = PowerClient(temporal_api="daily")
    client.cache_cfg["enabled"] = False
    df = client.get_point_data(
        lat=0, lon=0, start="2023-01-01", end="2023-01-03", params=["T2M", "RH2M"]
    )
    assert not df.empty
    assert len(df) == 3
    assert list(df.columns) == ["T2M", "RH2M"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert pd.isna(df.loc["2023-01-03", "T2M"])
    assert df.loc["2023-01-01", "T2M"] == 10.5
    assert df.loc["2023-01-02", "RH2M"] == 62.5


def test_get_hourly_point_data_success(mock_session):
    """
    Tests successful fetching and parsing of hourly data.
    """
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/hourly/point",
        json=SAMPLE_HOURLY_RESPONSE,
    )
    client = PowerClient(temporal_api="hourly")
    client.cache_cfg["enabled"] = False
    df = client.get_point_data(
        lat=0, lon=0, start="2023-01-01 00:00", end="2023-01-01 02:00", params=["T2M"]
    )
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2023-01-01 00:00:00")
    assert pd.isna(df.loc["2023-01-01 02:00:00", "T2M"])


def test_api_request_failure_raises_error(mock_session):
    """
    Tests that a proper exception is raised when the API call fails
    and there is no cache to fall back on.
    """
    mock_session.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        status_code=500,
        text="Server Error",
    )
    client = PowerClient()
    client.cache_cfg["enabled"] = False
    with pytest.raises(IOError, match="API request failed"):
        client.get_point_data(lat=0, lon=0, start="2023-01-01", end="2023-01-03", params=["T2M"])


def test_mixed_param_latency_returns_partial_data(mock_session):
    """
    Tests that requesting a mix of fast (weather) and slow (solar)
    parameters for a recent date correctly returns data for the available
    parameter (T2M) while filling the unavailable one (ALLSKY) with NaNs.
    """
    # Simulate a response for a date (2025-11-18) where weather data is
    # available (2-day lag) but solar data is not (5-7 day lag).
    mixed_response = {
        "properties": {
            "parameter": {
                "T2M": {"20251118": 15.0},
                # ALLSKY_SFC_SW_DWN is requested but not present in the response
            }
        }
    }
    mock_session.get("https://power.larc.nasa.gov/api/temporal/daily/point", json=mixed_response)
    client = PowerClient()
    client.cache_cfg["enabled"] = False
    df = client.get_point_data(
        lat=0,
        lon=0,
        start="2025-11-18",
        end="2025-11-18",
        params=["T2M", "ALLSKY_SFC_SW_DWN"],
    )
    assert len(df) == 1
    assert df.loc["2025-11-18", "T2M"] == 15.0
    assert pd.isna(df.loc["2025-11-18", "ALLSKY_SFC_SW_DWN"])


def test_all_weather_params_available_after_lag(mock_session):
    """
    Tests that all meteorological parameters are available together after their
    expected 2-3 day release lag.
    """
    weather_params = ["T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR", "WS10M", "PS"]
    # Simulate a response where all requested weather params have data.
    weather_response = {
        "properties": {
            "parameter": {param: {"20251118": 10.0 + i} for i, param in enumerate(weather_params)}
        }
    }
    mock_session.get("https://power.larc.nasa.gov/api/temporal/daily/point", json=weather_response)
    client = PowerClient()
    client.cache_cfg["enabled"] = False
    df = client.get_point_data(
        lat=0, lon=0, start="2025-11-18", end="2025-11-18", params=weather_params
    )
    assert not df.empty
    assert not df.isnull().values.any()
    assert list(df.columns) == weather_params


def test_all_solar_params_unavailable_before_lag(mock_session):
    """
    Tests that solar radiation parameters are unavailable inside their
    5-7 day release lag, returning NaNs.
    """
    solar_params = ["ALLSKY_SFC_PAR_TOT", "ALLSKY_SFC_SW_DWN"]
    # Simulate an empty response, as would happen if data isn't ready.
    empty_response = {
        "header": {"title": "NASA POWER Single Point Daily Data"},
        "properties": {"parameter": {}},  # No data returned for the requested params
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }
    mock_session.get("https://power.larc.nasa.gov/api/temporal/daily/point", json=empty_response)
    client = PowerClient()
    client.cache_cfg["enabled"] = False
    df = client.get_point_data(
        lat=0, lon=0, start="2025-11-18", end="2025-11-18", params=solar_params
    )
    assert not df.empty
    assert df.isnull().values.all()
    assert list(df.columns) == solar_params
