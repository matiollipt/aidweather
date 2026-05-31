# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch
import pandas as pd
from aidweather.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_help():
    """Test that the CLI help message displays properly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "NASA POWER" in result.stdout


def test_version():
    """Test that the CLI exposes a top-level version option."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "aidweather 0.1.0" in result.stdout


def test_params_list():
    """Test 'params list' sub-command."""
    result = runner.invoke(app, ["params", "list"])
    assert result.exit_code == 0
    assert "NASA POWER Parameters" in result.stdout
    assert "T2M" in result.stdout


def test_params_describe():
    """Test 'params describe' sub-command."""
    result = runner.invoke(app, ["params", "describe", "T2M"])
    assert result.exit_code == 0
    assert "Average Air Temperature" in result.stdout


def test_params_describe_invalid():
    """Test 'params describe' sub-command with invalid parameter."""
    result = runner.invoke(app, ["params", "describe", "INVALID_PARAM_XYZ"])
    assert result.exit_code == 1
    assert "Unknown parameter code" in result.stdout


def test_cache_info():
    """Test 'cache info' sub-command."""
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "Cache" in result.stdout


def test_fetch_missing_args():
    """Test 'fetch' command without required arguments."""
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 2
    assert "Missing option '--lat'" in result.output


def test_fetch_invalid_date():
    """Test 'fetch' command with invalid date format."""
    result = runner.invoke(
        app,
        ["fetch", "--lat", "-15.0", "--lon", "-47.0", "--start", "abc", "--end", "cde"],
    )
    assert result.exit_code == 2
    assert "Invalid date" in result.output


def test_fetch_multi_missing_file(tmp_path):
    """Test 'fetch-multi' with a non-existent file."""
    result = runner.invoke(
        app,
        [
            "fetch-multi",
            "--points-file",
            str(tmp_path / "does_not_exist.csv"),
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-31",
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_fetch_multi_invalid_csv(tmp_path):
    """Test 'fetch-multi' with a CSV lacking lat/lon columns."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("""a,b
1,2""")

    result = runner.invoke(
        app,
        [
            "fetch-multi",
            "--points-file",
            str(bad_csv),
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-31",
        ],
    )
    assert result.exit_code == 1
    assert "CSV must contain 'lat' and 'lon' columns" in result.stdout


@patch("aidweather.cli.PowerClient")
def test_fetch_success(mock_client_class, tmp_path):
    """Test 'fetch' command successfully downloads and writes point data."""
    mock_client = mock_client_class.return_value
    dummy_df = pd.DataFrame(
        {"T2M": [15.0, 16.0]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"])
    )
    dummy_df.index.name = "date"
    mock_client.get_point_data.return_value = dummy_df

    output_csv = tmp_path / "output.csv"
    result = runner.invoke(
        app,
        [
            "fetch",
            "--lat",
            "12.3",
            "--lon",
            "45.6",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--output",
            str(output_csv),
            "--format",
            "csv",
            "--summarize",
        ],
    )
    assert result.exit_code == 0
    assert "Data saved to" in result.stdout
    assert output_csv.exists()


@patch("aidweather.cli.PowerClient")
def test_fetch_output_extension_overrides_format(mock_client_class, tmp_path):
    """A .csv output path should produce CSV even if --format json is passed."""
    mock_client = mock_client_class.return_value
    dummy_df = pd.DataFrame(
        {"T2M": [15.0, 16.0]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"])
    )
    dummy_df.index.name = "date"
    mock_client.get_point_data.return_value = dummy_df

    output_csv = tmp_path / "weather.csv"
    result = runner.invoke(
        app,
        [
            "fetch",
            "--lat",
            "12.3",
            "--lon",
            "45.6",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-02",
            "--output",
            str(output_csv),
            "--format",
            "json",
            "--no-preview",
        ],
    )

    assert result.exit_code == 0
    assert "ignoring --format json" in result.stdout
    saved = pd.read_csv(output_csv)
    assert "date" in saved.columns
    assert saved["T2M"].tolist() == [15.0, 16.0]


@patch("aidweather.cli.PowerClient")
def test_fetch_output_without_extension_uses_format(mock_client_class, tmp_path):
    """When there is no recognized extension, --format still selects the writer."""
    mock_client = mock_client_class.return_value
    dummy_df = pd.DataFrame(
        {"T2M": [15.0]}, index=pd.to_datetime(["2023-01-01"])
    )
    dummy_df.index.name = "date"
    mock_client.get_point_data.return_value = dummy_df

    output_path = tmp_path / "weather-data"
    result = runner.invoke(
        app,
        [
            "fetch",
            "--lat",
            "12.3",
            "--lon",
            "45.6",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-01",
            "--output",
            str(output_path),
            "--format",
            "json",
            "--no-preview",
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text().startswith("[")


@patch("aidweather.cli.PowerClient")
def test_fetch_multi_success(mock_client_class, tmp_path):
    """Test 'fetch-multi' command successfully parses CSV and gets multi point data."""
    # Write a valid CSV points file
    points_csv = tmp_path / "points.csv"
    points_csv.write_text("""lat,lon,elevation
12.3,45.6,150.0
-12.3,-45.6,200.0""")

    mock_client = mock_client_class.return_value
    dummy_df = pd.DataFrame(
        {"lat": [12.3, -12.3], "lon": [45.6, -45.6], "T2M": [15.0, 16.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-01"]),
    )
    dummy_df.index.name = "date"
    mock_client.get_multi_point_data.return_value = (dummy_df, [])

    output_json = tmp_path / "output.json"
    result = runner.invoke(
        app,
        [
            "fetch-multi",
            "--points-file",
            str(points_csv),
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-01",
            "--output",
            str(output_json),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert "Data saved to" in result.stdout
    assert output_json.exists()


@patch("aidweather.cli.PowerClient")
def test_fetch_transect_success(mock_client_class, tmp_path):
    """Test 'fetch-transect' command successfully fetches coordinate expansions."""
    mock_client = mock_client_class.return_value
    dummy_df = pd.DataFrame(
        {"lat": [12.3, 12.4], "lon": [45.6, 45.6], "T2M": [15.0, 16.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-01"]),
    )
    dummy_df.index.name = "date"
    mock_client.get_expanded_point_data.return_value = dummy_df

    output_csv = tmp_path / "transect.csv"
    result = runner.invoke(
        app,
        [
            "fetch-transect",
            "--lat",
            "12.3",
            "--lon",
            "45.6",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-01",
            "--output",
            str(output_csv),
            "--axis",
            "lat",
            "--distance-km",
            "10",
            "--num-points",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Data saved to" in result.stdout
    assert output_csv.exists()


def test_cache_clear_success(tmp_path):
    """Test 'cache clear' unlinks the correct cached SQLite database."""
    db_file = tmp_path / "aidweather_cache.db"
    db_file.write_text("dummy sqlite content")

    with patch(
        "aidweather.cli.cfg.cache_config",
        return_value={"enabled": True, "path": str(tmp_path)},
    ):
        result = runner.invoke(app, ["cache", "clear", "--yes"])
        assert result.exit_code == 0
        assert "Cache cleared successfully" in result.stdout
        assert not db_file.exists()
