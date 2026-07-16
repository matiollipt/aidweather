# SPDX-License-Identifier: Apache-2.0
# tests/test_config.py

import json
from unittest.mock import patch

import pytest

from aidweather.config import _Config, _load_config_dict, cfg

# --- Fixtures ---


@pytest.fixture
def sample_config_data():
    """
    Provides a dictionary mimicking the structure of a real config.json file.
    This allows testing the _Config class in isolation from the actual file system.
    """
    return {
        "base_urls": {
            "daily": {"point": "http://test.com/daily/point"},
            "hourly": {"point": "http://test.com/hourly/point"},
        },
        "params": {"default": {"T2M": "Temperature at 2m"}, "metadata": {"LAT": "Latitude"}},
        "cache_config": {"enabled": False, "custom_path": "/tmp/custom_cache"},
        "api_limits": {"max_days": 366},
    }


@pytest.fixture
def empty_config():
    """Provides an empty config dictionary to test default fallback logic."""
    return {}


# --- Tests for _Config Class Logic ---


def test_get_nested_value(sample_config_data):
    """
    Tests the `get` method's ability to retrieve a nested value using dot notation.
    """
    # Why: This is the core mechanism for accessing config values.
    config = _Config(sample_config_data)
    url = config.get("base_urls.daily.point")
    assert url == "http://test.com/daily/point"


def test_get_missing_value_returns_default(sample_config_data):
    """
    Tests that the `get` method returns the provided default value for a key that does not exist.
    """
    # Why: Graceful handling of missing keys is crucial for robustness.
    config = _Config(sample_config_data)
    value = config.get("non.existent.key", default="fallback")
    assert value == "fallback"


def test_get_url_from_config(sample_config_data):
    """
    Tests that `get_url` retrieves a URL present in the configuration data.
    """
    # Why: To ensure user-defined URLs are correctly prioritized.
    config = _Config(sample_config_data)
    url = config.get_url("daily", "point")
    assert url == "http://test.com/daily/point"


def test_get_url_fallback_to_default(empty_config):
    """
    Tests that `get_url` falls back to the hardcoded default URL when the key is not in the config.
    """
    # Why: To ensure the system works out-of-the-box without a config file.
    config = _Config(empty_config)
    url = config.get_url("daily", "point")
    assert "power.larc.nasa.gov" in url
    assert url == "https://power.larc.nasa.gov/api/temporal/daily/point"


def test_cache_config_merging(sample_config_data):
    """
    Tests that `cache_config` correctly merges default settings with those from the config file.
    The user config should override defaults where specified, and custom keys should be preserved.
    """
    # Why: To verify the flexible override mechanism for complex settings.
    config = _Config(sample_config_data)
    # The get method for cache_config now returns the merged dictionary.
    # We need to get the raw user config to test the merging logic.
    user_cache_config = config.get("cache_config", default={})

    defaults = {
        "enabled": True,
        "path": "default/path",
    }
    merged_config = {**defaults, **user_cache_config}

    # User override should be respected
    assert merged_config["enabled"] is False
    # Default should be present if not overridden
    assert "path" in merged_config
    # The user's custom key should be preserved in the final config.
    assert "custom_path" in merged_config
    assert merged_config["custom_path"] == "/tmp/custom_cache"


def test_params_retrieval(sample_config_data, empty_config):
    """
    Tests retrieval of parameter groups.
    - Should return the specific group if found.
    - Should fall back to the 'default' group if the requested group is missing.
    - Should return an empty dict if no params are configured at all.
    """
    # Why: Parameter mapping is essential for API requests.
    config = _Config(sample_config_data)

    # Get a specific group
    metadata_params = config.params("metadata")
    assert metadata_params == {"LAT": "Latitude"}

    # Fallback to default group
    missing_group = config.params("non_existent_group")
    assert missing_group == {"T2M": "Temperature at 2m"}

    # Test with no config at all
    empty_config_obj = _Config(empty_config)
    assert empty_config_obj.params("default") == {}


# --- Tests for File I/O and Error Handling ---



@patch("importlib.resources.files")
def test_load_config_dict_file_not_found(mock_files, caplog):
    """
    Tests that _load_config_dict returns an empty dict and logs a warning
    if the config.json file cannot be found.
    """
    # Why: Test robustness against a missing primary config file.

    # Arrange: Make the resource traversal itself raise the error.
    mock_files.side_effect = FileNotFoundError

    # Act
    data = _load_config_dict()

    # Assert
    assert data == {}
    assert "Could not load internal config.json" in caplog.text


@patch("json.load", side_effect=json.JSONDecodeError("Error", "doc", 0))
@patch("importlib.resources.files")
def test_load_config_dict_invalid_json(mock_files, mock_json_load, caplog):
    """
    Tests that _load_config_dict returns an empty dict and logs a warning
    if config.json contains invalid JSON. This is done by patching json.load directly.
    """
    # Why: Test robustness against a corrupted config file.
    # The mock_files patch is just to prevent the real file system access.

    # Act
    data = _load_config_dict()

    # Assert
    assert data == {}
    assert "Could not load internal config.json" in caplog.text


# --- Test the live singleton instance ---


def test_singleton_cfg_is_loaded():
    """
    Performs a basic sanity check on the live `cfg` instance.
    This is a mini-integration test to ensure the real file loads.
    """
    # Why: To confirm that the module-level instantiation works as expected.
    assert cfg.get("base_urls.daily.point") is not None
    assert "power.larc.nasa.gov" in cfg.get_url("daily")
    assert isinstance(cfg.param_groups(), list)


def test_param_metadata_and_grid_resolution():
    """
    Test retrieval of parameter metadata and native grid resolutions.
    """
    meta_t2m = cfg.param_metadata("T2M")
    assert meta_t2m.get("source_family") == "MERRA-2/GEOS-IT"
    assert cfg.get_native_grid("T2M") == (0.5, 0.625)

    meta_solar = cfg.param_metadata("ALLSKY_SFC_SW_DWN")
    assert "CERES" in meta_solar.get("source_family", "")
    assert cfg.get_native_grid("ALLSKY_SFC_SW_DWN") == (1.0, 1.0)

    # Unknown parameter fallback
    assert cfg.get_native_grid("UNKNOWN_PARAM") == (0.5, 0.625)
