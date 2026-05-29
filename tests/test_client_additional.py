# SPDX-License-Identifier: Apache-2.0
import os

from aidweather.client import PowerClient, _load_env_file


def test_load_env_file_success(tmp_path, monkeypatch):
    """Tests that _load_env_file correctly parses a .env file."""
    env_content = """NASA_POWER_API_KEY=test_key_from_env
OTHER_VAR=value
# Commented=out"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    # Change current working directory to the temporary directory
    monkeypatch.chdir(tmp_path)

    # Ensure the environment variable is not already set
    monkeypatch.delenv("NASA_POWER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)

    _load_env_file()

    assert os.environ.get("NASA_POWER_API_KEY") == "test_key_from_env"
    assert os.environ.get("OTHER_VAR") == "value"


def test_power_client_loads_env_on_init(tmp_path, monkeypatch):
    """Tests that PowerClient loads the API key from .env on initialization."""
    env_content = "NASA_POWER_API_KEY=secret_key_123"
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NASA_POWER_API_KEY", raising=False)

    # We don't want to actually connect to the cache for this test
    monkeypatch.setenv("AGRILYZER_CACHE_ENABLED", "false")

    client = PowerClient()
    assert client.api_key == "secret_key_123"


def test_load_env_file_ignores_comments_and_empty_lines(tmp_path, monkeypatch):
    """Tests that _load_env_file ignores comments, empty lines, and empty values."""
    env_content = """
# This is a comment
NASA_POWER_API_KEY=valid_key

  # Another comment
EMPTY_VALUE=
  SPACES_AROUND  =  trimmed_value
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NASA_POWER_API_KEY", raising=False)
    monkeypatch.delenv("SPACES_AROUND", raising=False)

    _load_env_file()

    assert os.environ.get("NASA_POWER_API_KEY") == "valid_key"
    assert os.environ.get("SPACES_AROUND") == "trimmed_value"
    assert "EMPTY_VALUE" not in os.environ


def test_load_env_file_handles_quotes(tmp_path, monkeypatch):
    """Tests that _load_env_file correctly handles quoted values."""
    env_content = """SINGLE_QUOTED='value1'
DOUBLE_QUOTED="value2\""""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SINGLE_QUOTED", raising=False)
    monkeypatch.delenv("DOUBLE_QUOTED", raising=False)

    _load_env_file()

    assert os.environ.get("SINGLE_QUOTED") == "value1"
    assert os.environ.get("DOUBLE_QUOTED") == "value2"
