# SPDX-License-Identifier: Apache-2.0
"""Tests for environment variable overrides (AIDWEATHER_CACHE_DIR, AIDWEATHER_LOG_DIR, etc.)."""

import logging

from typer.testing import CliRunner

import aidweather
from aidweather import PowerClient
from aidweather.cli import app
from aidweather.config import cfg

runner = CliRunner()


def test_env_var_cache_dir_resolution(monkeypatch, tmp_path):
    """Test that AIDWEATHER_CACHE_DIR overrides default and JSON cache paths."""
    custom_dir = tmp_path / "custom_cache_dir"
    monkeypatch.setenv("AIDWEATHER_CACHE_DIR", str(custom_dir))

    cache_cfg = cfg.cache_config()
    assert cache_cfg["path"] == str(custom_dir)


def test_env_var_cache_dir_priority_over_json(monkeypatch, tmp_path):
    """Test that AIDWEATHER_CACHE_DIR takes priority over cache_config.path in JSON."""
    json_dir = tmp_path / "json_cache_dir"
    env_dir = tmp_path / "env_cache_dir"

    cfg.set("cache_config.path", str(json_dir))
    monkeypatch.setenv("AIDWEATHER_CACHE_DIR", str(env_dir))

    try:
        cache_cfg = cfg.cache_config()
        assert cache_cfg["path"] == str(env_dir)
    finally:
        cfg.set("cache_config.path", None)


def test_env_var_cache_dir_power_client_integration(monkeypatch, tmp_path):
    """Test that PowerClient initializes SQLite cache inside AIDWEATHER_CACHE_DIR."""
    custom_dir = tmp_path / "client_cache_dir"
    monkeypatch.setenv("AIDWEATHER_CACHE_DIR", str(custom_dir))

    client = PowerClient(temporal_api="daily")
    assert client.db_conn is not None

    db_file = custom_dir / "aidweather_cache.db"
    assert db_file.exists()


def test_env_var_cache_dir_cli_commands(monkeypatch, tmp_path):
    """Test that CLI cache info and cache clear respect AIDWEATHER_CACHE_DIR."""
    import sqlite3

    custom_dir = tmp_path / "cli_cache_dir"
    custom_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIDWEATHER_CACHE_DIR", str(custom_dir))

    db_file = custom_dir / "aidweather_cache.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "CREATE TABLE cache (key TEXT PRIMARY KEY, timestamp TEXT, data BLOB)"
        )

    # 1. Test cache info output reports env var source and path
    res_info = runner.invoke(app, ["cache", "info"])
    assert res_info.exit_code == 0
    assert "AIDWEATHER_CACHE_DIR env var" in res_info.stdout
    stdout_clean = res_info.stdout.replace("\n", "")
    assert "cli_cache_dir" in stdout_clean
    assert "aidweather_cache.db" in stdout_clean

    # 2. Test cache clear deletes the db file from custom_dir
    res_clear = runner.invoke(app, ["cache", "clear", "--yes"])
    assert res_clear.exit_code == 0
    assert not db_file.exists()


def test_env_var_log_dir_resolution(monkeypatch, tmp_path):
    """Test that AIDWEATHER_LOG_DIR overrides default log directory."""
    custom_log_dir = tmp_path / "custom_log_dir"
    monkeypatch.setenv("AIDWEATHER_LOG_DIR", str(custom_log_dir))

    log_cfg = cfg.logging_config()
    expected_file = str(custom_log_dir / "aidweather.log")
    assert log_cfg["filename"] == expected_file


def test_env_var_log_dir_priority_over_json(monkeypatch, tmp_path):
    """Test that AIDWEATHER_LOG_DIR takes priority over logging_config.path in JSON."""
    json_log_dir = tmp_path / "json_logs"
    env_log_dir = tmp_path / "env_logs"

    original_config = cfg.get("logging_config")
    try:
        cfg.set(
            "logging_config",
            {"enabled": True, "filename": "test.log", "path": str(json_log_dir)},
        )
        monkeypatch.setenv("AIDWEATHER_LOG_DIR", str(env_log_dir))

        log_cfg = cfg.logging_config()
        assert log_cfg["filename"] == str(env_log_dir / "test.log")
    finally:
        cfg.set("logging_config", original_config)


def test_env_var_log_dir_file_logging_integration(monkeypatch, tmp_path):
    """Test that file logging writes to the AIDWEATHER_LOG_DIR location when enabled."""
    custom_log_dir = tmp_path / "active_log_dir"
    monkeypatch.setenv("AIDWEATHER_LOG_DIR", str(custom_log_dir))

    original_config = cfg.get("logging_config")
    logger = logging.getLogger("aidweather")
    original_handlers = list(logger.handlers)

    try:
        cfg.set(
            "logging_config",
            {"enabled": True, "filename": "aidweather.log", "level": "INFO"},
        )
        aidweather._configure_file_logging()

        logger.info("Test message for env var log dir")
        for handler in logger.handlers:
            handler.flush()

        expected_file = custom_log_dir / "aidweather.log"
        assert expected_file.exists()
        assert "Test message for env var log dir" in expected_file.read_text()
    finally:
        logger.handlers = original_handlers
        cfg.set("logging_config", original_config)
