# SPDX-License-Identifier: Apache-2.0
import importlib.resources
import json
import pathlib

from aidweather.config import _load_config_dict, cfg


def test_config_methods_group_and_description():
    config = cfg
    # param_groups returns list of keys in params section
    groups = config.param_groups()
    assert isinstance(groups, list)
    # default should always be present
    assert "default" in groups
    # param_descriptions returns mapping of codes to descriptions
    desc = config.param_descriptions()
    assert isinstance(desc, dict)


def test_config_logging_defaults():
    config = cfg
    # logging_config should have enabled flag and level
    log_cfg = config.logging_config()
    assert isinstance(log_cfg, dict)
    assert "enabled" in log_cfg
    assert "level" in log_cfg


def test_config_load_real_file(tmp_path, monkeypatch):
    # create a temporary config.json and patch resource path
    data = {"base_urls": {"daily": {"point": "http://example.com"}}}
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "config.json").write_text(json.dumps(data))

    # monkeypatch the resource to point to our temp dir
    def fake_files(name):
        if name == "aidweather":
            return pathlib.Path(tmp_path)
        return importlib.resources.files(name)

    monkeypatch.setattr(importlib.resources, "files", fake_files)
    loaded = _load_config_dict()
    assert loaded["base_urls"]["daily"]["point"] == "http://example.com"


def test_config_set_method():
    # Test cfg.set on dynamic values
    from aidweather.config import cfg

    original_val = cfg.get("cache_config.path")

    cfg.set("cache_config.path", "relative_test_cache")
    assert cfg.get("cache_config.path") == "relative_test_cache"

    # Test setting a deeply nested new key path
    cfg.set("test_section.deep.key", 42)
    assert cfg.get("test_section.deep.key") == 42

    # Clean up / reset
    cfg.set("cache_config.path", original_val)


def test_config_relative_path_resolution(monkeypatch):
    from aidweather.config import cfg

    # Clear env var if set
    monkeypatch.delenv("AIDWEATHER_CACHE_DIR", raising=False)

    original_val = cfg.get("cache_config.path")
    try:
        cfg.set("cache_config.path", "my_relative/cache_dir")
        cache_cfg = cfg.cache_config()
        import os

        assert cache_cfg["path"] == os.path.abspath("my_relative/cache_dir")
    finally:
        cfg.set("cache_config.path", original_val)


def test_config_logging_path_resolution(monkeypatch):
    import os

    from aidweather.config import cfg

    # 1. Clear environment variable
    monkeypatch.delenv("AIDWEATHER_LOG_DIR", raising=False)

    # Backup original config
    original_config = cfg.get("logging_config")

    try:
        # 2. Test relative filename resolution with default (no custom path, no env override)
        cfg.set(
            "logging_config", {"enabled": True, "filename": "test.log", "path": None}
        )
        log_cfg = cfg.logging_config()
        # Should resolve using user_log_dir
        from platformdirs import user_log_dir

        expected_dir = user_log_dir("aidweather", appauthor=False)
        assert log_cfg["filename"] == os.path.join(expected_dir, "test.log")

        # 3. Test absolute filename (should not be overridden or modified)
        abs_path = os.path.abspath("some_absolute_file.log")
        cfg.set("logging_config", {"enabled": True, "filename": abs_path})
        log_cfg = cfg.logging_config()
        assert log_cfg["filename"] == abs_path

        # 4. Test custom path in JSON (should resolve relative to custom path)
        cfg.set(
            "logging_config",
            {"enabled": True, "filename": "test.log", "path": "custom_log_dir"},
        )
        log_cfg = cfg.logging_config()
        assert log_cfg["filename"] == os.path.join(
            os.path.abspath("custom_log_dir"), "test.log"
        )

        # 5. Test environment variable override
        monkeypatch.setenv("AIDWEATHER_LOG_DIR", "env_log_dir")
        log_cfg = cfg.logging_config()
        assert log_cfg["filename"] == os.path.join("env_log_dir", "test.log")

    finally:
        cfg.set("logging_config", original_config)
