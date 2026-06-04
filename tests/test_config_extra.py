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
