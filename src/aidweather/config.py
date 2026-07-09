# SPDX-License-Identifier: Apache-2.0
"""
Centralized configuration management for ``aidweather``.

Provides a singleton ``cfg`` instance with dot-notation access to settings
loaded from ``config.json``. Falls back to hardcoded defaults if the file is
missing or malformed.
"""

import json
import logging
import os
from collections.abc import Mapping
from importlib import resources
from typing import Any

from platformdirs import user_cache_dir, user_log_dir

# Hardcoded defaults — used only if config.json is absent or unreadable
_DEFAULT_URLS = {
    "daily": {
        "point": "https://power.larc.nasa.gov/api/temporal/daily/point",
        "regional": "https://power.larc.nasa.gov/api/temporal/daily/regional",
    },
    "hourly": {
        "point": "https://power.larc.nasa.gov/api/temporal/hourly/point",
        "regional": "https://power.larc.nasa.gov/api/temporal/hourly/regional",
    },
}


def _load_config_dict() -> dict:
    """Load and return the bundled config.json, or an empty dict on failure."""

    try:
        ref = resources.files("aidweather") / "assets" / "config.json"
        with ref.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        logging.warning("Could not load internal config.json. Using defaults.")
        return {}


class _Config:
    """Wrap the bundled JSON configuration as a typed accessor object.

    Not intended for direct instantiation; use the module-level ``cfg`` singleton.
    """

    def __init__(self, data: Mapping) -> None:
        self._data = dict(data or {})

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return a nested value by dot-notation key path (e.g. ``"section.key"``)."""
        value: Any = self._data
        for key in key_path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set a nested value by dot-notation key path (e.g. ``"section.key"``)."""
        parts = key_path.split(".")
        target = self._data
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    def get_url(self, temporal_api: str, endpoint_type: str = "point") -> str:
        """Return the base URL for *temporal_api* / *endpoint_type*, falling back to hardcoded defaults."""
        urls = self._data.get("base_urls", {})
        if temporal_api in urls and endpoint_type in urls[temporal_api]:
            return str(self.get(f"base_urls.{temporal_api}.{endpoint_type}"))
        temporal_defaults = _DEFAULT_URLS.get(temporal_api, _DEFAULT_URLS["daily"])
        return str(temporal_defaults.get(endpoint_type, ""))

    def params(self, group: str = "default") -> dict[str, str]:
        """Return a ``{code: name}`` mapping for the given parameter group."""
        params_root = self._data.get("params", {}) or {}
        return dict(self.get(f"params.{group}", default=params_root.get("default", {})))

    def param_groups(self) -> list[str]:
        """List all available parameter group names."""
        return list(self.get("params", default={}).keys())

    def param_descriptions(self) -> dict[str, str]:
        """Get a mapping of parameters to their full descriptions."""
        return dict(self.get("param_descriptions", default={}))

    def cache_config(self) -> dict[str, Any]:
        """Return the effective cache configuration dict with path resolved via env, JSON, or XDG."""
        xdg_default = user_cache_dir("aidweather", appauthor=False)
        env_override = os.environ.get("AIDWEATHER_CACHE_DIR")
        json_path = self.get("cache_config.path")

        # Determine effective path — env var wins, then JSON (resolved if set), then XDG
        if env_override:
            effective_path = env_override
        elif json_path:
            effective_path = os.path.abspath(json_path)
        else:
            effective_path = xdg_default

        defaults: dict[str, Any] = {
            "enabled": True,
            "path": effective_path,
        }
        json_overrides = {
            k: v for k, v in (self.get("cache_config") or {}).items() if k != "path"
        }
        return {**defaults, **json_overrides}

    def logging_config(self) -> dict[str, Any]:
        """Return the log configuration dict with filename resolved to an absolute path."""
        raw_config = self.get("logging_config", default={})
        filename = raw_config.get("filename", "aidweather.log")

        if os.path.isabs(filename):
            resolved_path = filename
        else:
            env_override = os.environ.get("AIDWEATHER_LOG_DIR")
            json_path = raw_config.get("path")

            if env_override:
                log_dir = env_override
            elif json_path:
                log_dir = os.path.abspath(json_path)
            else:
                log_dir = user_log_dir("aidweather", appauthor=False)

            resolved_path = os.path.join(log_dir, filename)

        defaults: dict[str, Any] = {
            "enabled": False,
            "filename": resolved_path,
            "level": "INFO",
        }
        json_overrides = {
            k: v for k, v in raw_config.items() if k not in ("filename", "path")
        }
        return {**defaults, **json_overrides}

    def api_limits(self) -> dict[str, Any]:
        """Return the API limits configuration dict."""
        return dict(self.get("api_limits", default={}))


# --- Singleton ---
_config_data = _load_config_dict()
cfg = _Config(_config_data)


# --- Convenience functions ---


def get_config() -> _Config:
    """Return the singleton ``_Config`` instance."""
    return cfg
