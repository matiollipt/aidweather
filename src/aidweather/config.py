# SPDX-License-Identifier: Apache-2.0
"""
Centralized configuration management for `aidweather`.

Provides a singleton ``cfg`` instance offering dot-notation access
to settings loaded from ``config.json`` via ``importlib.resources``. If
the config file is missing or malformed, it falls back to hardcoded
defaults to ensure the package remains importable.

Example:
    >>> from aidweather.config import cfg
    >>> daily_point_url = cfg.get_url("daily", "point")
    >>> print(daily_point_url)
    >>> cache_settings = cfg.cache_config()
    >>> print(cache_settings["enabled"])
"""

import json
import logging
import os
from collections.abc import Mapping
from importlib import resources
from typing import Any

from platformdirs import user_cache_dir

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
    """Load configuration from the bundled config.json file.

    Returns:
        dict: The parsed configuration dictionary, or an empty
            dictionary if the file is missing or invalid.
    """

    try:
        ref = resources.files("aidweather") / "assets" / "config.json"
        with ref.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        logging.warning("Could not load internal config.json. Using defaults.")
        return {}


class _Config:
    """Wrap the bundled JSON configuration as a typed object.

    Not intended to be instantiated directly; use the module-level
    singleton ``cfg`` for access.

    Attributes:
        _data: The dictionary holding the loaded configuration.
    """

    def __init__(self, data: Mapping) -> None:
        """Initialize the config object with the provided data."""
        self._data = dict(data or {})

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a nested value using dot-notation.

        Args:
            key_path: A dot-separated string (e.g., "section.key").
            default: Value to return if the path is not found.

        Returns:
            The value at key_path or default if missing.
        """
        value = self._data
        for key in key_path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set a nested value using dot-notation.

        Args:
            key_path: A dot-separated string (e.g., "section.key").
            value: The value to assign to the key.
        """
        parts = key_path.split(".")
        target = self._data
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    def get_url(self, temporal_api: str, endpoint_type: str = "point") -> str:
        """Get the base URL for a specific temporal API and endpoint type.

        Returns:
            str: The resolved URL or hardcoded default if not in config.
        """
        urls = self._data.get("base_urls", {})
        if temporal_api in urls and endpoint_type in urls[temporal_api]:
            return str(self.get(f"base_urls.{temporal_api}.{endpoint_type}"))
        temporal_defaults = _DEFAULT_URLS.get(temporal_api, _DEFAULT_URLS["daily"])
        return str(temporal_defaults.get(endpoint_type, ""))

    def params(self, group: str = "default") -> dict[str, str]:
        """Get a mapping of parameter codes to human-readable names.

        Returns:
            dict: A dictionary of {code: name} for the specified group.
        """
        params_root = self._data.get("params", {}) or {}
        return dict(self.get(f"params.{group}", default=params_root.get("default", {})))

    def param_groups(self) -> list[str]:
        """List all available parameter group names."""
        return list(self.get("params", default={}).keys())

    def param_descriptions(self) -> dict[str, str]:
        """Get a mapping of parameters to their full descriptions."""
        return dict(self.get("param_descriptions", default={}))

    def cache_config(self) -> dict[str, Any]:
        """Get the caching configuration.

        Returns:
            dict: Config including path (resolved via env, json, or xdg).
        """
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
        """Get the log configuration settings."""
        defaults: dict[str, Any] = {
            "enabled": False,
            "filename": "aidweather.log",
            "level": "INFO",
        }
        return {**defaults, **self.get("logging_config", default={})}

    def api_limits(self) -> dict[str, Any]:
        """Get the configuration for API limits."""
        return dict(self.get("api_limits", default={}))


# --- Singleton ---
_config_data = _load_config_dict()
cfg = _Config(_config_data)


# --- Convenience functions ---


def get_config() -> _Config:
    """Get the singleton ``_Config`` instance."""
    return cfg
