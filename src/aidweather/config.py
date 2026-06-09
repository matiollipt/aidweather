# SPDX-License-Identifier: Apache-2.0
"""
Centralized configuration management for the `aidweather` package.

Loads configuration settings from the bundled ``config.json`` file (located in
``aidweather/assets/``) via ``importlib.resources``, ensuring the file is found
regardless of how the package is installed (editable, wheel, zip-imported, etc.).

Access is provided through a singleton instance ``cfg`` of the ``_Config`` class.
The ``cfg`` object provides typed, dot-notation access to all configuration sections.
If the JSON file is missing or invalid, all methods fall back to hardcoded defaults —
the package never fails to import because of a missing config file.

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
    """Loads the configuration dictionary from the bundled ``config.json``.

    Uses ``importlib.resources`` to locate the asset file robustly, regardless
    of how the package is installed (wheel, editable install, zip import, etc.).

    Returns:
        A dictionary containing the configuration settings, or an empty dict
        if the file cannot be found or parsed.
    """
    try:
        ref = resources.files("aidweather") / "assets" / "config.json"
        with ref.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        logging.warning("Could not load internal config.json. Using defaults.")
        return {}


class _Config:
    """A thin, typed wrapper over the bundled JSON config.

    Not intended to be instantiated directly by users. Use the module-level
    singleton ``cfg`` for all access.

    Attributes:
        _data: The dictionary holding the loaded configuration.
    """

    def __init__(self, data: Mapping) -> None:
        """Initializes the _Config object.

        Args:
            data: A dictionary-like object containing the configuration data.
        """
        self._data = dict(data or {})

    def get(self, key_path: str, default: Any = None) -> Any:
        """Accesses a nested value using dot notation.

        Example:
            >>> cfg.get("base_urls.daily.point")

        Args:
            key_path: A dot-separated string representing the path to the
                nested key (e.g., ``"section.subsection.key"``).
            default: The value to return if the key is not found.

        Returns:
            The configuration value if found, otherwise ``default``.
        """
        value = self._data
        for key in key_path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Sets a nested value using dot notation, creating intermediate dictionaries if needed.

        Example:
            >>> cfg.set("cache_config.path", "/my/custom/path")

        Args:
            key_path: A dot-separated string representing the path to the
                nested key (e.g., ``"section.subsection.key"``).
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
        """Returns the base URL for a specified temporal API and endpoint type.

        Falls back to hardcoded NASA POWER defaults if the key is not present
        in the loaded configuration.

        Args:
            temporal_api: The temporal resolution, either ``"daily"`` or ``"hourly"``.
            endpoint_type: The endpoint type, either ``"point"`` or ``"regional"``.

        Returns:
            The corresponding API endpoint URL string.
        """
        urls = self._data.get("base_urls", {})
        if temporal_api in urls and endpoint_type in urls[temporal_api]:
            return str(self.get(f"base_urls.{temporal_api}.{endpoint_type}"))
        temporal_defaults = _DEFAULT_URLS.get(temporal_api, _DEFAULT_URLS["daily"])
        return str(temporal_defaults.get(endpoint_type, ""))

    def params(self, group: str = "default") -> dict[str, str]:
        """Returns a ``{code: long_name}`` mapping for a parameter group.

        If the requested group does not exist, falls back to ``"default"``.

        Args:
            group: The parameter group to retrieve (e.g., ``"all"``, ``"default"``).

        Returns:
            A dictionary mapping NASA POWER parameter codes to human-readable names.
        """
        params_root = self._data.get("params", {}) or {}
        return dict(self.get(f"params.{group}", default=params_root.get("default", {})))

    def param_groups(self) -> list[str]:
        """Returns a list of available parameter group names.

        Returns:
            A list of keys from the ``"params"`` section of the config.
        """
        return list(self.get("params", default={}).keys())

    def param_descriptions(self) -> dict[str, str]:
        """Returns a dictionary of full agronomic descriptions per parameter code.

        Returns:
            A dictionary mapping NASA POWER parameter codes to their full descriptions.
        """
        return dict(self.get("param_descriptions", default={}))

    def cache_config(self) -> dict[str, Any]:
        """Returns the caching configuration dictionary.

        Path resolution priority (highest to lowest):

        1. ``AIDWEATHER_CACHE_DIR`` environment variable.
        2. A ``path`` key in ``config.json`` or configured in script.
        3. Platform-appropriate user cache directory
           (``~/.cache/aidweather`` on Linux, ``~/Library/Caches/aidweather``
           on macOS, ``%LOCALAPPDATA%/aidweather/Cache`` on Windows).

        Returns:
            A dictionary of caching settings (``enabled``, ``path``, etc.).
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
        """Returns the logging configuration dictionary.

        Returns:
            A dictionary of logging settings (``enabled``, ``filename``, ``level``).
        """
        defaults: dict[str, Any] = {
            "enabled": False,
            "filename": "aidweather.log",
            "level": "INFO",
        }
        return {**defaults, **self.get("logging_config", default={})}

    def api_limits(self) -> dict[str, Any]:
        """Returns the NASA POWER API constraint configuration.

        Returns:
            A dictionary of API limits (e.g., max parameters per request).
        """
        return dict(self.get("api_limits", default={}))


# --- Singleton ---
_config_data = _load_config_dict()
cfg = _Config(_config_data)


# --- Convenience functions ---


def get_config() -> _Config:
    """Returns the singleton ``_Config`` instance.

    Returns:
        The module-level ``cfg`` singleton.
    """
    return cfg
