# SPDX-License-Identifier: Apache-2.0
"""
aidweather.config
~~~~~~~~~~~~~~~~~

Centralized configuration management for the `aidweather` package.

Loads configuration settings from the bundled ``config.json`` file (located in
``aidweather/assets/``) via ``importlib.resources``, ensuring the file is found
regardless of how the package is installed (editable, wheel, zip-imported, etc.).

Access is provided through a singleton instance ``cfg`` of the ``_Config`` class.
The ``cfg`` object provides typed, dot-notation access to all configuration sections.
If the JSON file is missing or invalid, all methods fall back to hardcoded defaults —
the package never fails to import because of a missing config file.

Ecosystem extension points
--------------------------
Downstream packages (``aidviz``, ``aidfarm``, etc.) can store their own defaults
under top-level keys in ``config.json``:

    {
        "aidviz":   {"default_theme": "dark", "dpi": 150},
        "aidfarm":  {"gdd_base_temp_c": 10.0}
    }

and retrieve them with ``cfg.get("aidviz.dpi")``.  ``aidweather.config.cfg``
is the single source of truth for the whole ecosystem.

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

        Merges built-in defaults with any overrides present in ``config.json``.
        The ``path`` default points to ``~/.aidweather_cache``.

        Returns:
            A dictionary of caching settings (``enabled``, ``path``, etc.).
        """
        defaults: dict[str, Any] = {
            "enabled": True,
            "path": os.path.expanduser("~/.aidweather_cache"),
        }
        return {**defaults, **self.get("cache_config", default={})}

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

    def visualization(self, key: str | None = None, default: Any = None) -> Any:
        """Returns the visualization configuration section or a specific key within it.

        The ``default_theme`` key is always ``None`` in ``aidweather``; theme
        resolution is the responsibility of ``aidviz``.

        Args:
            key: A specific key to retrieve from the visualization config.
            default: A default value to return if the key is not found.

        Returns:
            The requested visualization configuration value, or the full dict.
        """
        vis_config: dict[str, Any] = {"default_theme": None}
        from_json = self.get("visualization", default={})
        vis_config.update(from_json)

        if key:
            return vis_config.get(key, default)
        return vis_config

    def color_map(self) -> dict[str, str]:
        """Returns the canonical per-parameter color map for the ecosystem.

        Returns:
            A dictionary mapping NASA POWER parameter codes to hex color strings.
        """
        return dict(self.get("color_map", default={}))

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


def get_model_config() -> dict[str, Any]:
    """Loads and returns the default model configuration from ``model_config.json``.

    Returns:
        A dictionary of model hyperparameter defaults, or an empty dict
        if the file cannot be loaded.
    """
    try:
        ref = resources.files("aidweather") / "assets" / "model_config.json"
        with ref.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        logging.warning("Could not load model_config.json. Returning empty dict.")
        return {}
