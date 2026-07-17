# SPDX-License-Identifier: Apache-2.0
"""
Centralized configuration management for ``aidweather``.

Provides a singleton ``cfg`` instance with dot-notation access to settings
loaded from ``config.json``. Falls back to hardcoded defaults if the file is
missing or malformed.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_log_dir

__all__ = ["cfg", "get_config"]

_logger = logging.getLogger(__name__)

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


# Sentinel used by _Config.get() to distinguish a missing key from a key whose value is None.
_MISSING = object()


def _load_config_dict() -> dict:
    """Load and return the bundled config.json, or an empty dict on failure."""

    try:
        ref = resources.files("aidweather") / "assets" / "config.json"
        with ref.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        # AttributeError covers Python < 3.9 where importlib.resources.files()
        # is not available and the / operator raises instead of FileNotFoundError.
        _logger.warning("Could not load internal config.json. Using defaults.")
        return {}


class _Config:
    """Typed accessor for the bundled JSON configuration.

    Wraps the dict loaded from ``assets/config.json`` and provides dot-notation
    key access, group-based parameter lookups, URL resolution, and derived
    configuration for caching, logging, and API limits.

    Not intended for direct instantiation; use the module-level :data:`cfg`
    singleton or :func:`get_config` instead.

    Attributes:
        _data: The raw configuration dict loaded from ``config.json``.
    """

    def __init__(self, data: Mapping) -> None:
        # _load_config_dict always returns a dict, so no `or {}` guard is needed.
        self._data = dict(data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return a nested value by dot-notation key path.

        Traverses the configuration dict using ``"."`` as the level separator
        (e.g. ``"cache_config.path"``). Returns *default* when the key path is
        absent or an intermediate level is not a mapping. Correctly distinguishes
        a missing key from a key whose stored value is ``None`` via a private
        sentinel object.

        Args:
            key_path: Dot-separated path to the target value, e.g.
                ``"section.subsection.key"``.
            default: Value to return when the key path is not found.

        Returns:
            The value at *key_path*, or *default* if absent.
        """
        value: Any = self._data
        for key in key_path.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(key, _MISSING)
            if value is _MISSING:
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
        """Return the base URL for *temporal_api* and *endpoint_type*.

        Looks up the URL in the ``base_urls`` section of the configuration
        first; falls back to the hardcoded :data:`_DEFAULT_URLS` constant if
        the key is missing or the temporal API is unrecognised.

        Args:
            temporal_api: Temporal resolution key — ``"daily"`` or ``"hourly"``.
            endpoint_type: Endpoint variant — ``"point"`` or ``"regional"``.
                Defaults to ``"point"``.

        Returns:
            The full base URL string for the requested combination.
        """
        urls = self._data.get("base_urls", {})
        if temporal_api in urls and endpoint_type in urls[temporal_api]:
            return str(self.get(f"base_urls.{temporal_api}.{endpoint_type}"))
        if temporal_api not in _DEFAULT_URLS:
            _logger.warning(
                "Unknown temporal_api '%s'; falling back to 'daily' URLs.", temporal_api
            )
        temporal_defaults = _DEFAULT_URLS.get(temporal_api, _DEFAULT_URLS["daily"])
        return str(temporal_defaults.get(endpoint_type, ""))

    def params(self, group: str = "default") -> dict[str, str]:
        """Return a ``{code: short_name}`` mapping for the given parameter group.

        Args:
            group: Parameter group key as defined in the ``params`` section of
                the configuration (e.g. ``"default"`` or ``"all"``). Falls
                back to the ``"default"`` group if the requested key is absent.

        Returns:
            A dict mapping parameter codes (e.g. ``"T2M"``) to their short
            names (e.g. ``"Temperature at 2 Meters"``). Returns an empty dict
            if neither the group nor the default group is found.
        """
        params_root = self._data.get("params", {}) or {}
        result = self.get(f"params.{group}", default=params_root.get("default", {}))
        return dict(result) if isinstance(result, dict) else {}

    def param_groups(self) -> list[str]:
        """List all available parameter group names."""
        params = self.get("params", default={})
        return list(params.keys()) if isinstance(params, dict) else []

    def param_descriptions(self) -> dict[str, str]:
        """Get a mapping of parameters to their full descriptions."""
        result = self.get("param_descriptions", default={})
        return dict(result) if isinstance(result, dict) else {}

    def param_metadata(self, params: list[str] | str | None = None) -> dict[str, Any]:
        """Return structured scientific metadata for *params* or all parameters.

        Args:
            params: One of:

                - ``None`` — returns the full metadata dict for all parameters.
                - A single parameter code string (e.g. ``"T2M"``) — returns
                  that parameter's metadata dict directly.
                - A list of parameter code strings — returns a dict mapping
                  each code to its metadata dict.

        Returns:
            A dict of parameter metadata. The exact structure depends on
            the ``param_metadata`` section of ``config.json``, but typically
            includes keys such as ``short_name``, ``units``, ``source``,
            ``native_grid``, and ``temporal_coverage``.
        """
        all_meta = self.get("param_metadata", default={})
        if not isinstance(all_meta, dict):
            all_meta = {}
        if params is not None:
            if isinstance(params, str):
                return dict(all_meta.get(params, {}))
            return {p: dict(all_meta.get(p, {})) for p in params}
        return dict(all_meta)

    def get_native_grid(self, community: str) -> tuple[float, float]:
        """Return the native grid resolution for *community* in decimal degrees.

        Intended for use with NASA POWER parameter codes (``"T2M"``,
        ``"ALLSKY_SFC_SW_DWN"``, etc.) that map to a distinct source product
        grid. Used internally by :class:`~aidweather.client.PowerClient` to
        derive minimum effective transect spacings.

        Args:
            community: A parameter code or NASA POWER community identifier
                (e.g. ``"T2M"`` for MERRA-2, ``"ALLSKY_SFC_SW_DWN"`` for CERES).

        Returns:
            A ``(latitude_degrees, longitude_degrees)`` tuple representing the
            native grid cell size. Falls back to MERRA-2 defaults
            ``(0.5, 0.625)`` if *community* is not registered or its grid
            details are missing.
        """
        meta = self.param_metadata(community)
        grid = meta.get("native_grid", {}) if isinstance(meta, dict) else {}
        lat_deg = grid.get("latitude_degrees", 0.5)
        lon_deg = grid.get("longitude_degrees", 0.625)
        return float(lat_deg), float(lon_deg)

    def cache_config(self) -> dict[str, Any]:
        """Return the effective cache configuration dict.

        Resolves the on-disk cache directory using the following priority order:

        1. ``AIDWEATHER_CACHE_DIR`` environment variable (highest priority).
        2. The ``cache_config.path`` key in ``config.json`` (resolved to an
           absolute path).
        3. The XDG user cache directory (``platformdirs.user_cache_dir``).

        All other keys from the ``cache_config`` section of ``config.json``
        (e.g. ``enabled``) are merged over the defaults. The ``path`` key in
        the returned dict always reflects the resolved effective path.

        Returns:
            A dict with at least ``"enabled"`` (bool) and ``"path"`` (str)
            keys, plus any additional keys from the JSON configuration section.
        """
        xdg_default = user_cache_dir("aidweather", appauthor=False)
        env_override = os.environ.get("AIDWEATHER_CACHE_DIR")
        json_path = self.get("cache_config.path")

        # Determine effective path — env var wins, then JSON (resolved if set), then XDG
        if env_override:
            effective_path = env_override
        elif json_path:
            effective_path = str(Path(json_path).resolve())
        else:
            effective_path = xdg_default

        defaults: dict[str, Any] = {
            "enabled": True,
            "path": effective_path,
        }
        json_section = self.get("cache_config")
        if not isinstance(json_section, dict):
            json_section = {}
        json_overrides = {k: v for k, v in json_section.items() if k != "path"}
        return {**defaults, **json_overrides}

    def logging_config(self) -> dict[str, Any]:
        """Return the effective logging configuration dict.

        Resolves the log file path using the following priority order:

        1. If the configured filename is already an absolute path, it is used
           as-is.
        2. Otherwise, the log directory is resolved via (in order):
           ``AIDWEATHER_LOG_DIR`` environment variable, the ``logging_config.path``
           key in ``config.json``, then ``platformdirs.user_log_dir``.
        3. The resolved log file path is ``<log_dir>/<filename>``.

        Returns:
            A dict with at least ``"enabled"`` (bool), ``"filename"`` (str,
            absolute path), and ``"level"`` (str) keys, plus any additional
            keys from the ``logging_config`` JSON section.
        """
        raw_config = self.get("logging_config", default={})
        if not isinstance(raw_config, dict):
            raw_config = {}
        filename = raw_config.get("filename", "aidweather.log")

        if Path(filename).is_absolute():
            resolved_path = filename
        else:
            env_override = os.environ.get("AIDWEATHER_LOG_DIR")
            json_path = raw_config.get("path")

            if env_override:
                log_dir = env_override
            elif json_path:
                log_dir = str(Path(json_path).resolve())
            else:
                log_dir = user_log_dir("aidweather", appauthor=False)

            resolved_path = str(Path(log_dir) / filename)

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
        result = self.get("api_limits", default={})
        return dict(result) if isinstance(result, dict) else {}


# --- Singleton ---
_config_data = _load_config_dict()
cfg = _Config(_config_data)


# --- Convenience functions ---


def get_config() -> _Config:
    """Return the singleton ``_Config`` instance.

    This is a stable public alias for the ``cfg`` module-level singleton,
    provided for dependency-injection and testing scenarios where callers
    prefer a function call over importing a module-level name directly.
    """
    return cfg
