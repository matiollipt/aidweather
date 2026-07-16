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
    """Wrap the bundled JSON configuration as a typed accessor object.

    Not intended for direct instantiation; use the module-level ``cfg`` singleton.
    """

    def __init__(self, data: Mapping) -> None:
        # _load_config_dict always returns a dict, so no `or {}` guard is needed.
        self._data = dict(data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return a nested value by dot-notation key path (e.g. ``"section.key"``).

        Returns *default* when the key path is absent. Correctly distinguishes a
        missing key from a key whose stored value is ``None`` via a private sentinel.
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
        """Return the base URL for *temporal_api* / *endpoint_type*, falling back to hardcoded defaults."""
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
        """Return a ``{code: name}`` mapping for the given parameter group."""
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

    def param_metadata(self, code: str | None = None) -> dict[str, Any]:
        """Get structured parameter metadata dictionary for *code* or all parameters if ``None``."""
        all_meta = self.get("param_metadata", default={})
        if not isinstance(all_meta, dict):
            all_meta = {}
        if code is not None:
            return dict(all_meta.get(code, {}))
        return dict(all_meta)

    def get_native_grid(self, code: str) -> tuple[float, float]:
        """Return (latitude_degrees, longitude_degrees) native grid resolution for parameter *code*.

        Falls back to default MERRA-2 grid (0.5°, 0.625°) if *code* is not registered or missing grid details.
        """
        meta = self.param_metadata(code)
        grid = meta.get("native_grid", {}) if isinstance(meta, dict) else {}
        lat_deg = grid.get("latitude_degrees", 0.5)
        lon_deg = grid.get("longitude_degrees", 0.625)
        return float(lat_deg), float(lon_deg)

    def cache_config(self) -> dict[str, Any]:
        """Return the effective cache configuration dict with path resolved via env, JSON, or XDG."""
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
        """Return the log configuration dict with filename resolved to an absolute path."""
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
