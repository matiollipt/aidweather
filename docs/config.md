# config

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## Purpose
Provides a centralized configuration management system for the package. It loads settings from `assets/config.json` and exposes them via a singleton object `cfg`. It handles API endpoints, caching preferences, logging, and parameter maps.

## Key responsibilities
- Loading and parsing the JSON configuration.
- Providing type-safe accessors for config sections.
- Defining default values for critical settings (like API URLs).
- Managing access to package resources.

## Public API

### Objects
- `cfg`: The singleton instance of `_Config` used to access settings.

### Classes
- `_Config`: (Internal use, but exposed via `cfg`)
  - `get(key_path: str, default: Any = None) -> Any`: Access nested config values using dot notation.
  - `get_url(temporal_api: str, endpoint_type: str = "point") -> str`: Returns the NASA POWER API URL.
  - `params(group: str = "default") -> Dict[str, str]`: Returns parameter mappings for a given group.
  - `cache_config() -> Dict[str, object]`: Returns caching configuration (enabled, path).
  - `logging_config() -> Dict[str, object]`: Returns logging configuration (enabled, filename, level).
  - `api_limits() -> Dict[str, object]`: Returns the NASA POWER API constraint configuration.

## Data flow and dependencies
- **Internal imports**: `json`, `logging`, `os`, `importlib.resources`.
- **Assets**: Loads `aidweather/assets/config.json`.
- **Downstream**: Used by `client.PowerClient` and `cli`.

## Configuration and assets
- **`assets/config.json`**: Primary source of settings.
- **Defaults**: Hardcoded fallbacks for API URLs if the JSON is missing or incomplete.

## Minimal usage example
```python
from aidweather.config import cfg

# Get API URL
url = cfg.get_url("daily", "point")

# Get cache settings
if cfg.cache_config().get("enabled"):
    print("Caching is enabled")
```
