# __init__

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## Purpose
The top-level entry point for the `aidweather` package. It exposes the primary public-facing classes and functions, making them available directly from the `aidweather` namespace.

## Key responsibilities
- Exposing the core API for weather retrieval and data formatting.
- Defining package metadata (`__version__`, `__author__`, `__url__`).

## Public API

### Classes
- `PowerClient`: (Imported from `.client`) Client for fetching NASA POWER weather data.
- `GeoCoordinate`: (Imported from `.geo`) Represents a geographic coordinate in decimal degrees.

### Functions
- `normalize_coord_input()`: (Imported from `.geo`) Normalizes various common coordinate input formats into a GeoCoordinate object.
- `ensure_date_column()`: (Imported from `.utils`) Robustly finds, parses, and standardizes a datetime column in a pandas DataFrame.
- `get_config()`: (Imported from `.config`) Returns the singleton config instance.

### Objects
- `cfg`: (Imported from `.config`) Singleton configuration object.

### Constants
- `__version__`: Current package version.
- `__author__`: Package author.

## Data flow and dependencies
- **Internal imports**: Imports from submodules (`client`, `config`, `geo`, `utils`) to expose them.
- **External dependencies**: `logging`.

## Minimal usage example
```python
import aidweather

# Access core classes
client = aidweather.PowerClient(temporal_api="daily")
print(f"AidWeather version: {aidweather.__version__}")
```
