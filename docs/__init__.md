# AidWeather — Package Overview

`aidweather` retrieves, caches, and validates daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/).

The main components are available from the top-level import:

```python
from aidweather import (
    PowerClient,           # NASA POWER API client with smart SQLite cache
    GeoCoordinate,         # Geographic coordinate (type-safe lat/lon)
    normalize_coord_input, # Accepts coordinates in DMS, DDM, DD strings or raw floats
    cfg,                   # Config object (singleton)
    get_config,            # Returns the config object (singleton)
    ensure_date_column,    # Convert date-like entries in DataFrame to datetime64[ns]
)
```

For a complete inventory of every class and function in the package, see the [API Inventory](api_inventory.md).

---

## Quick start

```python
from aidweather import PowerClient

client = PowerClient(temporal_api="daily")
df = client.get_point_data(
    lat=-23.55,
    lon=-46.63,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR"],
)
print(df.head())
```

---

## What each module does

| Module | Main export | Use it for |
|---|---|---|
| `client` | `PowerClient`, `PointRequest`, `ExpandedPointRequest` | Fetching weather data (single point, multi-point, transect, regional) |
| `geo` | `GeoCoordinate`, `normalize_coord_input`, coordinate parsers | Parsing, validating, and converting coordinates |
| `config` | `cfg`, `get_config` | Accessing API URLs, parameter catalogues, cache settings |
| `utils` | `ensure_date_column`, `DateColumnOptions` | Standardizing date columns before merging with your own data |
| `cli` | `aidweather` command handlers | Fetching, parameter lookup, cache management, and file output from the shell |

---

## Package metadata

```python
import aidweather

print(aidweather.__version__)
print(aidweather.__author__)
print(aidweather.__url__)
```

---

## Logging

`aidweather` uses Python's standard `logging` module internally and adds a `NullHandler` at the package level so you won't see unexpected output unless you configure a handler yourself.

To enable debug output in your own application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
