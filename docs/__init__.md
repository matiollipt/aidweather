# aidweather — Package Overview

`aidweather` retrieves, caches, and validates daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). It is designed as the foundational data ingestion layer for the `aid*` toolkit.

Everything you need for normal use is available directly from the top-level import:

```python
from aidweather import (
    PowerClient,           # NASA POWER API client with SQLite cache
    GeoCoordinate,         # Type-safe lat/lon value object
    normalize_coord_input, # Accepts DMS, DDM, DD strings or raw floats
    cfg,                   # Singleton config object
    get_config,            # Alternative accessor for cfg
    ensure_date_column,    # Robust DataFrame date column normalization
)
```

For a complete callable-by-callable inventory of every class and function in the
package, including internal helpers and CLI handlers, see the
[API Inventory](api_inventory.md).

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
| `config` | `cfg`, `get_config`, `get_model_config` | Accessing API URLs, parameter catalogues, cache settings |
| `utils` | `ensure_date_column`, `DateColumnOptions` | Standardizing date columns before merging with your own data |
| `cli` | `aidweather` command handlers | Fetching, parameter lookup, cache management, and file output from the shell |

---

## Package metadata

```python
import aidweather

print(aidweather.__version__)  # e.g., "0.1.0"
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
