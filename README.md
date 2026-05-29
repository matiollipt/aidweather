# aidweather

**Weather data retrieval and validation for agricultural applications.**

`aidweather` is a focused, production-grade Python library for fetching, caching, and validating daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). It provides a clean public API, a local SQLite cache, a Typer/Rich CLI, and robust geospatial coordinate utilities.

---

## Installation

```bash
pip install aidweather
```

## Quick Start

```python
from aidweather import PowerClient

client = PowerClient(temporal_api="daily")
df = client.get_point_data(
    lat=-23.55,
    lon=-46.63,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR", "RH2M"],
)
print(df.head())
```

## CLI

```bash
# Fetch weather data for a single point
aidweather fetch --lat -23.55 --lon -46.63 \
    --start 2023-01-01 --end 2023-12-31 \
    --params T2M,PRECTOTCORR --output data.csv

# Fetch for multiple points from a CSV file
aidweather fetch-multi --points-file sites.csv \
    --start 2023-01-01 --end 2023-12-31 --output multi.parquet --format parquet

# Fetch along a spatial transect
aidweather fetch-transect --lat -23.55 --lon -46.63 \
    --start 2023-01-01 --end 2023-01-31 \
    --axis lat --distance-km 50 --num-points 10

# List available NASA POWER parameters
aidweather params list --group all

# Describe a specific parameter
aidweather params describe T2M

# Cache management
aidweather cache info
aidweather cache clear --yes
```

## Public API

```python
from aidweather import (
    PowerClient,        # NASA POWER API client with SQLite cache
    GeoCoordinate,      # Type-safe lat/lon value object
    normalize_coord_input,  # Parse DMS / DDM / DD coordinate strings
    cfg,                # Singleton config (used by downstream packages)
    get_config,         # Convenience accessor for cfg
    ensure_date_column, # Robust DataFrame date column normalization
)
```

## Configuration

`aidweather` reads its settings from a bundled `assets/config.json`. The cache is stored by default at `~/.aidweather_cache/aidweather_cache.db`.

Set your NASA POWER API key via environment variable or a local `.env` file:

```bash
# .env
NASA_POWER_API_KEY=your_key_here
```

## Ecosystem

`aidweather` is the foundation of the `aid*` toolkit:

| Package | Purpose |
|---|---|
| **`aidweather`** ← you are here | Weather data retrieval, caching, validation |
| `aidviz` *(coming soon)* | Custom agricultural weather plots |
| `aidfarm` *(coming soon)* | EDA, feature engineering, ML for farm data |

Install future packages alongside `aidweather`:

```bash
pip install aidweather[aidviz]
pip install aidweather[aidfarm]
pip install aidweather[all]
```

## License

Apache-2.0. See [LICENSE](LICENSE).
