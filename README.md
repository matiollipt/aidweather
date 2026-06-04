# aidweather

**Weather data retrieval and validation from NASA POWER API**

`aidweather` is a Python library for fetching, caching, and validating daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). The main user interfaces are a simple Python API, a CLI (see below), and utilities for geospatial coordinate operations.

> [!WARNING]
> **Beta:** Public API and CLI output may change before 1.0. Please [report issues](https://github.com/matiollipt/aidweather/issues) — feedback is welcome.

> [!IMPORTANT]
> **NASA POWER Compliance:** Review the [License & Data Usage Guidelines](docs/NASA_POWER_Licence_Usage.md) and [API Usage & Guardrails](docs/aidweather_nasa_power_usage.md).

---

## Installation

### Quick Install (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash
```

Optional flags passed via `-s --`:

| Flag | Description |
|---|---|
| `--dev` | Also install developer tools |
| `--dev -y` | Developer tools, skip prompts |
| `--pipx` | Install globally in an isolated environment via pipx |

Example:

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --dev -y
```

### Local Development

```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
./install.sh --dev
```

Run the test suite:

```bash
uv run --with-editable . --extra test pytest -q
```

Before publishing, follow the [Release Checklist](docs/release_checklist.md).

## Quick Start

See the [Client Documentation](docs/client.md) for full details on the `PowerClient`.

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

# Create a small CSV file sites.csv with 3 points in São Paulo
cat << 'EOF' > sites.csv
lat,lon,name
-23.55,-46.63,Sao Paulo Center
-23.56,-46.64,Sao Paulo South
-23.54,-46.62,Sao Paulo North
EOF

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

# Cache management-
aidweather cache info
aidweather cache clear --yes
```

## Public API

The package exposes key modules for managing data integration. For a top-level overview, see the [\_\_init\_\_](docs/__init__.md) documentation.
For a complete inventory of all package classes and functions, see the [API Inventory](docs/api_inventory.md).

```python
from aidweather import (
    PowerClient,            # NASA POWER API client with SQLite cache. Read more: docs/client.md
    GeoCoordinate,          # Type-safe lat/lon value object. Read more: docs/geo.md
    normalize_coord_input,  # Parse DMS / DDM / DD coordinate strings. Read more: docs/geo.md
    cfg,                    # Singleton config (used by downstream packages). Read more: docs/config.md
    get_config,             # Convenience accessor for cfg. Read more: docs/config.md
    ensure_date_column,     # Robust DataFrame date column normalization. Read more: docs/utils.md
)
```

## Configuration & Assets

`aidweather` reads its settings from a bundled `assets/config.json`. The cache is stored in your platform's user cache directory by default:

| Platform | Default cache path |
|---|---|
| Linux | `~/.cache/aidweather/aidweather_cache.db` |
| macOS | `~/Library/Caches/aidweather/aidweather_cache.db` |

The cache is **shared across all your projects** — if you query the same location in two different scripts, the second call is instant.

To use a custom location, set an environment variable:

```bash
export AIDWEATHER_CACHE_DIR=/your/shared/cache
```

Check your current cache state:

```bash
aidweather cache info
```

For a full breakdown of the configuration files, see [Assets](docs/ASSETS.md) and [Config](docs/config.md).

Set your NASA POWER API key via environment variable or a local `.env` file:

```bash
# .env
NASA_POWER_API_KEY=your_key_here
```


## Citation & Attribution

When publishing analyses based on data retrieved with `aidweather`, cite NASA POWER as the data provider and mention the package version used in your workflow.

Suggested acknowledgement:

```text
Weather and solar data were obtained from the NASA POWER Project using aidweather v0.1.0.
```

## License

Apache-2.0. See [LICENSE](LICENSE).
