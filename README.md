# aidweather

**Weather data retrieval and validation from NASA POWER API**

`aidweather` is a Python library for fetching, caching, and validating daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). The main user interfaces are a simple Python API, a CLI (see below), and utilities for geospatial coordinate operations.

> [!WARNING]
> **Beta:** Public API and CLI output may change before 1.0. Please [report issues](https://github.com/matiollipt/aidweather/issues) — feedback is welcome.

> [!IMPORTANT]
> **NASA POWER Compliance:** Review the [License & Data Usage Guidelines](docs/NASA_POWER_Licence_Usage.md) and [API Usage & Guardrails](docs/aidweather_nasa_power_usage.md).
> **Choosing a data source:** See the [Data Source Comparison](docs/data_source_comparison.md) for a detailed comparison of NASA POWER vs. Meteostat (and future sources), with installation instructions, known limitations, and a decision guide.

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
| `--uv-tool` | Install globally in an isolated environment via uv tool |

Example:

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --dev -y
```

#### Install system-wide via `uv tool`

To install system-wide in an isolated environment using `uv tool`:

```bash
# Via curl installation script:
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --uv-tool -y

# Or directly using uv:
uv tool install git+https://github.com/matiollipt/aidweather.git
```

### Quick Install (Windows PowerShell)

```powershell
# Default install in .venv:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/matiollipt/aidweather/main/install.ps1))) -Yes

# Install with developer tools:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/matiollipt/aidweather/main/install.ps1))) -Dev -Yes

# Install system-wide as a uv tool:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/matiollipt/aidweather/main/install.ps1))) -UvTool -Yes
```

Optional parameters:

| Parameter | Description |
|---|---|
| `-Dev` | Also install developer tools |
| `-UvTool` | Install globally in an isolated environment via uv tool |
| `-NoVenv` | Skip venv creation (use active/global Python) |
| `-VenvPath DIR` | Custom venv path (default: `.venv`) |
| `-Clean` | Wipe and recreate venv before installing |
| `-Yes` | Skip confirmation prompts |

### Local Development

#### Unix/macOS
```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
./install.sh --dev
```

#### Windows (PowerShell)
```powershell
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
.\install.ps1 -Dev
```


Run the test suite:

```bash
uv run --with-editable . --extra test pytest -q
```

By default, the test suite skips live API calls to prevent service spam. To run the live integration tests against the actual NASA POWER endpoints, set the `AIDWEATHER_RUN_LIVE_TESTS` environment variable to `1`:

```bash
AIDWEATHER_RUN_LIVE_TESTS=1 uv run --with-editable . --extra test pytest
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

### 1D transect

Fetch data along a straight-line path between two endpoints. Sampling is controlled
by `num_points` or `spacing_km`; the 0.5° NASA POWER grid resolution (~55 km) is
enforced as a minimum spacing:

```python
from aidweather import GeoCoordinate

coord_a = GeoCoordinate.from_decimal(-25.0, -48.0)
coord_b = GeoCoordinate.from_decimal(-20.0, -48.0)  # ~555 km north

df = client.get_transect_data(
    start_coord=coord_a,
    end_coord=coord_b,
    start="2023-01-01", end="2023-01-31",
    params=["T2M", "PRECTOTCORR"],
    num_points=5,
)
print(df.head())  # DataFrame with lat, lon columns + parameters
```

### Regional grid data

Fetch data on a 0.5° × 0.5° grid within a bounding box (max 4.5° × 4.5°, one parameter per request):

```python
df = client.get_regional_data(
    lat_min=-23.5, lat_max=-20.0,
    lon_min=-47.0, lon_max=-44.0,
    start="2023-01-01", end="2023-01-31",
    params=["T2M"],
)
print(df.head())  # DataFrame with lat, lon, elevation, and T2M columns
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

# Fetch along a spatial transect (start and end coordinates)
aidweather fetch-transect \
    --lat-start -25.0 --lon-start -48.0 \
    --lat-end   -20.0 --lon-end   -48.0 \
    --start 2023-01-01 --end 2023-01-31 \
    --num-points 5

# Fetch regional grid data for a bounding box
aidweather fetch-regional --lat-min -23.5 --lat-max -20.0 \
    --lon-min -47.0 --lon-max -44.0 \
    --start 2023-01-01 --end 2023-01-31 \
    --params T2M --output regional.csv

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
To understand how NASA POWER compares to alternative weather data sources (Meteostat, ERA5, etc.) and how to choose among them, see the [Data Source Comparison](docs/data_source_comparison.md).

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

## Citation & Attribution

When publishing analyses based on data retrieved with `aidweather`, cite NASA POWER as the data provider and mention the package version used in your workflow.

Suggested acknowledgement:

```text
Weather and solar data were obtained from the NASA POWER Project using aidweather v0.1.2.
```

## License

Apache-2.0. See [LICENSE](LICENSE).
