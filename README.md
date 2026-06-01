# aidweather

**Weather data retrieval and validation from NASA POWER API**

`aidweather` is a Python library for fetching, caching, and validating daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). The main user interfaces are a simple Python API, a CLI (see below), and utilities for geospatial coordinate operations.

> [!IMPORTANT]
> **NASA POWER Compliance:** Please review the [NASA POWER License and Data Usage Guidelines](docs/NASA_POWER_Licence_Usage.md) and our [API Usage & Guardrails](docs/aidweather_nasa_power_usage.md) before using this tool in production.

For a high-level understanding of how `aidweather` transforms geographic points into validated weather data, refer to the [Workflow Guide](docs/aidweather_workflow_guide.md).

---

## Beta Status

`aidweather` is in beta. The public API exposed from `aidweather.__init__` is intended to be usable for early scientific and agricultural workflows, but details around CLI output, cache internals, and future ecosystem integrations may still change before a stable 1.0 release. Please report confusing behavior, missing NASA POWER parameters, and documentation gaps in the project issue tracker.

NASA POWER remains the authoritative source for data availability, parameter definitions, and service limits. API keys can improve reliability and attribution, but they do not remove NASA POWER usage policies or rate limits.

## Installation

### 1. Installation Script (Linux / macOS)
The installation script below automatically detects your environment, creates a virtual environment and installs `aidweather` (with optional developer tools).

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash
```

You can pass arguments directly to the installer script through curl using `-s --`. For example, to install developer tools and skip interactive prompts:

- Install developer tools: `--dev`

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --dev -y
```

- Install globally via pipx: `--pipx`
```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --pipx
```

If installed using pipx, you can use:

```bash
aidweather fetch --lat -23.55 --lon -46.63 --start 2023-01-01 --end 2023-12-31 --resolution daily --elevation 800 --params T2M,PRECTOTCORR,RH2M --output weather_data.csv --format csv --summarize
```

### 2. Global CLI Installation (pipx)
If you want to use the `aidweather` CLI globally without manually managing virtual environments or risking dependency conflicts, use [pipx](https://github.com/pypa/pipx) to install it in an isolated user-level environment:

```bash
pipx install aidweather
```

You can also install it via the installer script using the `--pipx` flag, which can even be done remotely via curl:
```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --pipx
```

### 3. Standard Pip Installation
If you prefer standard package managers to manage your own virtual environment, install directly from PyPI:

```bash
pip install aidweather
```

### 4. Local Development Installation
For custom development, clone the repository and run the setup script locally.

```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
./install.sh --dev
```

Run the beta test gate with:

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

## Ecosystem

`aidweather` is the foundation of the `aid*` toolkit:

| Package | Purpose |
|---|---|
| **`aidweather`** ← you are here | Weather data retrieval, caching, validation |
| `aidviz` *(coming soon)* | Custom agricultural weather plots |
| `aidfarm` *(coming soon)* | EDA, feature engineering, ML for farm data |

The future `aidviz` and `aidfarm` packages are roadmap items. Their package extras are reserved in the metadata, but they do not install additional dependencies in this beta.

## Citation & Attribution

When publishing analyses based on data retrieved with `aidweather`, cite NASA POWER as the data provider and mention the package version used in your workflow.

Suggested acknowledgement:

```text
Weather and solar data were obtained from the NASA POWER Project using aidweather v0.1.0.
```

## License

Apache-2.0. See [LICENSE](LICENSE).
