# aidweather

**Weather data retrieval and validation for agricultural applications.**

`aidweather` is a focused, production-grade Python library for fetching, caching, and validating daily and hourly meteorological data from [NASA's POWER API](https://power.larc.nasa.gov/). It provides a clean public API, a local SQLite cache, a Typer/Rich CLI, and robust geospatial coordinate utilities.

> [!IMPORTANT]
> **NASA POWER Compliance:** Please review the [NASA POWER License and Data Usage Guidelines](docs/NASA_POWER_Licence_Usage.md) and our [API Usage & Guardrails](docs/aidweather_nasa_power_usage.md) before using this tool in production.

For a high-level understanding of how `aidweather` transforms geographic points into validated weather data, refer to the [Workflow Guide](docs/aidweather_workflow_guide.md).

---

## Installation

### 1. Quick One-Liner (Linux / macOS)
For a streamlined, production-grade setup that automatically detects your environment, creates a virtual environment, and installs `aidweather` (with optional developer tools) in a single command, run:

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash
```

> [!TIP]
> You can pass arguments directly to the installer script through curl using `-s --`. For example, to install developer tools and skip interactive prompts:
> ```bash
> curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --dev -y
> ```

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
For custom development, clone the repository and run the setup script locally. For more details on the testing suite, see the [Test Coverage](docs/TEST_COVERAGE.md) document.

```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
./install.sh --dev
```

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

The package exposes key modules for managing data integration. For a top-level overview, see the [\_\_init\_\_](docs/__init__.md) documentation.

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

`aidweather` reads its settings from a bundled `assets/config.json`. The cache is stored by default at `~/.aidweather_cache/aidweather_cache.db`.

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

Install future packages alongside `aidweather`:

```bash
pip install aidweather[aidviz]
pip install aidweather[aidfarm]
pip install aidweather[all]
```

## License

Apache-2.0. See [LICENSE](LICENSE).
