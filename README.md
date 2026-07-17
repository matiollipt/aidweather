# aidweather

**Agroclimatic & Solar Data Integration from NASA POWER**

`aidweather` is a Python package for fetching, caching, and analyzing meteorological and solar radiation data from [NASA's POWER API](https://power.larc.nasa.gov/). It provides structured parameter metadata, parameter-aware spatial sampling along 1D transects, bounding-box queries, SQLite request caching, and geospatial coordinate parsing.

> [!WARNING]
> **Beta Status:** `aidweather` is in active beta. Public API methods and CLI options are stable, but additions may occur prior to 1.0. Feedback and [issue reports](https://github.com/matiollipt/aidweather/issues) are welcome.

> [!IMPORTANT]
> **Data Provenance & Spatial Support:** A coordinate submitted to NASA POWER resolves to a source product grid cell; it does not produce a point measurement at the precise coordinate. Different parameters originate from different source products (e.g. MERRA-2 at 0.5° × 0.625°, CERES solar radiation at 1.0° × 1.0°). Nearby coordinates mapping to the same source cell will return identical series.

---

## Supported NASA POWER Parameters

`aidweather` provides built-in metadata, unit validation, and spatial grid specifications for 15 core parameters across meteorological and radiative families:

| Code | Parameter Name | Source Product | Native Grid (Lat × Lon) | Daily Unit | Hourly Unit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T2M` | Temperature at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `T2M_MAX` | Maximum Temperature at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `T2M_MIN` | Minimum Temperature at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `T2M_RANGE` | Diurnal Temperature Range at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `T2MWET` | Wet Bulb Temperature at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `T2MDEW` | Dew Point Temperature at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `TS` | Surface Skin Temperature | MERRA-2 / GEOS-IT | 0.50° × 0.625° | °C | °C |
| `RH2M` | Relative Humidity at 2 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | % | % |
| `PRECTOTCORR` | Corrected Total Precipitation | MERRA-2 / GEOS-IT | 0.50° × 0.625° | mm/day | mm/hr |
| `ALLSKY_SFC_SW_DWN` | Surface Shortwave Downward Irradiance | CERES SYN1deg / SRB | 1.00° × 1.00° | kWh/m²/day | W/m² |
| `ALLSKY_SFC_PAR_TOT` | Surface PAR Total | CERES SYN1deg / SRB | 1.00° × 1.00° | MJ/m²/day | W/m² |
| `WS10M` | Wind Speed at 10 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | m/s | m/s |
| `WS10M_MAX` | Maximum Wind Speed at 10 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | m/s | m/s |
| `WD10M` | Wind Direction at 10 m | MERRA-2 / GEOS-IT | 0.50° × 0.625° | degrees | degrees |
| `PS` | Surface Pressure | MERRA-2 / GEOS-IT | 0.50° × 0.625° | kPa | kPa |

---

## Installation

### Unix / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash
```

Or install via `uv tool`:

```bash
uv tool install git+https://github.com/matiollipt/aidweather.git
```

### Windows (PowerShell)

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/matiollipt/aidweather/main/install.ps1))) -Yes
```

---

## Documentation Guides

The documentation is organized into dedicated scientific and developer reference guides:

- [User Guide](docs/user_guide.md) — Comprehensive guide for scientific users and data analysts.
- [Spatial Interpretation Guide](docs/spatial_interpretation.md) — Native resolution grids, transects, regional boxes, and station comparisons.
- [Parameter & Provenance Guide](docs/parameter_provenance.md) — Complete catalogue, source product lineage, temporal ranges, and time standards.
- [API Reference](docs/api_reference.md) — Complete python reference for `PowerClient`, `GeoCoordinate`, and query functions.
- [Configuration Reference](docs/config_reference.md) — Environment variables, JSON assets, and SQLite cache layout.
- [Developer Guide](docs/developer_guide.md) — Cache key versioning, request lifecycles, and architecture notes.
- [Contributing & Testing](docs/contributing.md) — Quality standards, `pytest` invocation, and mypy typing.

---

## Quick Start (Python)

```python
from aidweather import PowerClient, GeoCoordinate

# Initialize client for daily temporal resolution
client = PowerClient(temporal_api="daily")

# 1. Fetch single-point meteorological and solar data
df = client.get_point_data(
    lat=-23.55,
    lon=-46.63,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"],
)
print(df.head())

# 2. Sample 1D spatial transect (clamping respects native parameter grid spacing)
coord_a = GeoCoordinate.from_decimal(-25.0, -48.0)
coord_b = GeoCoordinate.from_decimal(-20.0, -48.0)

df_transect = client.get_transect_data(
    start_coord=coord_a,
    end_coord=coord_b,
    start="2023-01-01",
    end="2023-01-07",
    params=["T2M"],
    spacing_km=100.0,
)
print(df_transect.head())
```

---

## Command Line Interface (CLI)

```bash
# Fetch single point data to CSV
aidweather fetch --lat -23.55 --lon -46.63 \
    --start 2023-01-01 --end 2023-01-31 \
    --params T2M,PRECTOTCORR --output point_data.csv

# Sample along a spatial transect
aidweather fetch-transect \
    --lat-start -25.0 --lon-start -48.0 \
    --lat-end   -20.0 --lon-end   -48.0 \
    --start 2023-01-01 --end 2023-01-31 \
    --spacing-km 100.0

# Fetch regional bounding box data (max 4.5° x 4.5°, 1 parameter per request)
aidweather fetch-regional --lat-min -23.5 --lat-max -20.0 \
    --lon-min -47.0 --lon-max -44.0 \
    --start 2023-01-01 --end 2023-01-31 \
    --params T2M --output regional.csv

# Inspect parameter metadata & local cache
aidweather params list --group all
aidweather params describe T2M
aidweather cache info
```

---

## Data Licensing & Official NASA Attribution

NASA POWER data are open access and free of charge. In accordance with [NASA Earthdata Data Use and Attribution Policy](https://www.earthdata.nasa.gov/learn/use-nasa-data), scientific publications, reports, and applications utilizing `aidweather` must properly cite the NASA POWER Project and acknowledge the primary satellite/reanalysis source products.

**Suggested Acknowledgment Text:**
> *"Meteorological and solar energy parameters were retrieved from the NASA Prediction Of Worldwide Energy Resources (POWER) project using aidweather v0.1.3. NASA POWER meteorological products originate from the GMAO MERRA-2 / GEOS-IT assimilation models, and solar radiation products originate from NASA LaRC CERES / FLASHFlux / SRB satellite observations."*

---

## License

Apache-2.0. See [LICENSE](LICENSE).

