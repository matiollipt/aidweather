<p align="center">
  <img src="docs/img/aidweather-logo-transparent.png" alt="aidweather logo" width="380">
</p>

<h1 align="center">aidweather</h1>

<p align="center">
  <strong>Agroclimatic & Solar Data Integration from NASA POWER</strong>
</p>

`aidweather` bridges your project with reliable weather and solar radiation data from [NASA's POWER API](https://power.larc.nasa.gov/). Key features include:

1. **Seamless NASA POWER Integration**: Serves as a bridge between your application and NASA POWER weather services, hiding the complexities of constructing API requests, handling timeouts, and verifying payload structures.
2. **Versatile Spatial Fetching**: Flexible data retrieval across single points, 1D transect paths, and 2D regional bounding boxes, with internal tuning levers to ensure accurate and consistent spatial sampling.
3. **SQLite Caching & API Compliance**: Built-in SQLite caching prevents redundant downloads, saving bandwidth and avoiding unnecessary hits to NASA POWER servers while enforcing rate limits and data usage policies.
4. **Python Bindings & Full CLI**: Dual interface offering an intuitive Python API for data science workflows and a full-featured Command Line Interface (CLI) for quick command-line extractions.
5. **Smart Error Catching & Logging**: Comprehensive activity logging and intelligent error handling, providing clear insights into query execution and data processing.

> [!WARNING]
> **Beta Status:** `aidweather` is in active beta. Public API methods and CLI options are stable, but additions may occur prior to 1.0. Feedback and [issue reports](https://github.com/matiollipt/aidweather/issues) are welcome.

> [!IMPORTANT]
> **Data Provenance & Spatial Support:** A coordinate submitted to NASA POWER resolves to a source product grid cell; it does not produce a point measurement at the precise coordinate. Different parameters originate from different source products (e.g. MERRA-2 at 0.5° × 0.625°, CERES solar radiation at 1.0° × 1.0°). Nearby coordinates mapping to the same source cell will return identical series.

---

## Supported NASA POWER Parameters

`aidweather` includes built-in metadata, unit validation, and spatial grid specifications for 18 core weather, solar radiation, and soil moisture parameters:

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
| `GWETTOP` | Surface Soil Wetness | MERRA-2 / GEOS-IT | 0.50° × 0.625° | 0–1 | — |
| `GWETROOT` | Root Zone Soil Wetness | MERRA-2 / GEOS-IT | 0.50° × 0.625° | 0–1 | — |
| `GWETPROF` | Profile Soil Moisture | MERRA-2 / GEOS-IT | 0.50° × 0.625° | 0–1 | — |

> `—` indicates the parameter is not published on NASA POWER's hourly endpoint; requesting it with `temporal_api="hourly"` raises `ValueError`.

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

The documentation is organized into dedicated guides for scientific users and developers:

- [User Guide](docs/user_guide.md) — Step-by-step introduction to fetching and analyzing weather data.
- [Spatial Interpretation Guide](docs/spatial_interpretation.md) — Understanding native grid resolutions, transects, regional boxes, and station comparisons.
- [Parameter & Provenance Guide](docs/parameter_provenance.md) — Parameter catalogue, NASA source models, temporal availability, and time standards.
- [Geospatial Coordinate Reference](docs/geo_reference.md) — Coordinate notation systems (DD, DDM, DMS), string parsing, and `GeoCoordinate` usage.
- [DataFrame Date Utilities Reference](docs/utils_reference.md) — Date column matching, DatetimeIndex fallbacks, and timezone handling.
- [API Reference](docs/api_reference.md) — Complete Python API reference for `PowerClient`, `GeoCoordinate`, and query functions.
- [Configuration Reference](docs/config_reference.md) — Environment variables, JSON configuration assets, and SQLite cache settings.
- [Developer Guide](docs/developer_guide.md) — Internal request lifecycles, cache key design, and architecture details.
- [Contributing & Testing](docs/contributing.md) — Guidelines for contributing, running `pytest`, and checking typing with `mypy`.

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

# 2. Sample along a 1D spatial transect (respecting native parameter grid spacing)
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
# Fetch single-point data to CSV
aidweather fetch --lat -23.55 --lon -46.63 \
    --start 2023-01-01 --end 2023-01-31 \
    --params T2M,PRECTOTCORR --output point_data.csv

# Sample along a 1D spatial transect
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

NASA POWER data are open access and free of charge. In accordance with the [NASA Earthdata Data Use and Attribution Policy](https://www.earthdata.nasa.gov/learn/use-nasa-data), scientific publications, reports, and applications utilizing `aidweather` must properly cite the NASA POWER Project and acknowledge the primary satellite/reanalysis source products.

**Suggested Acknowledgment Text:**
> *"Meteorological and solar energy parameters were retrieved from the NASA Prediction Of Worldwide Energy Resources (POWER) project using aidweather v0.1.3. NASA POWER meteorological products originate from the GMAO MERRA-2 / GEOS-IT assimilation models, and solar radiation products originate from NASA LaRC CERES / FLASHFlux / SRB satellite observations."*

---

## License

Apache-2.0. See [LICENSE](LICENSE).
