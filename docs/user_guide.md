# User Guide — `aidweather`

This guide explains how to retrieve, clean, and work with NASA POWER agroclimatic data using `aidweather`.

---

## 1. Core Concepts

`aidweather` acts as a scientific bridge to the official [NASA POWER API](https://power.larc.nasa.gov/).

Key properties:
- **No Station Interpolation**: Returned numbers reflect satellite and reanalysis model grid cells (e.g. MERRA-2, CERES), not direct weather station observations.
- **Explicit Missing Values**: Missing values or undocumented records are converted directly to pandas `NaN` (or `pd.NA`). `aidweather` never silently fills, interpolates, or forward-fills missing observation dates or parameters.
- **SQLite Cache**: API queries are cached locally in SQLite to prevent duplicate network traffic and respect NASA POWER API guidelines.

---

## 2. Temporal Resolutions

`PowerClient` supports two primary temporal endpoints:
- `temporal_api="daily"`: Daily mean, minimum, maximum, and integrated values (1981-present for meteorology, 1984-present for solar).
- `temporal_api="hourly"`: Hourly timestamped data (2001-present).

```python
from aidweather import PowerClient

# Daily client
daily_client = PowerClient(temporal_api="daily")

# Hourly client
hourly_client = PowerClient(temporal_api="hourly")
```

---

## 3. Querying Point Data

Use `get_point_data` or `get_point_data_from_coordinate`:

```python
from aidweather import PowerClient, GeoCoordinate

client = PowerClient(temporal_api="daily")
coord = GeoCoordinate.from_decimal(-23.55, -46.63)

df = client.get_point_data_from_coordinate(
    coord=coord,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"],
```

---

## 4. Multi-Point Queries

When retrieving data for multiple field locations, use `get_multi_point_data` to execute parallel async-managed requests:

```python
points = [
    {"lat": -23.55, "lon": -46.63, "name": "Site_A"},
    {"lat": -22.90, "lon": -43.20, "name": "Site_B"},
]

df_multi, failed = client.get_multi_point_data(
    points=points,
    start="2023-01-01",
    end="2023-01-10",
    params=["T2M"],
    max_workers=5,
)
```

---

## 5. 1D Transect Queries

To sample along a transect between two geographic coordinates:

```python
coord_start = GeoCoordinate.from_decimal(-25.0, -48.0)
coord_end = GeoCoordinate.from_decimal(-20.0, -48.0)

df_transect = client.get_transect_data(
    start_coord=coord_start,
    end_coord=coord_end,
    start="2023-01-01",
    end="2023-01-07",
    params=["T2M", "ALLSKY_SFC_SW_DWN"],
    spacing_km=50.0,
)
```

> [!NOTE]
> `aidweather` automatically clamps transect point sampling to prevent sub-resolution duplication. The effective minimum spacing is derived from the finest native resolution of the requested parameters.

---

## 6. Regional Bounding-Box Queries

The regional endpoint returns grid cell series over a 2D bounding box:

```python
df_regional = client.get_regional_data(
    lat_min=-23.5, lat_max=-20.0,
    lon_min=-47.0, lon_max=-44.0,
    start="2023-01-01", end="2023-01-07",
    params=["T2M"],
)
```

> [!WARNING]
> Regional API Constraints:
> - Maximum bounding box size: 4.5° × 4.5°.
> - Exactly 1 parameter can be requested per regional API call.

---

## 7. Working with Output DataFrames & Date Utilities

Point queries in `PowerClient` return DataFrames with a `DatetimeIndex` named `"date"`. To convert the index to an explicit `datetime64[ns]` column (e.g. for joins or CSV exports) or standardise heterogeneous date columns across input files, use `ensure_date_column`:

```python
from aidweather.utils import ensure_date_column

# Convert DatetimeIndex or alternative column ('timestamp', 'dt') into a 'date' column
df_flat = ensure_date_column(df, name="date", candidates=["timestamp", "time"])
```

See the [DataFrame Date Utilities Reference](utils_reference.md) for full details on candidate searching, timezone stripping, and `DateColumnOptions`.

