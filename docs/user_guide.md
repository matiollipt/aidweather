<p align="center">
  <img src="img/aidweather-logo-transparent.png" alt="aidweather logo" width="220">
</p>

# User Guide — `aidweather`

This guide provides a practical walkthrough for retrieving, cleaning, and analyzing NASA POWER agroclimatic data using `aidweather`.

---

## 1. Core Concepts

`aidweather` acts as a scientific bridge between your local environment and the official [NASA POWER API](https://power.larc.nasa.gov/). It simplifies data access while enforcing scientific integrity:

- **Model & Satellite Grid Data (Not Weather Stations)**: Returned values originate from global reanalysis models and satellite observations (such as MERRA-2 and CERES), representing spatial cell averages rather than direct measurements from ground-based weather stations.
- **Explicit Missing Values**: Missing observations or unrecorded dates are explicitly converted to pandas `NaN` (or `pd.NA`). `aidweather` never silently fills, interpolates, or forward-fills missing dates or values, ensuring your downstream statistics remain untainted.
- **SQLite Caching Layer**: API requests are automatically cached locally in SQLite to eliminate redundant downloads, save network bandwidth, and abide by NASA POWER server usage guidelines.

---

## 2. Temporal Resolutions

`PowerClient` supports two primary temporal resolution endpoints:
- `temporal_api="daily"`: Daily means, minimums, maximums, and integrated daily sums (available 1981–present for meteorology, 1984–present for solar radiation).
- `temporal_api="hourly"`: Hourly timestamped data series (available 2001–present).

```python
from aidweather import PowerClient

# Daily client for historical daily summaries
daily_client = PowerClient(temporal_api="daily")

# Hourly client for sub-daily time series
hourly_client = PowerClient(temporal_api="hourly")
```

---

## 3. Querying Point Data

To fetch weather data for a single location, use `get_point_data` or `get_point_data_from_coordinate`:

```python
from aidweather import PowerClient, GeoCoordinate

client = PowerClient(temporal_api="daily")
coord = GeoCoordinate.from_decimal(-23.55, -46.63)

df = client.get_point_data_from_coordinate(
    coord=coord,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"],
)
```

---

## 4. Multi-Point Queries

When retrieving weather data for multiple field locations or farm sites, use `get_multi_point_data` to run managed concurrent requests in parallel:

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

To sample weather parameters along a linear path between two geographic locations (e.g. across a river basin or agricultural gradient), use `get_transect_data`:

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
> `aidweather` automatically clamps sampling points along a transect to prevent sub-resolution duplication. The minimum effective sampling distance is determined by the native grid resolution of the requested parameters.

---

## 6. Regional Bounding-Box Queries

To extract grid cell time series across a 2D geographic region, use `get_regional_data`:

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

Point queries in `PowerClient` return DataFrames indexed by a `DatetimeIndex` named `"date"`. To convert the index into an explicit `datetime64[ns]` column (useful for merging, exporting to CSV, or joining with external datasets) or to standardize date columns across input files, use `ensure_date_column`:

```python
from aidweather.utils import ensure_date_column

# Convert DatetimeIndex or alternative columns ('timestamp', 'time') into a canonical 'date' column
df_flat = ensure_date_column(df, name="date", candidates=["timestamp", "time"])
```

See the [DataFrame Date Utilities Reference](utils_reference.md) for full details on candidate resolution, timezone handling, and `DateColumnOptions`.
