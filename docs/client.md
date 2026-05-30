# client

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## Purpose

Provides a robust client (`PowerClient`) for accessing the NASA POWER API. It handles data fetching, parsing, caching (via SQLite), and automatic retries.

## Key responsibilities

- Fetching daily or hourly weather/solar data.
- Caching API responses to `aidweather_cache.db` to minimize network calls.
- Handling retries with exponential backoff.
- merging and deduplicating data from multiple fetch operations.
- Parallel fetching for multiple points.

## Public API

### Classes

- `PowerClient`:
  - `__init__(temporal_api="daily", session=None)`: Initializes client and cache.
  - `get_point_data(lat, lon, start, end, params, elevation=None) -> pd.DataFrame`: Fetches data for a single point.
  - `get_point_data_from_coordinate(coord, start, end, params, elevation=None) -> pd.DataFrame`: Same as above but using `GeoCoordinate`.
  - `get_multi_point_data(points, start, end, params, max_workers=8) -> Tuple[pd.DataFrame, List]`: Parallel fetch for list of points. Returns (combined_df, failed_points).
  - `get_regional_data(lat_lon_list, start, end, params) -> pd.DataFrame`: Fetches data from regional endpoint.
  - `get_expanded_point_data(lat, lon, start, end, params, axis="lat", distance_km=10.0, num_points=10, ...) -> pd.DataFrame`: Generates a transect of points and fetches data for them.

## Data flow and dependencies

- **Internal imports**: `cfg` (configuration), `GeoCoordinate` (from `.geo`).
- **External dependencies**: `requests`, `pandas`, `numpy`, `sqlite3`, `gzip`, `hashlib`, `concurrent.futures`.
- **Cache**: Stores data in `<cache_path>/aidweather_cache.db`.

## Configuration and assets

- **Config**:
  - `cfg.get_url()`: Resolves API endpoints.
  - `cfg.cache_config()`: Determines if caching is enabled (`enabled`) and where to store the DB (`path`).

## Error handling and edge cases

- **Network Errors**: Uses a custom `requests.Session` with `Retry` (backoff factor 0.5) for transient errors (429, 50x).
- **Cache Failures**: If SQLite fails (init, read, or write), logs warnings and falls back to direct API usage without crashing.
- **Empty Responses**: Returns empty DataFrames or DataFrames filled with NaNs if API returns no data for a range.
- **Partial Cache Hits**: Identifies missing date ranges and only fetches those from the API, merging with cached data.

## Minimal usage example

```python
from aidweather import PowerClient

client = PowerClient(temporal_api="daily")
df = client.get_point_data(
    lat=34.05, lon=-118.25,
    start="20230101", end="20230110",
    params=["T2M"]
)
print(df.head())
```
