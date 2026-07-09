# PowerClient

`PowerClient` fetches meteorological and solar energy data from NASA's POWER API and returns a `pandas` DataFrame indexed by date. It handles caching, retries, and parallel requests. API service limits, attribution requirements, and usage policies are not overridden.

---

## NASA POWER API Limits

| API Resolution        | Max Parameters per Request | Spatial Constraint                                         | Temporal Constraint                |
| --------------------- | -------------------------: | ---------------------------------------------------------- | ---------------------------------- |
| **Point (Daily)**     |                         20 | Single latitude/longitude point                            | 1981-01-01 to Near Real Time (NRT) |
| **Point (Hourly)**    |                         15 | Single latitude/longitude point                            | 2001-01-01 to Near Real Time (NRT) |
| **Regional (Daily)**  |                          1 | Bounding box ≤ 4.5° × 4.5°; returned as 0.5° × 0.5° grid  | 1981-01-01 to Near Real Time (NRT) |
| **Regional (Hourly)** |              Not supported | Not supported                                              | Not supported                      |

`PowerClient` enforces these limits at the time of request submission. If you exceed a limit, the client will raise a `ValueError` before initiating the request.

> [!IMPORTANT]
> NASA POWER data are made available for broad reuse under Creative Commons Attribution 4.0 (CC BY 4.0). Commercial use is generally permitted under CC BY 4.0, provided that users comply with NASA POWER attribution requirements, applicable NASA data-use guidance, and API service limits. `aidweather` is an independent client for accessing NASA POWER data; it is not affiliated with, sponsored by, endorsed by, or approved by NASA. References to NASA POWER are provided solely for factual attribution and data-provenance purposes.

When publishing analyses, reports, dashboards, or derived products based on NASA POWER data, cite NASA POWER appropriately. NASA POWER requests that users include the following attribution:

> These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program.

See the [NASA POWER API documentation](https://power.larc.nasa.gov/docs/services/api/), [NASA POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/), and [NASA Earthdata Data Use and Citation Guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) for current terms, citation guidance, and responsible-use expectations.

---

## Basic usage

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

The result is a DataFrame indexed by date, with one column per parameter.

For a complete inventory of request models, public methods, and internal client helpers, see the [API Inventory](api_inventory.md#aidweatherclient).

---

## Temporal resolution

| Value      | What it returns                               |
| ---------- | --------------------------------------------- |
| `"daily"`  | Daily averages, min/max, totals (default)     |
| `"hourly"` | Hourly values — max 15 parameters per request |

```python
client = PowerClient(temporal_api="hourly")
```

For hourly requests, pass `start` and `end` dates for the NASA POWER request window. Returned rows are hourly when the API response contains hourly keys.

---

## Caching

By default, responses are cached in a local SQLite database at your platform's user cache directory:

| Platform | Default cache path                                        |
| -------- | --------------------------------------------------------- |
| Linux    | `~/.cache/aidweather/aidweather_cache.db`                 |
| macOS    | `~/Library/Caches/aidweather/aidweather_cache.db`         |
| Windows  | `%LOCALAPPDATA%\aidweather\Cache\aidweather_cache.db`     |

The cache is shared across all projects, so if you query São Paulo daily temperature twice from two different scripts, the second call is instant. Data is stored gzip-compressed.

To use a different location, set an environment variable before running:

```bash
export AIDWEATHER_CACHE_DIR=/your/shared/cache
```

Check your current cache state at any time:

```bash
aidweather cache info
```

---

## Fetching multiple points

```python
points = [
    {"lat": -23.55, "lon": -46.63, "name": "São Paulo"},
    {"lat": -22.90, "lon": -43.17, "name": "Rio de Janeiro"},
]

# max_workers defaults to 5 (safe recommended value)
df, failed = client.get_multi_point_data(
    points=points,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR"],
)
```

Returns a combined DataFrame with `lat` and `lon` columns added, plus a list of `(point, error_message)` pairs for any points that failed, so you can see *why* each one failed rather than just that it did. A `name` column is added too, but only for points that provided one — plain `(lat, lon)` tuples produce no `name` column.

> [!NOTE]
> The client defaults `max_workers` to `5` and automatically clamps higher values to protect the NASA POWER service. A client-side sliding-window rate limiter is also active by default at 30 requests per minute.

---

## Fetching a spatial transect

A transect fetches data for a series of evenly-spaced points along a straight-line
path between two geographic endpoints. Points are resolved in parallel using the
standard point API.

Define the transect with two `GeoCoordinate` endpoints and either a point count or
a target spacing:

```python
from aidweather import GeoCoordinate

# Define the start and end of the transect
coord_a = GeoCoordinate.from_decimal(-25.0, -48.0)  # southern end
coord_b = GeoCoordinate.from_decimal(-20.0, -48.0)  # northern end (~555 km)

# Using num_points
df = client.get_transect_data(
    start_coord=coord_a,
    end_coord=coord_b,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M", "PRECTOTCORR"],
    num_points=5,
)

# Using spacing_km instead — points derived from transect length / spacing
df = client.get_transect_data(
    start_coord=coord_a,
    end_coord=coord_b,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M"],
    spacing_km=111,  # one point per degree (~111 km)
)
```

A convenience wrapper mirrors the regional API pattern:

```python
df = client.get_transect_data_from_coordinates(
    coord_a=coord_a,
    coord_b=coord_b,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M"],
    num_points=5,
)
```

> [!IMPORTANT]
> The NASA POWER grid has a native **0.5° (~55 km) resolution**. Requesting a
> point spacing finer than this would return duplicate data for the same grid
> cell. `get_transect_data` enforces this minimum and automatically clamps
> `num_points` with an `INFO` log if the requested density exceeds it.

When both `num_points` and `spacing_km` are provided, `num_points` takes priority
and the effective spacing is logged as `INFO`.

---

## Fetching regional grid data

The regional endpoint returns data on a **0.5° × 0.5° grid** within a geographic bounding box. Each cell in the grid is a separate data point with its own lat/lon/elevation.

```python
df = client.get_regional_data(
    lat_min=-23.5,
    lat_max=-20.0,
    lon_min=-47.0,
    lon_max=-44.0,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M"],
)
print(df.head())
```

The result is a DataFrame indexed by date with `lat`, `lon`, `elevation`, and parameter columns — one row per grid cell per day.

You can also define the bounding box using two corner `GeoCoordinate` objects:

```python
from aidweather import GeoCoordinate

sw = GeoCoordinate.from_decimal(-23.5, -47.0)
ne = GeoCoordinate.from_decimal(-20.0, -44.0)

df = client.get_regional_data_from_coordinates(
    coord_sw=sw,
    coord_ne=ne,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M"],
)
```

> [!WARNING]
> The bounding box must not exceed **4.5° × 4.5°** on either axis, and only **one parameter** can be requested per regional call. The client validates both constraints before sending the request.

---

## Getting a summary

After fetching, call `summarize()` to print Rich-formatted tables covering data coverage, missing values, cache performance, and API connection state:

```python
df = client.get_point_data(...)
client.summarize(df)
```

---

## Error behavior

- **Network failures:** retried automatically for HTTP 429, 500, 502, 503, and 504 with exponential backoff.
- **Cache failure:** logs a warning and falls back to live API requests.
- **Stale cache on network error:** if the API is unreachable and cached data exists, the cached data is returned with a warning.
- **Empty response:** returns a DataFrame filled with `NaN` for the requested date range and columns.
- **Unrecognized parameter codes:** issues a `UserWarning` rather than failing — NASA POWER may still accept codes not in the bundled catalogue.
- **`start` after `end`:** raises a `ValueError` before any request is sent.
- **`wind_elevation` out of range:** if provided, it must be between 10 and 300 meters, or a `ValueError` is raised.
