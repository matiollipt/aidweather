# PowerClient

`PowerClient` is the main way to get weather data out of NASA POWER. Give it a location, a date range, and a list of parameters, and it returns a clean pandas DataFrame.

It handles caching, retries, and parallel fetching for you. You don't need an API key for basic use, but registering one is recommended for regular work. API keys improve reliability and attribution; they do not remove NASA POWER service limits or usage policies.

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

For a complete inventory of request models, public methods, and internal client
helpers, see the [API Inventory](api_inventory.md#aidweatherclient).

---

## Temporal resolution

| Value | What it returns |
|---|---|
| `"daily"` | Daily averages, min/max, totals (default) |
| `"hourly"` | Hourly values — max 15 parameters per request |

```python
client = PowerClient(temporal_api="hourly")
```

For hourly requests, pass `start` and `end` dates for the NASA POWER request window. Returned rows are hourly when the API response contains hourly keys.

---

## API key

Without a key, requests run against IP-based limits that may be shared with other users on the same network. For repeated workflows, register a free key at [NASA POWER](https://power.larc.nasa.gov/) and set it in your environment:

```bash
# .env file or shell
NASA_POWER_API_KEY=your_key_here
```

The client loads `.env` automatically from the current directory to retrieve the `NASA_POWER_API_KEY` only; other environment configurations (like cache path overrides) must be set in your shell environment.

---

## Caching

By default, responses are cached in a local SQLite database at your platform's user cache directory:

- Linux: `~/.cache/aidweather/aidweather_cache.db`
- macOS: `~/Library/Caches/aidweather/aidweather_cache.db`

The cache is shared across all projects, so if you query São Paulo daily temperature twice from two different scripts, the second call is instant. Data is stored gzip-compressed.

To use a different location, set an environment variable before running:

```bash
export AIDWEATHER_CACHE_DIR=/your/shared/cache
```

Check the current cache state any time with:

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

Returns a combined DataFrame with `lat`, `lon`, and `name` columns added, plus a list of any points that failed.

> **Concurrency & Rate Limiting**: The client defaults `max_workers` to `5` and automatically clamps values above the configured limit to protect the NASA POWER server. Additionally, a surgical client-side sliding window rate limiter is active (defaulting to 30 requests per minute) to ensure full compliance with NASA POWER API guidelines.

---

## Fetching a spatial transect

Generates a line of evenly spaced points expanding from a center coordinate and fetches them in parallel.

```python
df = client.get_expanded_point_data(
    lat=-23.55,
    lon=-46.63,
    start="2023-01-01",
    end="2023-01-31",
    params=["T2M"],
    axis="lat",       # expand along latitude
    distance_km=100,  # total transect length
    num_points=10,
)
```

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
    coord_sw=sw, coord_ne=ne,
    start="2023-01-01", end="2023-01-31",
    params=["T2M"],
)
```

> **Bounding box constraints**: The bounding box must not exceed **4.5° × 4.5°** on either axis (NASA POWER API limit), and only **one parameter** can be requested per call. The client validates both constraints before sending the request.

---

## Getting a summary

After fetching, call `summarize()` to print Rich-formatted tables covering data coverage, missing values, cache performance, and API connection state:

```python
df = client.get_point_data(...)
client.summarize(df)
```

---

## Parameters cap

| Resolution | Max parameters per request | Spatial constraint |
|---|---|---|
| Daily (point) | 20 | — |
| Hourly (point) | 15 | — |
| Regional (daily) | 1 | Bounding box ≤ 4.5° × 4.5° |

The client raises a `ValueError` before the request if you exceed these limits.

---

## Error behavior

- **Network failures**: retried automatically (HTTP 429, 500, 502, 503, 504) with exponential backoff.
- **Cache failure**: logs a warning and falls back to live API — never crashes.
- **Stale cache on network error**: if the API is unreachable and cached data exists, the cached data is returned with a warning.
- **Empty response**: returns a DataFrame filled with `NaN` for the requested date range and columns.
