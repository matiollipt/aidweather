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

The client loads `.env` automatically from the current directory, so no extra setup is needed.

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

df, failed = client.get_multi_point_data(
    points=points,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR"],
    max_workers=5,
)
```

Returns a combined DataFrame with `lat`, `lon`, and `name` columns added, plus a list of any points that failed.

> Keep `max_workers` at or below 5. NASA explicitly discourages high concurrency from a single IP.

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

## Getting a summary

After fetching, call `summarize()` to print Rich-formatted tables covering data coverage, missing values, cache performance, and API connection state:

```python
df = client.get_point_data(...)
client.summarize(df)
```

---

## Parameters cap

| Resolution | Max parameters per request |
|---|---|
| Daily | 20 |
| Hourly | 15 |
| Regional (daily) | 1 |

The client raises a `ValueError` before the request if you exceed these limits.

---

## Error behavior

- **Network failures**: retried automatically (HTTP 429, 500, 502, 503, 504) with exponential backoff.
- **Cache failure**: logs a warning and falls back to live API — never crashes.
- **Stale cache on network error**: if the API is unreachable and cached data exists, the cached data is returned with a warning.
- **Empty response**: returns a DataFrame filled with `NaN` for the requested date range and columns.
