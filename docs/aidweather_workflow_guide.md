# How aidweather works

`aidweather` turns geographic coordinates and a date range into a clean, validated weather dataset. Here's what happens under the hood and why the package is built the way it is.

---

## The core flow

```
Coordinates + Date Range
        ↓
  Coordinate normalization (geo)
        ↓
  Config resolution (config)
        ↓
  Cache lookup → partial or full hit?
     ↓ miss               ↓ hit
  NASA POWER API      return cached data
     ↓
  Retry / backoff
     ↓
  Parse JSON → DataFrame
     ↓
  Write to cache (merged with existing)
        ↓
  Validated time series DataFrame
```

---

## Why these modules exist

### `geo` — coordinate normalization

Coordinates arrive from field records, GPS exports, and user input in DMS, DDM, and decimal degree formats — sometimes with Unicode degree symbols or inconsistent hemisphere notation. The `geo` module absorbs all of that variation before anything hits the API.

### `config` — centralized settings

API endpoint URLs, parameter metadata, cache paths, and logging settings live in one place (`config.json`) and are accessible through the `cfg` singleton. Downstream packages in the `aid*` ecosystem share this config file, so there is no duplication of settings.

### `client` — data acquisition

`PowerClient` handles the full request lifecycle: building payloads, checking the cache, fetching missing date ranges only, merging new data with cached data, and writing the result back to the cache. Retry logic and gzip compression are built in.

### `utils` — downstream compatibility

`ensure_date_column` standardizes the time axis of any DataFrame so it's ready to merge with `PowerClient` output or pass to feature engineering pipelines.

---

## Practical outcome

In practice, you call `aidweather` once per location and date range, and every later call for the same request is served from cache instead of hitting the API again. The output is always a `pandas` DataFrame with a `DatetimeIndex` and numeric columns — ready for analysis, joining with field data, or passing to a model.

---

## Related documentation

- [Client documentation](client.md) — complete `PowerClient` API reference.
- [Data Source Comparison](data_source_comparison.md) — when to use NASA POWER vs. Meteostat and other alternatives.
- [NASA POWER License & Usage](aidweather_nasa_power_usage.md) — API limits, rate limiting, licensing, attribution, and responsible use.
