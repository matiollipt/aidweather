# Configuration Reference — `aidweather`

This document describes configuration loading, environment variables, SQLite caching, and parameter metadata schemas.

---

## 1. Environment Variables

`aidweather` respects the following environment variables:

| Variable | Description | Default Path |
| :--- | :--- | :--- |
| `AIDWEATHER_CACHE_DIR` | Custom directory path for local SQLite cache database. | Platform XDG cache (`~/.cache/aidweather`) |
| `AIDWEATHER_LOG_DIR` | Custom directory path for `aidweather.log` file outputs. | Platform XDG log dir (`~/.cache/aidweather`) |
| `AIDWEATHER_RUN_LIVE_TESTS` | Set to `1` to allow live NASA POWER API test suite execution. | Unset (`0`) |

---

## 2. SQLite Cache Database Schema

Location: `<AIDWEATHER_CACHE_DIR>/aidweather_cache.db`

### Table: `cache`
- `key` (`TEXT PRIMARY KEY`): SHA-256 hash prefixed with `v1_` generated from payload parameters (`temporal_api`, `parameters`, `latitude`, `longitude`, `elevation`, `wind-elevation`, `wind-surface`, `community`, `time-standard`).
- `data` (`BLOB`): Gzip-compressed JSON string of fetched time-series records.
- `timestamp` (`DATETIME`): Access timestamp of the cached record.

```bash
# Check cache metrics
aidweather cache info

# Clear cache database
aidweather cache clear --yes
```

---

## 3. Bundled `config.json` Asset Schema

`aidweather` bundles `src/aidweather/assets/config.json` containing base API URLs, parameter groups, and parameter metadata dictionaries:

```json
{
  "base_urls": {
    "daily": {
      "point": "https://power.larc.nasa.gov/api/temporal/daily/point",
      "regional": "https://power.larc.nasa.gov/api/temporal/daily/regional"
    },
    "hourly": {
      "point": "https://power.larc.nasa.gov/api/temporal/hourly/point"
    }
  },
  "param_metadata": {
    "T2M": {
      "short_name": "Temperature at 2 Meters",
      "units": {"daily": "°C", "hourly": "°C"},
      "source_family": "MERRA-2/GEOS-IT",
      "native_grid": {"latitude_degrees": 0.5, "longitude_degrees": 0.625},
      "availability": {"daily_start": "1981-01-01", "hourly_start": "2001-01-01"},
      "time_standards": ["LST", "UTC"],
      "provisional_tail": true,
      "attribution": "NASA POWER / GMAO MERRA-2"
    }
  }
}
```
