# Changelog

All changes to `aidweather` are documented here.

---

## [0.1.0] — 2026-06-10

### Breaking Changes

- **Regional API refactored for NASA POWER compliance.** `get_regional_data()` now accepts a bounding box (`lat_min`, `lat_max`, `lon_min`, `lon_max`) instead of a list of lat/lon tuples. `get_regional_data_from_coordinates()` now accepts two corner `GeoCoordinate` objects (`coord_sw`, `coord_ne`) instead of a list. The old `lonlat` payload parameter was non-compliant with the NASA POWER API and has been removed.

### What's in this release

I'm thrilled to present the initial **beta release** of `aidweather`, a Python package for fetching weather data from the NASA POWER API. The initial beta release includes:

- `PowerClient` — NASA POWER API client with local SQLite cache, gzip compression, exponential-backoff retries, and parallel multi-point fetching.
- `GeoCoordinate` — type-safe coordinate value object with DMS/DDM/DD parsing and formatting.
- `normalize_coord_input` — accepts numbers, strings, or tuples in any coordinate format.
- `ensure_date_column` — robust DataFrame date-column normalization for downstream pipelines.
- `cfg` / `get_config` — singleton config with dot-notation access and XDG-compliant cache path resolution.
- CLI — `aidweather fetch`, `fetch-multi`, `fetch-transect`, `params list/describe`, `cache info/clear`.
- Bundled `config.json` with NASA POWER parameter catalogue, color map, and API limits.
- Apache-2.0 license.
