# Changelog

All notable changes to `aidweather` are documented here.

---

## [0.1.0] — 2026-05-30

Initial beta release.

### What's in this release

- `PowerClient` — NASA POWER API client with local SQLite cache, gzip compression, exponential-backoff retries, and parallel multi-point fetching.
- `GeoCoordinate` — type-safe coordinate value object with DMS/DDM/DD parsing and formatting.
- `normalize_coord_input` — accepts numbers, strings, or tuples in any coordinate format.
- `ensure_date_column` — robust DataFrame date-column normalization for downstream pipelines.
- `cfg` / `get_config` — singleton config with dot-notation access and XDG-compliant cache path resolution.
- CLI — `aidweather fetch`, `fetch-multi`, `fetch-transect`, `params list/describe`, `cache info/clear`.
- Bundled `config.json` with NASA POWER parameter catalogue, color map, and API limits.
- Apache-2.0 license.
