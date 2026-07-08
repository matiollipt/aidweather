# Changelog

All changes to `aidweather` are documented here.

---

## [0.1.3] — 2026-07-08

### Bug Fixes

- **Ambiguous slash-separated dates are now rejected instead of silently misread.**
  `--start`/`--end` (CLI) and `start`/`end` (Python API) previously went through
  `pandas.to_datetime` with its default month-first convention, so a date like
  `05/03/2023` was silently read as May 3rd rather than March 5th — a real risk for
  Brazil-first users typing day-first dates. These now raise a clear error asking for
  an unambiguous format (e.g. `YYYY-MM-DD`).
- **Multi-point and transect fetch failures no longer disappear behind a generic "no
  data" message.** `get_multi_point_data`'s `failed_points` return value now carries
  `(point, error_message)` pairs instead of bare points, and the CLI's `fetch-multi`
  command (which previously discarded this value entirely) now prints the actual
  reason for each failed point. `get_transect_data` likewise includes a sample of the
  underlying errors when every point in a transect fails.

## [0.1.2] — 2026-07-01

### Release Engineering

- **First version actually tagged and published.** Versions 0.1.0 and 0.1.1 below were
  never tagged on GitHub or published to PyPI/TestPyPI — `0.1.2` is the first release
  where the version recorded in `pyproject.toml`, `src/aidweather/__init__.py`, and this
  changelog are verified to agree before publishing.
- Added `types-requests` to the `test` extra so `mypy` runs clean.
- Stopped tracking generated example run artifacts under `outputs/`.
- Added CI workflows for publishing to TestPyPI (`publish-testpypi.yml`, manual trigger)
  and PyPI (`publish.yml`, triggered by GitHub Releases) via trusted publishing (OIDC).
- Removed unreleased dependency `aidviz` from `pyproject.toml` and consolidated example-specific dependencies into a local dev helper script (not distributed with the package; `examples/` is untracked).

## [0.1.1] — 2026-06-25

### Features & Improvements

- **Compatibility & Clean APIs**: Production maintenance release ensuring compatibility and metadata stability.

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
- Bundled `config.json` with NASA POWER parameter catalogue and API limits.
- Apache-2.0 license.
