# API Inventory

This page is the complete inventory of classes and functions defined in
`src/aidweather`, for anyone who wants a quick overview without reading the source.

Names starting with `_` are implementation details. They are documented here for
transparency, debugging, and automated code navigation, but they are not stable
public API.

## Stable Top-Level API

Import these from `aidweather` for normal use:

| Name | Type | Purpose |
|---|---|---|
| `PowerClient` | class | NASA POWER API client for point, multi-point, transect, and regional weather data. |
| `GeoCoordinate` | class | Immutable latitude/longitude value object with validation and format conversion. |
| `normalize_coord_input` | function | Convert common coordinate inputs into a `GeoCoordinate`. |
| `cfg` | object | Singleton package configuration. |
| `get_config` | function | Return the singleton configuration object. |
| `ensure_date_column` | function | Standardize a DataFrame date column for joins and downstream analysis. |

Package metadata:

| Name | Purpose |
|---|---|
| `__version__` | Package version string. |
| `__author__` | Package author metadata. |
| `__url__` | Project URL metadata. |

## `aidweather.client`

The client module is the data-access layer. It builds NASA POWER requests,
parses responses into pandas DataFrames, manages a local SQLite cache, and
provides safe helpers for multi-point workflows.

### Request Models

| Class | Purpose | Key Fields |
|---|---|---|
| `PowerQuery` | Base Pydantic request model for date range and parameters. | `start`, `end`, `params` |
| `PointRequest` | Request model for one point. | `lat`, `lon`, `elevation`, `wind_elevation`, `wind_surface` |
| `TransectRequest` | Request model for a 1D transect of point API calls between two endpoints. | `start_coord`, `end_coord`, `num_points`, `spacing_km`, `max_workers` |
| `RegionalRequest` | Request model for a regional bounding-box query. | `lat_min`, `lat_max`, `lon_min`, `lon_max` |

### `PowerClient`

`PowerClient(temporal_api="daily", session=None)` creates a client
for the NASA POWER daily or hourly endpoint.

Initialization:

| Method | Purpose |
|---|---|
| `__init__(temporal_api="daily", session=None)` | Validate temporal resolution, configure URLs/session/cache. |

Primary methods:

| Method | Purpose | Returns |
|---|---|---|
| `get_point_data(request=None, **kwargs)` | Fetch data for one latitude/longitude pair. Accepts either a `PointRequest` or keyword arguments. | `pd.DataFrame` indexed by date/time |
| `get_point_data_from_coordinate(coord, start, end, params, elevation=None, wind_elevation=None, wind_surface=None)` | Same as `get_point_data`, but starts from a `GeoCoordinate`. | `pd.DataFrame` |
| `get_multi_point_data(points, start, end, params, max_workers=5)` | Fetch several points in parallel. Points may be dicts, tuples, or a DataFrame with `lat`/`lon`. A `name` column is added to the result only for points that provided one (e.g. dicts with a `"name"` key) — plain `(lat, lon)` tuples get no `name` column. | `(combined_df, failed_points)` |
| `get_transect_data(request=None, **kwargs)` | Fetch data along a 1D transect between two `GeoCoordinate` endpoints. Sampling controlled by `num_points` or `spacing_km`; minimum spacing 0.5° (~55 km) enforced. | Combined `pd.DataFrame` |
| `get_transect_data_from_coordinates(coord_a, coord_b, start, end, params, num_points=None, spacing_km=None, max_workers=5)` | Transect helper accepting two `GeoCoordinate` objects as start and end endpoints. | `pd.DataFrame` |
| `get_regional_data(lat_min, lat_max, lon_min, lon_max, start, end, params, request=None)` | Fetch regional data on a 0.5° grid within a bounding box (≤ 4.5° × 4.5°). Daily only, one parameter. | `pd.DataFrame` with `lat`, `lon`, `elevation` columns |
| `get_regional_data_from_coordinates(coord_sw, coord_ne, start, end, params)` | Regional helper accepting two corner `GeoCoordinate` objects (SW and NE). | `pd.DataFrame` |
| `summarize(df)` | Print Rich tables describing coverage, cache/network metrics, and API connection state. | `None` |

Lifecycle and representation:

| Method | Purpose |
|---|---|
| `__repr__()` | Return a compact client representation showing temporal API and base URL. |
| `__del__()` | Close the SQLite connection when the client is destroyed. |

Important behavior:

- `temporal_api` must be `"daily"` or `"hourly"`.
- Daily point requests allow up to 20 parameters; hourly point requests allow up
  to 15; daily regional requests allow one parameter.
- Regional bounding boxes must not exceed 4.5° on either axis.
- Unrecognized parameter codes raise a `UserWarning` rather than failing outright
  (NASA POWER may still accept codes not in the bundled catalogue).
- `start` must not be after `end`, or a `ValueError` is raised.
- `wind_elevation` (when given) must be between 10 and 300 meters, or a
  `ValueError` is raised.
- Missing NASA fill values (`-999`) become pandas missing values.
- Empty API responses become DataFrames with the requested date range and
  requested columns filled with `NaN`.
- Cache failures are logged and do not stop live API fetching.
- If a refresh fails but cached data exists, the cached date range is returned.

### Client Implementation Helpers

These helpers are not intended as public API, but they are useful when reading
the code or writing focused tests.

| Name | Purpose |
|---|---|
| `_session_with_retries(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))` | Create a `requests.Session` with retry/backoff behavior and the package user agent. |
| `_make_cache_key(payload, temporal_api="daily")` | Build a deterministic SHA-256 cache key, excluding dates. |
| `_format_bytes(size)` | Format bytes as `B`, `KiB`, `MiB`, or larger units. |
| `_safe_payload_repr(payload)` | Build a compact, safe string representation of a request payload for logging, tolerating non-JSON-serializable values. |
| `_to_naive(ts)` | Strip timezone information from a pandas `Timestamp`. |
| `_get_date_ranges_to_fetch(requested_start, requested_end, cached_df, temporal_api)` | Identify missing leading/trailing date ranges not present in cached data. |
| `_convert_df_to_cacheable_json(df, temporal_api)` | Convert a parsed DataFrame back into a compact POWER-like JSON shape for storage. |
| `_fetch_and_parse(session, url, payload, temporal_api)` | Execute one GET request and parse the response into a DataFrame plus byte count. |
| `_merge_and_deduplicate(df_list)` | Combine DataFrames, remove duplicate index rows, and sort by date/time. |
| `_filter_df_by_date(df, start, end)` | Return an inclusive date/time slice. |
| `_parse_json_response(resp)` | Decode JSON from a `requests.Response` and raise a clear error on invalid JSON. |
| `_response_to_dataframe(data, temporal_api)` | Convert NASA POWER point JSON into a numeric DataFrame with a `DatetimeIndex`. |
| `_regional_response_to_dataframe(data)` | Parse a GeoJSON FeatureCollection from the regional API into a flat DataFrame with `lat`, `lon`, `elevation` columns. |
| `_ensure_all_params_in_df(df, params)` | Add missing requested parameter columns and order columns as requested. |

`PowerClient` also has private methods for cache I/O, payload building, missing
range fetching, parallel future collection, and Rich table construction:

| Method | Purpose |
|---|---|
| `_init_cache_db()` | Create/connect the SQLite cache database. |
| `_read_from_cache_db(key)` | Read, decompress, and parse one cache entry. |
| `_write_to_cache_db(key, data)` | Compress and write one cache entry. |
| `_format_date(date_str)` | Convert a parseable date to NASA POWER `YYYYMMDD`. |
| `_validate_request(params, is_regional=False)` | Enforce NASA POWER parameter/request constraints before making a request. |
| `_build_point_payload(...)` | Build one point endpoint payload. |
| `_build_regional_payload(...)` | Build one regional bounding-box endpoint payload with bbox validation. |
| `_fetch_and_parse_ranges(ranges, base_payload, url)` | Fetch missing cache date ranges. |
| `_fetch_data(base_payload, url=None)` | Orchestrate cache lookup, live fetching, merging, fallback, and filtering. |
| `_fetch_regional_data(payload)` | Fetch and parse a regional GeoJSON response (no caching). |
| `_parse_points_input(points)` | Normalize point inputs from lists or DataFrames. |
| `_submit_point_futures(executor, parsed_points, start, end, params)` | Submit parallel point requests. |
| `_collect_futures_results(future_to_point)` | Collect parallel results and failed point metadata. |
| `_resolve_transect_num_points(start_coord, end_coord, num_points, spacing_km)` | Resolve and clamp transect sample count; enforces 0.5° minimum spacing. |
| `_build_profile_table(df)` | Build the data profile summary table. |
| `_build_perf_table()` | Build the transfer/cache performance table. |
| `_build_stats_table()` | Build the request/cache hit-rate table. |
| `_build_conn_table()` | Build the NASA POWER connection information table. |

## `aidweather.geo`

The geo module standardizes coordinate parsing and formatting. Internal storage
is always decimal degrees: latitude in `[-90, 90]`, longitude in `[-180, 180]`.

### `GeoCoordinate`

| Method | Purpose | Returns |
|---|---|---|
| `__post_init__()` | Validate coordinate bounds after dataclass construction. | `None` |
| `GeoCoordinate.from_decimal(lat, lon)` | Create from numeric decimal degrees. | `GeoCoordinate` |
| `GeoCoordinate.from_dd_str(lat_str, lon_str)` | Create from decimal-degree strings. | `GeoCoordinate` |
| `GeoCoordinate.from_ddm_str(lat_str, lon_str)` | Create from degree/decimal-minute strings. | `GeoCoordinate` |
| `GeoCoordinate.from_dms_str(lat_str, lon_str)` | Create from degree/minute/second strings. | `GeoCoordinate` |
| `GeoCoordinate.from_strings(lat_str, lon_str)` | Auto-detect DD, DDM, or DMS strings. | `GeoCoordinate` |
| `as_decimal()` | Return `(lat, lon)`. | `tuple[float, float]` |
| `to_dd()` | Return raw decimal-degree tuple. | `tuple[float, float]` |
| `to_dd_str(lat_precision=5, lon_precision=5)` | Format decimal degrees with hemisphere labels. | `(lat_str, lon_str)` |
| `to_ddm_str(minute_precision=3)` | Format as degrees/decimal minutes. | `(lat_str, lon_str)` |
| `to_dms_str(second_precision=0)` | Format as degrees/minutes/seconds. | `(lat_str, lon_str)` |

### Coordinate Functions

| Name | Purpose |
|---|---|
| `normalize_coord_input(lat, lon=None)` | Accept a `GeoCoordinate`, `(lat, lon)` pair, or separate lat/lon values and return a `GeoCoordinate`. |
| `parse_dd(s, is_lat)` | Parse a decimal-degree coordinate string. |
| `parse_ddm(s, is_lat)` | Parse a degree/decimal-minute coordinate string. |
| `parse_dms(s, is_lat)` | Parse a degree/minute/second coordinate string. |
| `parse_any_coord_string(s, is_lat)` | Try DMS, then DDM, then DD parsing. |
| `decimal_to_ddm_components(value, is_lat)` | Convert decimal degrees to DDM components. |
| `decimal_to_dms_components(value, is_lat)` | Convert decimal degrees to DMS components. |

Internal helpers:

| Name | Purpose |
|---|---|
| `_validate_lat_lon(lat, lon)` | Validate geographic bounds. |
| `_hemisphere_from_sign(value, is_lat)` | Choose `N/S` or `E/W` from sign. |
| `_apply_hemisphere_sign(value, hem, is_lat)` | Apply hemisphere sign and validate hemisphere type. |
| `_normalize_coord_string(s)` | Normalize degree/quote variants before regex parsing. |
| `_looks_like_number(x)` | Check whether a value can be converted to `float`. |
| `_normalize_two_values(a, b)` | Normalize two raw coordinate values into a `GeoCoordinate`. |

## `aidweather.config`

The config module loads bundled JSON assets and exposes typed accessors. It is
safe to import even if an asset is missing; hardcoded defaults are used where
needed.

### Configuration Object

| Name | Purpose |
|---|---|
| `cfg` | Module-level singleton `_Config` instance. |
| `_Config(data)` | Thin wrapper around nested config dictionaries. |
| `_Config.__init__(data)` | Store a plain dictionary copy of the loaded config data. |

`_Config` methods:

| Method | Purpose |
|---|---|
| `get(key_path, default=None)` | Read nested config values with dot notation, such as `"base_urls.daily.point"`. |
| `get_url(temporal_api, endpoint_type="point")` | Resolve a NASA POWER base URL, falling back to hardcoded defaults. |
| `params(group="default")` | Return a `{parameter_code: short_name}` mapping. |
| `param_groups()` | Return available parameter group names. |
| `param_descriptions()` | Return full agronomic descriptions keyed by parameter code. |
| `cache_config()` | Return cache settings with environment-aware path resolution. |
| `logging_config()` | Return logging settings with defaults. |
| `api_limits()` | Return API limit metadata from config. |

Module functions:

| Name | Purpose |
|---|---|
| `get_config()` | Return the `cfg` singleton. |
| `_load_config_dict()` | Load `assets/config.json`; return `{}` on missing/invalid assets. |

## `aidweather.utils`

The utils module contains DataFrame helpers intended for downstream analysis
pipelines.

| Class/Function | Purpose |
|---|---|
| `DateColumnOptions` | Dataclass holding options used by `ensure_date_column`. |
| `ensure_date_column(df, name="date", **kwargs)` | Ensure a DataFrame has a timezone-naive `datetime64[ns]` column with the requested name. |

`ensure_date_column` keyword options:

| Option | Default | Purpose |
|---|---|---|
| `inplace` | `False` | Modify the original DataFrame instead of returning a copy. |
| `candidates` | `None` | Alternative source column names to try. |
| `index_fallback` | `True` | Use a `DatetimeIndex` if no source column is found. |
| `normalize` | `False` | Normalize datetimes to midnight. |
| `strip_timezone` | `True` | Remove timezone information. |

Internal helpers:

| Name | Purpose |
|---|---|
| `_find_date_source_column(work, name, candidates)` | Choose the source column for dates. |
| `_coerce_date_column(work, name, src_col, index_fallback, candidates)` | Parse/copy the date source into the target column. |
| `_standardize_datetime_column(work, name, strip_timezone, normalize)` | Remove timezone and normalize the target datetime column. |

## `aidweather.cli`

The CLI module defines the installed `aidweather` command using Typer. These
functions are primarily command handlers, not library API, but they are useful
for automation and testing.

Top-level CLI helpers:

| Name | Purpose |
|---|---|
| `main(version=False)` | Typer callback that handles global options. |
| `_version_callback(value)` | Print `aidweather <version>` and exit when `--version` is used. |
| `_parse_date(date_str)` | Parse CLI date input and return `YYYYMMDD`. |
| `_resolve_output_format(output, fmt)` | Resolve output format from file extension first, then `--format`. |
| `_save_output(df, output, fmt)` | Save a DataFrame as CSV, JSON, or Parquet. |

CLI commands:

| Function | Command | Purpose |
|---|---|---|
| `fetch(...)` | `aidweather fetch` | Fetch one point and optionally save/preview/summarize output. |
| `fetch_multi(...)` | `aidweather fetch-multi` | Read a CSV of points and fetch all points. |
| `fetch_transect(...)` | `aidweather fetch-transect` | Generate and fetch a spatial transect. |
| `fetch_regional(...)` | `aidweather fetch-regional` | Fetch regional grid data for a bounding box (≤ 4.5° × 4.5°, one parameter). |
| `params_list(group="default")` | `aidweather params list` | Print configured NASA POWER parameter codes. |
| `params_describe(code)` | `aidweather params describe CODE` | Print one parameter's full description. |
| `cache_info()` | `aidweather cache info` | Show cache location, size, and entry counts. |
| `cache_clear(yes=False)` | `aidweather cache clear` | Delete the local SQLite cache after confirmation. |

Output format rule:

- If `--output` ends in `.csv`, `.json`, `.parquet`, or `.pq`, the extension
  decides the saved format.
- If the extension is not recognized, `--format` decides the format.
- If neither is available, CSV is used.
