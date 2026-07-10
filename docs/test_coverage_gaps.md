# Test Coverage Gaps

> Snapshot as of 2026-07-10. This is a point-in-time list, not a live/exhaustive audit — it will
> go stale as the code changes. The standing rule in `CLAUDE.md` ("Test Coverage") is what's meant
> to prevent *new* gaps from accumulating; this file just captures what's already known.

`pytest-cov` is a declared `test` extra in `pyproject.toml` but isn't wired into any documented
pytest invocation (no `--cov` flags, no `.coveragerc`), so it produces no coverage report. These
gaps were identified by manually cross-referencing `tests/` against `src/aidweather/`, not from a
coverage report.

## `client.py`

- **Cache corruption**: `_read_from_cache_db` hitting a corrupted BLOB (`gzip.BadGzipFile` or
  `json.JSONDecodeError` on decompress/parse) is untested.
- **`_init_cache_db` failure path**: `sqlite3.Error` during connect/create-table (e.g. an
  unwritable cache directory), which falls back to `db_conn=None`, is untested.
- **Raw `OSError` re-raise**: `_fetch_and_parse`/the regional fetch path re-raising `OSError` when
  a live fetch fails *and* no cached data exists at all — only the "stale cache fallback when data
  exists" path is tested, not the no-cache failure path.
- **HTTP 429 / retry backoff**: the `Retry`/`HTTPAdapter` configuration in
  `_session_with_retries` and any special 429 handling aren't exercised by a test that actually
  triggers a retry sequence.
- **`get_regional_data` GeoJSON edge cases**: features with `coords` length < 2 (skipped), missing
  elevation, an empty `features` list, and multiple parameters merged into `date_map` — only one
  happy-path GeoJSON fixture exists.
- **`_response_to_dataframe` mixed-date-key-length branch**: the `len(sample_key) not in (8, 10)`
  fallback and its failure path (`return pd.DataFrame()`) are untested.
- **Cache round-trip integrity**: `_convert_df_to_cacheable_json`'s NaN→`-999` fill on write and
  the corresponding `-999`→`pd.NA` on read aren't explicitly asserted for edge values beyond the
  main happy path.
- **`get_multi_point_data`/`_parse_points_input`**: DataFrame input and `(lat, lon, elevation)`
  tuple input appear untested — only list-of-dict input is exercised.
- **`_resolve_transect_num_points`**: the `ValueError` branch (`spacing_km <= 0`, or neither
  `num_points` nor `spacing_km` given) isn't directly unit-tested at the client level (only
  indirectly via the CLI).

## `geo.py`

- DMS seconds/minutes ≥60 (the parsing regex allows any digit string, so out-of-range values
  aren't rejected — untested either way).
- Conflicting sign + hemisphere letter (e.g. `-5° N`) — behavior isn't asserted.
- The fallback-through-all-three-parsers error message in `parse_any_coord_string` when nothing
  matches.

## `config.py`

- True end-to-end precedence isn't verified: env var wins over `config.json` cache path when
  **both** `AIDWEATHER_CACHE_DIR` and a JSON `cache_config.path` are set simultaneously. Existing
  tests check the env-cleared case and the JSON-set case individually, not together.

## `cli.py`

- **`fetch-regional`** has no test at all.
- **`fetch-multi`** and **`fetch-transect`** aren't tested for `json`/`parquet` output (only
  `fetch` gets that treatment).
- **`_save_output`** write-failure/permission-error path (should trigger `typer.Exit(code=1)`) is
  untested.
- **`cache info`/`cache clear`**: only the empty-cache case is tested, not the populated-cache
  stats path (entry count, oldest/newest timestamp, compressed size) or `sqlite3.Error` handling.

## Concurrency

- The rate limiter is tested for throttling *timing* but not correctness under genuine concurrent
  multi-thread bursts (only sequential calls are verified).
- SQLite lock-contention/timeout behavior (`timeout=10`) has only a simple concurrent-write test,
  without asserting correctness under real contention or failure.
