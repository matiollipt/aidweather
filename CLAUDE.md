# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aidweather` is a Python library + CLI for fetching, caching, and validating daily/hourly weather
and solar data from NASA's POWER API, aimed at agricultural applications. Public API is pre-1.0
(beta) and may still change.

## Commands

```bash
# Install for local dev (creates .venv, installs dev tools: pytest, ruff, mypy, build)
./install.sh --dev

# Run the test suite (network calls mocked via requests-mock by default)
uv run --with-editable . --extra test pytest -q

# Run a single test file / test
uv run --with-editable . --extra test pytest tests/test_client.py -q
uv run --with-editable . --extra test pytest tests/test_client.py::test_name -q

# Run live tests against the real NASA POWER API (normally skipped)
AIDWEATHER_RUN_LIVE_TESTS=1 uv run --with-editable . --extra test pytest

# Lint / type check
uv run ruff check src/
uv run mypy src/aidweather
```

Full release process (version bump, TestPyPI dry run, wheel smoke test, PyPI publish) is in
`docs/release_checklist.md` — follow it before tagging a release.

## Architecture

Four modules under `src/aidweather/`: `client.py` (the API surface), `config.py`, `geo.py`,
`cli.py`, plus `utils.py` for shared DataFrame helpers.

**`config.py`** — `cfg` is a module-level singleton loaded once from the bundled
`assets/config.json` (via `importlib.resources`, so it works from an installed wheel). If the
JSON is missing/malformed, it silently falls back to hardcoded defaults so the package stays
importable. Resolution precedence for cache/log paths: env var (`AIDWEATHER_CACHE_DIR`,
`AIDWEATHER_LOG_DIR`) > `config.json` > platformdirs XDG default. `cfg.api_limits()` and
`cfg.cache_config()` are read once by `PowerClient.__init__`, not live per-call.

**`client.py` — `PowerClient`** — wraps NASA POWER's point and regional endpoints;
`temporal_api` ("daily"/"hourly") is fixed at construction.

- *Networking*: a `requests.Session` with urllib3 `Retry` (5 retries, exponential backoff on
  429/500/502/503/504) layered under an app-level thread-safe `RateLimiter` (sliding window,
  default 30 calls/60s) acquired before every call. `max_workers` for parallel fetches is
  clamped to `api_limits.max_workers` (default 5, matching NASA's recommended concurrency cap) —
  excess is silently reduced with a warning log, not an error.
- *Caching*: SQLite at `<cache_dir>/aidweather_cache.db`, opened `check_same_thread=False` with
  a busy timeout, guarded by an in-process `threading.Lock`. Safe for the thread pool used by
  multi-point/transect fetches, but not for multiple *processes* sharing one DB file beyond
  SQLite's own file locking. Cached blobs are gzip-compressed JSON. Cache key is a `"v1_"`-prefixed
  SHA-256 hash of the request payload with `start`/`end` stripped out — so one row covers a
  location+params+temporal_api regardless of date range (bump to `"v2_"` if the cache schema ever
  changes, per `docs/technical_debt.md`). `_get_date_ranges_to_fetch` diffs the requested span
  against the cached span and only fetches the missing edges, then merges and re-caches. Coverage
  is inferred purely from timestamps present in the cached data ("look and understand" strategy,
  documented in the function's own docstring and `docs/technical_debt.md` item 4) — it relies on
  NASA always returning a key per day/hour (never omitting one), so a genuinely truncated trailing
  response would keep re-fetching that edge on every call rather than caching it, by design. If a
  live fetch raises `OSError` but stale cached data exists, the stale data is served instead of raising
  — *except* for HTTP 400/422 (client-side validation errors), which are always re-raised even when
  stale data is available, since retrying or serving stale data for a malformed request would mask
  the bug.
- *Endpoints*: `get_point_data`/`get_point_data_from_coordinate` (single location, up to 20
  daily / 15 hourly params, uses the cache path above); `get_multi_point_data` and
  `get_transect_data` fan out point calls across a `ThreadPoolExecutor` (transect points via
  `np.linspace`, with spacing clamped to the requested parameters' native grid resolution — 0.5°
  (~55km) for MERRA-2-sourced params, 1.0° (~111km) for CERES-sourced params like
  `ALLSKY_SFC_SW_DWN`, taking the finest among requested params as the floor — silently reduces
  point count with an INFO log rather than erroring); `get_regional_data`
  (bounding-box GeoJSON, hard-capped to 1 parameter and a 4.5°×4.5° box, validated pre-request
  with `ValueError` — does not use the point cache/date-diff mechanism, parses a different
  response shape).
- *Data conventions*: fill value `-999` is converted to `pd.NA` everywhere. NASA POWER
  timestamps are Local Solar Time (LST), not civil time. `parse_date_strict` deliberately rejects
  ambiguous slash dates (e.g. `"05/03/2023"`, raises `AmbiguousDateError`) since NASA/pandas
  disagree on day-first vs. month-first defaults; the CLI surfaces this as `typer.BadParameter`.

**`geo.py` — `GeoCoordinate`** — frozen dataclass validating lat∈[-90,90], lon∈[-180,180].
Parses DD, DDM, and DMS coordinate strings with Unicode normalization (curly quotes, `''`→`"`,
`º/˚`→`°`); when hemisphere letters are omitted, DD trusts the numeric sign while DDM/DMS default
to N/E. `normalize_coord_input` accepts a `GeoCoordinate`, a (lat, lon) tuple/list, or two
separate numeric/string args.

**`cli.py`** — Typer app (`aidweather`) with `params` and `cache` subgroups plus top-level
`fetch`, `fetch-multi`, `fetch-transect`, `fetch-regional` commands, each a thin wrapper over the
matching `PowerClient` method. Shares `_parse_date` (via `parse_date_strict`) and `_save_output`
(format inferred from file extension: csv/json/parquet).

**`utils.py`** — `ensure_date_column` is the one shared primitive: locates a date column by name
or falls back to a DatetimeIndex, coercing to `datetime64[ns]`. Needed because different fetch
paths (indexed vs. reset_index'd, single-point vs. multi-point) produce dates in different shapes.

## NASA POWER API guardrails

Documented in `docs/parameter_provenance.md` and `docs/spatial_interpretation.md` — respect these when writing code that calls
the API directly rather than through `PowerClient`:

- Max 20 params per daily point request, 15 for hourly, 1 for regional.
- Max bounding box for regional requests: 4.5° × 4.5°.
- Max 5 concurrent connections (NASA's own recommendation).
- Timestamps are Local Solar Time, not civil/UTC time.
- The on-disk cache is unencrypted.
- For large/bulk historical extraction, NASA's AWS S3 Zarr ARD store is recommended over the
  live API — don't try to bulk-fetch years of data through `PowerClient` in a loop.

## Tests

`tests/conftest.py` auto-patches `PowerClient.__init__` to track and close all `db_conn`s per
test, avoiding leaked-SQLite-connection warnings. Standard tests stub HTTP via the `requests_mock`
fixture — no real network calls. `tests/test_live.py` contains real-network tests gated behind
`AIDWEATHER_RUN_LIVE_TESTS=1` (skipped by default, see Commands above).

## Code Style & Conventions

These codify what the codebase already does consistently — match them rather than introducing new
patterns.

- **Docstrings**: Google-style (`Args:`/`Returns:`/`Raises:`) for public functions/methods and any
  non-trivial private helper. A one-line docstring is fine for small/obvious private helpers (e.g.
  `_format_bytes`). Module docstrings use a `~~~~` underline header with an `Example:` doctest
  block where useful (see `geo.py`, `client.py`).
- **Typing**: full type hints on every signature, including private helpers.
  `from __future__ import annotations` in every module. Pydantic `BaseModel` for request/payload
  objects (`PointRequest`, `TransectRequest`, `RegionalRequest`); frozen `@dataclass` for value
  objects (`GeoCoordinate`). `Literal["daily", "hourly"]` for the temporal-api parameter.
- **Naming**: PascalCase classes, snake_case functions, `_`-prefixed module/class-private helpers,
  ALL_CAPS module constants (`_`-prefixed too if private). Verb-prefix families are consistent —
  extend them rather than inventing new ones: `_validate_*`, `_build_*`, `_parse_*`/`parse_*`,
  `_get_*`, `_format_*`, `_resolve_*`.
- **File organization**: imports grouped stdlib → third-party → local; constants/regex after
  imports; module logger next; then free functions, then Pydantic models, then the main class.
  Large files use `# --- Section Name ---` banner comments to divide logical regions (see
  `client.py`'s `Module-level Helpers` / `Parsing Helpers` / `Main Client Class` banners, and
  `geo.py`'s equivalent) — add banners once a file grows multiple logical sections, don't force
  them on small files.
- **Errors**: custom exceptions subclass a builtin and end in `Error`
  (`AmbiguousDateError(ValueError)`, `APIRequestError(OSError)`); validation raises plain
  `ValueError` with an f-string naming
  the offending value and constraint; network failures are normalized to
  `OSError(f"...: {e}") from e` so callers only need to catch one type; CLI commands uniformly
  wrap client calls in `try/except Exception as e: console.print(...); raise
  typer.Exit(code=1) from e`.
- **Comments**: moderate density, explaining *why* (rationale, workarounds, step-labeling in
  multi-stage logic) — not restating what the code already says.
- **Logging**: `logger = logging.getLogger(__name__)` per module (`cli.py`/`geo.py`/`utils.py` use
  `rich.Console` for user-facing output instead and don't need their own logger). `info` for
  normal flow/cache events, `warning` for degraded-but-recoverable situations, `error` for real
  failures.

## Documentation Accuracy

Treat claims in `README.md`, `docs/*.md`, and this file (limits, defaults, cache paths, behavior
descriptions) as claims, not facts — verify them against the actual source (e.g.
`assets/config.json`, the relevant `client.py` validation code) before relying on them or
repeating them elsewhere. If you find drift between docs and code, fix the documentation in the
same change rather than leaving it for later.

## Test Coverage

Before considering a change to `client.py`, `config.py`, `geo.py`, `cli.py`, or `utils.py`
complete, check whether the touched function/branch has corresponding coverage in `tests/`, and
add tests for meaningfully new behavior or previously-untested error paths you touched. If you
notice a gap you're not fixing, say so (one line, with why) rather than leaving it silent.

Technical debt and release limits are tracked in `docs/technical_debt.md`.
