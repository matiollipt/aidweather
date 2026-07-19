# SPDX-License-Identifier: Apache-2.0
"""
aidweather.client
~~~~~~~~~~~~~~~~~

NASA POWER API client with SQLite caching and retry logic.

Exposes :class:`PowerClient` for fetching daily and hourly meteorological data
from the NASA POWER point and regional endpoints. The client:

- Validates and normalises geographic coordinates via :mod:`aidweather.geo`.
- Builds request payloads conforming to the NASA POWER query schema.
- Maintains a local gzip-compressed SQLite cache keyed by a SHA-256 digest
  of the request payload (excluding date range), enabling gap-aware partial
  fetching — only missing date segments are fetched from the network.
- Enforces a sliding-window rate limiter and a thread-pool size ceiling to
  comply with NASA POWER API guidelines.
- Returns :class:`pandas.DataFrame` objects with a :class:`~pandas.DatetimeIndex`
  named ``"date"`` and one column per requested parameter.

Missing values (NASA POWER fill code ``-999``) are replaced with
:data:`pandas.NA` on read. Data invariants (no silent imputation, preserved
units, coordinate precision) are documented in ``docs/developer_guide.md``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from urllib3.util.retry import Retry

from aidweather import __version__
from aidweather.config import cfg
from aidweather.geo import GeoCoordinate

__all__ = ["PowerClient"]


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Enforces a maximum number of calls within a rolling time window. Designed
    for use with the NASA POWER API, which recommends no more than 30 requests
    per 60-second window per client.

    Attributes:
        max_calls: Maximum number of calls allowed within *period* seconds.
        period: Length of the sliding window in seconds.
        lock: Threading lock protecting the call-history list.
        calls: Timestamps (``time.time()``) of recent calls within the window.
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls: list[float] = []

    def acquire(self) -> None:
        """Block the calling thread until a call slot is available.

        If ``max_calls <= 0`` or ``period <= 0``, the limiter is effectively
        disabled and this method returns immediately. Otherwise it spins in a
        lock-protected loop, sleeping until the oldest call in the sliding
        window has aged out and a new slot becomes available.
        """

        # If no rate limiting is configured, no limiting is applied.
        if self.max_calls <= 0 or self.period <= 0:
            return

        while True:
            with self.lock:

                # Record the current time.
                now = time.time()

                # Clean up calls older than the sliding window period
                self.calls = [t for t in self.calls if now - t < self.period]

                # If the number of calls within the sliding window is less than the maximum
                # number of calls allowed, add the current time to the list of calls and return.
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                # Calculate the time to sleep to respect the rate limit.
                sleep_time = self.calls[0] + self.period - now

            # Sleep until the next call is allowed
            if sleep_time > 0:
                time.sleep(sleep_time)


# --- Module-level Helpers ---

# The user agent string for requests to the NASA POWER API. It identifies the
# library as a good citizen and provides contact information in case of issues.
USER_AGENT = f"aidweather/{__version__} (+https://github.com/matiollipt/aidweather)"


def _session_with_retries(
    total: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Build a ``requests.Session`` with automatic retry and connection pooling.

    Mounts a :class:`~requests.adapters.HTTPAdapter` with
    :class:`~urllib3.util.retry.Retry` on both ``http://`` and ``https://``
    prefixes. The ``User-Agent`` header is set to :data:`USER_AGENT`.

    Args:
        total: Total number of retries (also applied to read, connect, and
            status sub-limits). Defaults to ``5``.
        backoff_factor: Sleep multiplier between retries
            (``sleep = backoff_factor * 2 ** (attempt - 1)``). Defaults to
            ``0.5``.
        status_forcelist: HTTP status codes that trigger a retry. Defaults to
            ``(429, 500, 502, 503, 504)``.

    Returns:
        A configured :class:`requests.Session` ready for use.
    """
    # Added read, connect, status retries to total retries
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        status=total,
        backoff_factor=backoff_factor,
        allowed_methods=frozenset(["GET"]),
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )

    # added pool_connections and pool_maxsize to requests session
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# Configure logging for the module
logger = logging.getLogger(__name__)

_AMBIGUOUS_SLASH_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")


class AmbiguousDateError(ValueError):
    """Raised when a date string's day/month order cannot be determined."""


class APIRequestError(OSError):
    """Raised when a NASA POWER API request fails after all retries are exhausted."""


def parse_date_strict(date_value: Any) -> pd.Timestamp:
    """Parse a date value into a :class:`pandas.Timestamp`, rejecting ambiguous slash formats.

    Slash-separated date strings (e.g. ``"05/03/2023"``) are explicitly
    rejected because NASA POWER's day-first convention and pandas' default
    month-first parsing would silently disagree on the day/month order.

    Args:
        date_value: A date-like value — ISO string (``"2023-01-15"``),
            ``YYYYMMDD`` string, :class:`datetime.datetime`,
            :class:`datetime.timedelta`, or any value accepted by
            :func:`pandas.to_datetime`.

    Returns:
        A tz-naive :class:`pandas.Timestamp`.

    Raises:
        AmbiguousDateError: If *date_value* is a slash-separated string whose
            day/month order is ambiguous.
        ValueError: If *date_value* cannot be parsed as a date.
    """
    if isinstance(date_value, str) and _AMBIGUOUS_SLASH_DATE_RE.match(
        date_value.strip()
    ):
        raise AmbiguousDateError(
            f"Ambiguous date '{date_value}': day/month order is not clear from "
            "a slash-separated date. Use an unambiguous format instead, e.g. YYYY-MM-DD."
        )
    return pd.to_datetime(date_value)


def _make_cache_key(payload: dict[str, Any], temporal_api: str = "daily") -> str:
    """Return a versioned SHA-256 cache key for *payload*.

    The ``"start"`` and ``"end"`` date keys are excluded from the digest so
    that the same spatial request with different date ranges maps to the same
    cache key, allowing the cache layer to store the full fetched history
    under one key and fetch only missing date segments.

    The returned string is prefixed with ``"v1_"`` for schema versioning.

    Args:
        payload: The NASA POWER query parameter dict.
        temporal_api: Temporal resolution string appended to the payload
            before hashing. Defaults to ``"daily"``.

    Returns:
        A ``"v1_<sha256hex>"`` string uniquely identifying the spatial request.
    """
    key_payload = payload.copy()
    key_payload.pop("start", None)
    key_payload.pop("end", None)
    key_payload["_temporal_api"] = temporal_api
    encoded = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    return "v1_" + hashlib.sha256(encoded).hexdigest()


def _format_bytes(size: float) -> str:
    """Formats a byte count into a human-readable string (e.g., KiB, MiB)."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _safe_payload_repr(payload: Any) -> str:
    """Creates a safe, compact string representation of a request payload.

    Tolerates non-JSON-serializable values and falls back to repr if needed.
    """
    if not isinstance(payload, dict):
        return repr(payload)
    try:
        # Copy the payload to avoid modifying the original
        clean_payload = dict(payload)
        # Redact or truncate long values if any (e.g. if parameters is extremely long)
        for k, v in clean_payload.items():
            if isinstance(v, str) and len(v) > 200:
                clean_payload[k] = v[:197] + "..."
        return json.dumps(clean_payload, sort_keys=True, default=str)
    except Exception:
        return repr(payload)


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Strip timezone info from a Timestamp, safe for already tz-naive values."""
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _get_date_ranges_to_fetch(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    cached_df: pd.DataFrame | None,
    temporal_api: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the date sub-ranges not yet covered by *cached_df*.

    Compares the requested date window against the min/max index of the cached
    DataFrame and returns a list of ``(start, end)`` tuples representing the
    gaps that still need to be fetched from the API.

    "Look and understand" coverage strategy — read this before touching cache
    logic: coverage is inferred purely from the timestamps *present* in
    *cached_df*, not from any separately stored record of what date range was
    actually requested. This relies on an invariant of the NASA POWER API: a
    response for a given day/hour always includes a key for every day/hour in
    that span (using the ``-999`` fill code for missing values), never
    omitting a key outright. Under that invariant, "furthest timestamp we've
    seen" and "furthest date NASA has confirmed data for" are the same thing,
    so no separate coverage bookkeeping is needed.

    Consequence if that invariant is ever violated (e.g. a response that
    truncates rather than fill-codes a gap — see the provisional-data-tail
    caveat in ``docs/parameter_provenance.md``): this function will keep
    reporting the trailing dates as "not yet covered" on every call, so a
    request touching that range re-fetches from the API every time instead of
    being served from cache. This is intentional, not a bug — it self-heals
    once NASA backfills the gap — but it does mean the cache gives no
    "call once and never again" guarantee for very recent/provisional dates.
    See ``docs/technical_debt.md`` for the fuller write-up of this trade-off.

    Args:
        requested_start: Inclusive start of the requested date range.
        requested_end: Inclusive end of the requested date range.
        cached_df: Previously cached DataFrame with a DatetimeIndex, or
            ``None`` / empty if no cache entry exists.
        temporal_api: ``"daily"`` or ``"hourly"`` — controls the granularity
            of gap edges (1-day vs 1-hour steps).

    Returns:
        A list of ``(start, end)`` tuples. Returns a single-element list
        covering the full requested range when the cache is empty or ``None``.
        Returns an empty list when the cache fully covers the request.
    """
    if cached_df is None or cached_df.empty:
        return [(requested_start, requested_end)]

    cached_start = _to_naive(cached_df.index.min())
    cached_end = _to_naive(cached_df.index.max())
    req_start = _to_naive(requested_start)
    req_end = _to_naive(requested_end)

    to_fetch = []
    delta = pd.Timedelta(hours=1) if temporal_api == "hourly" else pd.Timedelta(days=1)

    if req_start < cached_start:
        to_fetch.append((req_start, cached_start - delta))

    if req_end > cached_end:
        to_fetch.append((cached_end + delta, req_end))

    return to_fetch


def _convert_df_to_cacheable_json(
    df: pd.DataFrame, temporal_api: str
) -> dict[str, Any]:
    """Convert a DataFrame back into the JSON dict format used for cache storage."""
    df_copy = df.copy()
    date_format = "%Y%m%d%H" if temporal_api == "hourly" else "%Y%m%d"
    df_copy.index = pd.DatetimeIndex(df_copy.index).strftime(date_format)
    # NASA POWER uses -999 as its fill value. We replicate that sentinel here so
    # the cache stores data in the same format as the raw API response.
    # NOTE: any genuine measurement of exactly -999 will be treated as missing on
    # read-back (_response_to_dataframe); this is an accepted trade-off for safe
    # data analysis.
    df_copy = df_copy.fillna(-999)
    param_dict = df_copy.to_dict(orient="dict")
    return {"properties": {"parameter": param_dict}}


def _fetch_and_parse(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    temporal_api: Literal["daily", "hourly"],
) -> tuple[pd.DataFrame, int]:
    """Execute a single GET request and return the parsed result.

    Handles HTTP 400/422 error bodies with structured NASA POWER ``messages``
    fields, raising an :class:`APIRequestError` with the joined message text.
    Raises :class:`APIRequestError` on connection/timeout errors.

    Args:
        session: The :class:`requests.Session` to use.
        url: Full NASA POWER API endpoint URL.
        payload: Query parameter dict.
        temporal_api: Temporal resolution — ``"daily"`` or ``"hourly"``.

    Returns:
        A ``(DataFrame, byte_count)`` tuple where *DataFrame* has a
        :class:`~pandas.DatetimeIndex` named ``"date"`` and *byte_count* is
        the raw response size in bytes. Returns ``(empty DataFrame, 0)`` if
        the API response contains an ``"error"`` key.

    Raises:
        APIRequestError: If the request fails after all retries, or if the
            server returns HTTP 400/422 with a parseable error message.
    """
    try:
        resp = session.get(url, params=payload)

        if resp.status_code in (400, 422):
            try:
                err_data = resp.json()
                if isinstance(err_data, dict) and "messages" in err_data:
                    err_msg = "; ".join(err_data["messages"])
                    raise APIRequestError(
                        f"NASA POWER API Error ({resp.status_code}): {err_msg}"
                    )
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        resp.raise_for_status()
        byte_count = len(resp.content)
        data = _parse_json_response(resp)

        if "error" in data:
            logger.error("API Error for payload %s: %s", payload, data.get("error"))
            return pd.DataFrame(), 0

        return _response_to_dataframe(data, temporal_api), byte_count

    except requests.exceptions.RequestException as e:
        resp_obj = getattr(e, "response", None)
        if resp_obj is not None and getattr(resp_obj, "status_code", None) == 429:
            logger.error("Rate limit exceeded (HTTP 429). Please slow down requests.")
        logger.error("API request failed for payload %s: %s", payload, e)
        raise APIRequestError(f"API request failed: {e}") from e


def _merge_and_deduplicate(
    df_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate, deduplicate, and sort a list of DataFrames by index."""
    if not df_list:
        return pd.DataFrame()

    processed_dfs = []
    for raw_df in df_list:
        df_work = raw_df
        if "date" in df_work.columns and not isinstance(
            df_work.index, pd.DatetimeIndex
        ):
            df_work = df_work.set_index("date")
        processed_dfs.append(df_work)

    combined_df = pd.concat(processed_dfs)
    final_df = combined_df[~combined_df.index.duplicated(keep="first")]
    return final_df.sort_index()


def _filter_df_by_date(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Return *df* filtered to the inclusive date range [*start*, *end*]."""
    if df.empty:
        return df
    return df.loc[start:end]


# --- Parsing Helpers ---


def _parse_json_response(resp: requests.Response) -> dict[str, Any]:
    """Parse and return the JSON body of *resp*, raising ``ValueError`` on failure."""
    try:
        return dict(resp.json())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"API request failed (Status {resp.status_code}). "
            f"Could not decode JSON from response. Text: '{resp.text[:200]}...'"
        ) from e


def _response_to_dataframe(
    data: dict[str, Any], temporal_api: Literal["daily", "hourly"]
) -> pd.DataFrame:
    """Parse a NASA POWER JSON response body into a DatetimeIndex DataFrame.

    Extracts the ``properties.parameter`` dict from *data*, constructs a
    :class:`~pandas.DataFrame` with date strings as the index, converts the
    index to a :class:`~pandas.DatetimeIndex`, and replaces the NASA POWER
    fill value ``-999`` with :data:`pandas.NA`.

    Args:
        data: Parsed JSON response dict. Expected structure:
            ``{"properties": {"parameter": {param: {date_str: value}}}}``.
        temporal_api: ``"daily"`` or ``"hourly"`` — determines the expected
            date key format (``YYYYMMDD`` vs ``YYYYMMDDHH``).

    Returns:
        A :class:`~pandas.DataFrame` with a :class:`~pandas.DatetimeIndex`
        named ``"date"`` and one numeric column per parameter. Missing values
        are :data:`pandas.NA`. Returns an empty DataFrame if *data* contains
        no parameter series.

    Raises:
        ValueError: If the date key format in the response is neither 8- nor
            10-character (i.e. not ``YYYYMMDD`` or ``YYYYMMDDHH``).
    """
    properties = data.get("properties", {}).get("parameter", {})
    if not properties:
        return pd.DataFrame()

    df = pd.DataFrame(properties)
    if df.empty:
        return pd.DataFrame()

    sample_key = str(df.index[0])
    if len(sample_key) == 10:
        date_format = "%Y%m%d%H"
    elif len(sample_key) == 8:
        date_format = "%Y%m%d"
    else:
        # "mixed" would be the only remaining option, but it requires pandas ≥ 2.0.
        # Raise explicitly with a clear diagnostic instead of letting pd.to_datetime
        # fail silently and return an empty DataFrame with no explanation.
        raise ValueError(
            f"Unrecognised date key format '{sample_key}': expected 8-char daily "
            "(YYYYMMDD) or 10-char hourly (YYYYMMDDHH). "
            "Note: the 'mixed' fallback requires pandas \u2265 2.0."
        )

    try:
        df["date"] = pd.to_datetime(df.index, format=date_format)
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to parse date index with format '%s': %s. Returning empty.",
            date_format,
            exc,
        )
        return pd.DataFrame()

    df = df.reset_index(drop=True).set_index("date")
    # Replace NASA POWER's -999 fill value with pd.NA for safe downstream analysis.
    # See also: _convert_df_to_cacheable_json where NaN→-999 round-trip is applied.
    df = df.replace(-999, pd.NA)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_index()


def _ensure_all_params_in_df(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """Add any missing *params* columns to *df*, filled with ``pd.NA``."""
    for param in params:
        if param not in df.columns:
            df[param] = pd.NA
    return df[params]


def _regional_response_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Parse a NASA POWER regional GeoJSON FeatureCollection into a long-form DataFrame.

    Iterates over ``features``, extracting the ``geometry.coordinates``
    (longitude, latitude, optional elevation) and the
    ``properties.parameter`` time-series for each grid cell. Assembles a
    long-form record list with columns ``date``, ``lat``, ``lon``,
    optionally ``elevation``, and one column per parameter.

    Args:
        data: Parsed GeoJSON response dict. Expected structure::

            {"features": [{"geometry": {"coordinates": [lon, lat]},
                           "properties": {"parameter": {param: {date_str: value}}}}]}

    Returns:
        A long-form :class:`~pandas.DataFrame` indexed by ``"date"`` sorted
        ascending, with columns ``"lat"``, ``"lon"``, optionally
        ``"elevation"``, and one column per parameter. Missing values are
        :data:`pandas.NA`. Returns an empty DataFrame if *data* has no
        features or all features lack valid coordinates.
    """
    features = data.get("features", [])
    if not features:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for feature in features:
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        elevation = coords[2] if len(coords) > 2 else None

        params_data = feature.get("properties", {}).get("parameter", {})
        # Build a dict of {date_str: {param: value, ...}} across all params
        date_map: dict[str, dict[str, Any]] = {}
        for param_name, time_series in params_data.items():
            for date_str, value in time_series.items():
                if date_str not in date_map:
                    date_map[date_str] = {}
                date_map[date_str][param_name] = value if value != -999 else pd.NA

        for date_str, param_values in date_map.items():
            record: dict[str, Any] = {
                # Regional endpoint only supports daily resolution; hourly regional
                # is not available via the NASA POWER API (hardcoded daily format).
                "date": pd.to_datetime(date_str, format="%Y%m%d"),
                "lat": lat,
                "lon": lon,
            }
            if elevation is not None:
                record["elevation"] = elevation
            record.update(param_values)
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("date").sort_index()
    return df


class PowerQuery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    start: datetime | str | timedelta | pd.Timestamp
    end: datetime | str | timedelta | pd.Timestamp
    params: list[str]


class PointRequest(PowerQuery):
    lat: float
    lon: float
    elevation: float | None = None
    wind_elevation: float | None = None
    wind_surface: float | None = None


class TransectRequest(PowerQuery):
    """Request model for a 1D transect of individual point API calls.

    Defines a straight-line transect between two ``GeoCoordinate`` endpoints.
    Points are sampled along the path at a minimum spacing of 0.5° (~55 km)
    to match the NASA POWER native grid resolution — requesting denser
    spacing would return duplicate data.

    Attributes:
        start_coord: Starting endpoint of the transect.
        end_coord: Ending endpoint of the transect.
        num_points: Number of sample points (takes priority over
            ``spacing_km`` when both are supplied).
        spacing_km: Approximate spacing between samples in kilometres.
            Derived and logged when ``num_points`` is also given.
        max_workers: Thread-pool size for concurrent point fetching.
    """

    start_coord: GeoCoordinate
    end_coord: GeoCoordinate
    num_points: int | None = None
    spacing_km: float | None = None
    max_workers: int = 5


class RegionalRequest(PowerQuery):
    """Request model for the NASA POWER regional bounding-box endpoint.

    The NASA POWER regional API accepts a geographic bounding box defined
    by four coordinates and returns data on a 0.5° × 0.5° grid within
    that box. The bounding box must not exceed 4.5° on either axis.

    Attributes:
        lat_min: Southern edge of the bounding box (latitude).
        lat_max: Northern edge of the bounding box (latitude).
        lon_min: Western edge of the bounding box (longitude).
        lon_max: Eastern edge of the bounding box (longitude).
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# --- Main Client Class ---


class PowerClient:
    """NASA POWER API client with SQLite caching, retry logic, and rate limiting.

    ``PowerClient`` is the main entry point for fetching agroclimatic and
    solar radiation data from the NASA POWER API. It handles the full request
    lifecycle: coordinate normalisation, payload construction, cache lookup,
    gap-aware partial fetching, response parsing, and cache update.

    Typical usage::

        from aidweather import PowerClient

        client = PowerClient(temporal_api="daily")
        df = client.get_point_data(
            lat=-23.55, lon=-46.63,
            start="2023-01-01", end="2023-01-31",
            params=["T2M", "PRECTOTCORR"],
        )

    See the `User Guide <docs/user_guide.md>`_ and
    `API Reference <docs/api_reference.md>`_ for complete usage examples.

    Attributes:
        temporal_api: Temporal resolution of the active API endpoint.
            Either ``"daily"`` or ``"hourly"``.
        base_url: Base URL for the point endpoint.
        regional_base_url: Base URL for the regional bounding-box endpoint.
        session: The :class:`requests.Session` used for all HTTP calls.
        db_conn: Open :class:`sqlite3.Connection` to the cache database, or
            ``None`` if caching is disabled or failed to initialise.
        cache_cfg: Effective cache configuration dict from
            :func:`~aidweather.config._Config.cache_config`.
        api_limits: API limits dict from
            :func:`~aidweather.config._Config.api_limits`.
        rate_limiter: Active :class:`RateLimiter` instance.
    """

    def __init__(
        self,
        temporal_api: Literal["daily", "hourly"] = "daily",
        session: requests.Session | None = None,
    ):
        """Initialise the client, configure caching, rate limiter, and HTTP session.

        Args:
            temporal_api: Temporal resolution for all API calls made by this
                instance. Must be ``"daily"`` or ``"hourly"``.
            session: Optional pre-configured :class:`requests.Session`. A
                retry-enabled session is created automatically if omitted.

        Raises:
            ValueError: If *temporal_api* is not ``"daily"`` or ``"hourly"``.
        """
        if temporal_api not in ["daily", "hourly"]:
            raise ValueError("`temporal_api` must be 'daily' or 'hourly'.")

        self.temporal_api = temporal_api
        self.base_url = cfg.get_url(temporal_api, endpoint_type="point")
        self.regional_base_url = cfg.get_url(temporal_api, endpoint_type="regional")
        self.params_desc = cfg.params(group="all")
        self.session = session or _session_with_retries()
        self.db_conn: sqlite3.Connection | None = None
        self.db_lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "total_requests": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "total_downloaded_bytes": 0,
            "fetch_duration": 0.0,
            "cache_initial_bytes": 0,
            "cache_final_bytes": 0,
        }

        # Caching setup
        self.cache_cfg = cfg.cache_config()
        if self.cache_cfg.get("enabled", False):
            self._init_cache_db()

        # API Limits & Concurrency Setup
        self.api_limits = cfg.api_limits()
        self.max_workers_limit = self.api_limits.get("max_workers", 5)

        # Rate Limiter Setup
        rate_limit_calls = self.api_limits.get("rate_limit_calls", 30)
        rate_limit_period = self.api_limits.get("rate_limit_period_seconds", 60)
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)

    def _init_cache_db(self) -> None:
        """Initialise the SQLite cache database and create the cache table if absent.

        Raises:
            sqlite3.Error: If the connection or table creation fails.
        """
        db_path = ""
        try:
            cache_dir = Path(self.cache_cfg.get("path", "."))
            cache_dir.mkdir(parents=True, exist_ok=True)

            db_path = str(cache_dir / "aidweather_cache.db")
            # Set a timeout to prevent errors with concurrent writes
            self.db_conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)

            with self.db_conn:
                self.db_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        data BLOB NOT NULL
                    )
                """
                )
        except sqlite3.Error as e:
            logger.error("Failed to initialize cache database at %s: %s", db_path, e)
            self.db_conn = None

    def _validate_inputs(
        self,
        params: list[str],
        start: Any,
        end: Any,
    ) -> None:
        """Validate *params* and the date range; emit warnings for unknown parameter codes.

        Checks that all codes in *params* are in the ``"all"`` parameter group
        and that *start* does not exceed *end*. Unknown codes produce a
        :class:`UserWarning` rather than raising, so that users can still
        query experimental or less-documented parameters.

        Args:
            params: List of NASA POWER parameter codes to validate.
            start: Start date in any form accepted by :func:`parse_date_strict`.
            end: End date in any form accepted by :func:`parse_date_strict`.

        Raises:
            ValueError: If *start* is later than *end*.
            AmbiguousDateError: If either date is a slash-separated string.

        Warns:
            UserWarning: If any code in *params* is not in the known parameter
                catalogue.
        """
        # Validate parameters
        known_params = set(cfg.params("all").keys())
        unknown = [p for p in params if p not in known_params]
        if unknown:
            warnings.warn(
                f"Unknown parameter(s): {', '.join(unknown)}. These might not be supported by NASA POWER.",
                UserWarning,
                stacklevel=3,
            )

        # Validate date range
        start_dt = parse_date_strict(start)
        end_dt = parse_date_strict(end)
        if start_dt > end_dt:
            raise ValueError("start date must be before or equal to end date")

    def _read_from_cache_db(self, key: str) -> pd.DataFrame | None:
        """Load and decompress a cached DataFrame for *key*, or return ``None`` if missing."""
        if not self.db_conn:
            return None

        with self.db_lock:
            try:
                with self.db_conn:
                    cur = self.db_conn.execute(
                        "SELECT timestamp, data FROM cache WHERE key=?", (key,)
                    )
                    row = cur.fetchone()
            except sqlite3.Error as e:
                logger.warning(
                    "Failed to read from cache database for key %s: %s", key, e
                )
                return None

        if row:
            _, compressed_data = row
            try:
                self._metrics["cache_initial_bytes"] = len(compressed_data)
                raw_data = json.loads(gzip.decompress(compressed_data))
                cached_df = _response_to_dataframe(raw_data, self.temporal_api)

                if cached_df.empty:
                    return None

                return cached_df
            except (json.JSONDecodeError, gzip.BadGzipFile) as e:
                logger.warning(
                    "Could not decode or decompress cache data for key %s: %s", key, e
                )

        return None

    def _write_to_cache_db(self, key: str, data: dict[str, Any]) -> None:
        """Compress *data* and upsert it into the SQLite cache under *key*."""
        if not self.db_conn:
            return

        with self.db_lock:
            try:
                compressed_data = gzip.compress(json.dumps(data).encode("utf-8"))
                self._metrics["cache_final_bytes"] = len(compressed_data)
                timestamp = datetime.now().isoformat()

                with self.db_conn:
                    self.db_conn.execute(
                        "INSERT OR REPLACE INTO cache (key, timestamp, data) VALUES (?, ?, ?)",
                        (key, timestamp, compressed_data),
                    )
            except (sqlite3.Error, TypeError) as e:
                logger.warning("Could not write to cache for key %s: %s", key, e)

    def _format_date(self, date_str: Any) -> str:
        """Format *date_str* into the ``YYYYMMDD`` string required by the API."""
        dt = parse_date_strict(date_str).to_pydatetime()
        return dt.strftime("%Y%m%d")

    def _validate_request(self, params: list[str], is_regional: bool = False) -> None:
        """Raise ``ValueError`` if *params* exceed the NASA POWER API limits for the active endpoint."""
        num_params = len(params)
        if is_regional:
            if self.temporal_api == "hourly":
                raise ValueError(
                    "Hourly temporal API does not support regional requests."
                )
            if num_params > 1:
                raise ValueError(
                    f"Regional requests support a maximum of 1 parameter. "
                    f"You requested {num_params}."
                )
        elif self.temporal_api == "hourly":
            if num_params > 15:
                raise ValueError(
                    f"Hourly point requests support a maximum of 15 parameters. "
                    f"You requested {num_params}."
                )
        elif num_params > 20:
            raise ValueError(
                f"{self.temporal_api.capitalize()} point requests support a "
                f"maximum of 20 parameters. You requested {num_params}."
            )

    def _build_point_payload(  # noqa: PLR0913
        self,
        params: list[str],
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        lon: float,
        lat: float,
        elevation: float | None = None,
        wind_elevation: float | None = None,
        wind_surface: float | None = None,
    ) -> dict[str, Any]:
        """Build the NASA POWER query parameter dict for a single-point request.

        Args:
            params: List of NASA POWER parameter codes.
            start: Inclusive start date.
            end: Inclusive end date.
            lon: Longitude in decimal degrees.
            lat: Latitude in decimal degrees.
            elevation: Optional site elevation in metres above sea level.
                Included in the payload as ``"site-elevation"`` when provided.
            wind_elevation: Optional wind elevation in metres (10–300 m).
                Included as ``"wind-elevation"`` when provided.
            wind_surface: Optional wind surface identifier string. Included as
                ``"wind-surface"`` when provided.

        Returns:
            A dict suitable for use as the ``params`` argument to
            :meth:`requests.Session.get`.

        Raises:
            ValueError: If *wind_elevation* is outside [10, 300] or if *params*
                exceeds the API parameter limit for the active endpoint.
        """
        self._validate_request(params, is_regional=False)
        payload = {
            "parameters": ",".join(params),
            "community": "AG",
            "format": "JSON",
            "start": self._format_date(start),
            "end": self._format_date(end),
            "longitude": lon,
            "latitude": lat,
        }
        if elevation is not None:
            payload["site-elevation"] = elevation
        if wind_elevation is not None:
            if not (10 <= wind_elevation <= 300):
                raise ValueError(
                    f"wind-elevation must be between 10 and 300 meters, got {wind_elevation}."
                )
            payload["wind-elevation"] = wind_elevation
        if wind_surface is not None:
            payload["wind-surface"] = wind_surface
        return payload

    def _build_regional_payload(
        self,
        params: list[str],
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> dict[str, Any]:
        """Build the payload dict for a regional bounding-box request.

        Raises:
            ValueError: If the bounding box exceeds 4.5° on either axis or min ≥ max.
        """
        self._validate_request(params, is_regional=True)

        max_bbox = self.api_limits.get("max_bbox_degrees", 4.5)
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        if lat_min >= lat_max:
            raise ValueError(
                f"lat_min ({lat_min}) must be less than lat_max ({lat_max})."
            )
        if lon_min >= lon_max:
            raise ValueError(
                f"lon_min ({lon_min}) must be less than lon_max ({lon_max})."
            )
        if lat_span > max_bbox or lon_span > max_bbox:
            raise ValueError(
                f"Bounding box too large: {lat_span:.2f}° lat × {lon_span:.2f}° lon. "
                f"NASA POWER regional API supports a maximum of "
                f"{max_bbox}° × {max_bbox}°."
            )

        payload: dict[str, Any] = {
            "parameters": ",".join(params),
            "community": "AG",
            "format": "JSON",
            "start": self._format_date(start),
            "end": self._format_date(end),
            "latitude-min": lat_min,
            "latitude-max": lat_max,
            "longitude-min": lon_min,
            "longitude-max": lon_max,
        }
        return payload

    def _fetch_and_parse_ranges(
        self,
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
        base_payload: dict[str, Any],
        url: str,
    ) -> list[pd.DataFrame]:
        """Fetch each date range in *ranges* against *base_payload* and return a list of DataFrames."""
        newly_fetched_dfs: list[pd.DataFrame] = []
        if not ranges:
            return newly_fetched_dfs

        total_bytes = 0
        start_time = time.perf_counter()
        logger.info("Fetching %d missing date range(s).", len(ranges))

        for start, end in ranges:
            payload = base_payload.copy()
            payload["start"] = self._format_date(start)
            payload["end"] = self._format_date(end)
            self.rate_limiter.acquire()
            df, b = _fetch_and_parse(self.session, url, payload, self.temporal_api)
            self._metrics["api_calls"] += 1
            if not df.empty:
                newly_fetched_dfs.append(df)
                total_bytes += b

        self._metrics["total_downloaded_bytes"] += total_bytes
        self._metrics["fetch_duration"] += time.perf_counter() - start_time
        return newly_fetched_dfs

    def _fetch_data(
        self, base_payload: dict[str, Any], url: str | None = None
    ) -> pd.DataFrame:
        """Orchestrate cache lookup, gap fetching, merge, and cache update.

        This is the central data-retrieval method called by all public fetch
        methods. When caching is enabled, it:

        1. Computes the cache key from *base_payload*.
        2. Reads any existing cached DataFrame from SQLite.
        3. Determines which date sub-ranges are missing from the cache.
        4. Fetches only the missing ranges from the API via
           :meth:`_fetch_and_parse_ranges`.
        5. Merges freshly fetched data with the cache, deduplicates, and
           writes the combined result back to SQLite.
        6. Returns the final DataFrame filtered to the requested window.

        When caching is disabled, it performs a direct single API call.

        Args:
            base_payload: The full NASA POWER query parameter dict (with
                ``"start"`` and ``"end"`` date keys).
            url: Optional override for the endpoint URL. Defaults to
                :attr:`base_url`.

        Returns:
            A :class:`~pandas.DataFrame` filtered to the requested date range.
            Returns a NaN-filled DataFrame with the correct date index and
            columns if no data was returned by the API.
        """
        self._metrics["total_requests"] += 1
        fetch_url = url or self.base_url
        use_cache = self.cache_cfg.get("enabled", False) and self.db_conn

        if not use_cache:
            start_time = time.perf_counter()
            self.rate_limiter.acquire()
            df, b = _fetch_and_parse(
                self.session, fetch_url, base_payload, self.temporal_api
            )
            self._metrics["api_calls"] += 1
            self._metrics["total_downloaded_bytes"] += b
            self._metrics["fetch_duration"] = time.perf_counter() - start_time
            return df

        cache_key = _make_cache_key(base_payload, self.temporal_api)

        def _parse_payload_date(d_str: str, is_end: bool = False) -> pd.Timestamp:
            """Parses a payload date string (YYYYMMDD or YYYYMMDDHH) to a Timestamp."""
            fmt = "%Y%m%d%H" if len(str(d_str)) == 10 else "%Y%m%d"
            ts = pd.to_datetime(str(d_str), format=fmt)
            if is_end and fmt == "%Y%m%d" and self.temporal_api == "hourly":
                ts = ts + pd.Timedelta(hours=23)
            return ts

        req_start = _parse_payload_date(base_payload["start"], is_end=False)
        req_end = _parse_payload_date(base_payload["end"], is_end=True)

        cached_df = self._read_from_cache_db(cache_key)

        ranges_to_fetch = _get_date_ranges_to_fetch(
            req_start, req_end, cached_df, self.temporal_api
        )

        if not ranges_to_fetch and cached_df is not None:
            logger.info("Retrieved full date range from cache for key %s.", cache_key)
            self._metrics["cache_hits"] += 1
            return _filter_df_by_date(cached_df, req_start, req_end)

        fetch_failed = False
        try:
            newly_fetched_dfs = self._fetch_and_parse_ranges(
                ranges_to_fetch, base_payload, fetch_url
            )
        except OSError as e:
            fetch_failed = True
            # Do not swallow client-side validation or range errors (HTTP 400, 422)
            if (
                "Error (400)" in str(e)
                or "Error (422)" in str(e)
                or "400 Client Error" in str(e)
                or "422 Client Error" in str(e)
            ):
                raise
            if cached_df is not None:
                logger.warning(
                    "API request failed: %s. Serving stale data from cache.", e
                )
                return _filter_df_by_date(cached_df, req_start, req_end)
            else:
                raise

        all_dfs = ([cached_df] if cached_df is not None else []) + newly_fetched_dfs
        combined_df = _merge_and_deduplicate(all_dfs)

        if not fetch_failed and not combined_df.empty:
            logger.info("Updating cache for key %s with merged data.", cache_key)
            cacheable_json = _convert_df_to_cacheable_json(
                combined_df, self.temporal_api
            )
            self._write_to_cache_db(cache_key, cacheable_json)

        if combined_df.empty:
            logger.warning("Fetching and merging resulted in an empty DataFrame.")
            date_range = pd.date_range(
                start=req_start,
                end=req_end,
                freq="D" if self.temporal_api == "daily" else "h",
                name="date",
            )
            params = base_payload.get("parameters", "").split(",")
            # Create a DataFrame with the correct index and columns, filled with NaNs
            if not params or not params[0]:
                return pd.DataFrame(index=date_range)
            return pd.DataFrame(np.nan, index=date_range, columns=params)

        return _filter_df_by_date(combined_df, req_start, req_end)

    def get_point_data(
        self,
        request: PointRequest | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch time-series weather data for a single geographic point.

        Accepts either a :class:`PointRequest` object or keyword arguments
        matching its fields. Delegates to
        :meth:`get_point_data_from_coordinate` after normalising the
        coordinate.

        Args:
            request: A pre-constructed :class:`PointRequest`. When provided,
                all *kwargs* are ignored.
            **kwargs: Keyword arguments forwarded to :class:`PointRequest`
                when *request* is ``None``. Must include at least ``lat``,
                ``lon``, ``start``, ``end``, and ``params``.

        Returns:
            A :class:`~pandas.DataFrame` with a :class:`~pandas.DatetimeIndex`
            named ``"date"`` and one numeric column per requested parameter.
            Missing values are :data:`pandas.NA`. Returns a NaN-filled frame
            with the correct index if the API returned no data.
        """
        if request is None:
            request = PointRequest(**kwargs)

        coord = GeoCoordinate.from_decimal(request.lat, request.lon)
        return self.get_point_data_from_coordinate(
            coord,
            request.start,
            request.end,
            request.params,
            request.elevation,
            request.wind_elevation,
            request.wind_surface,
        )

    def get_parameter_metadata(self, code: str | None = None) -> dict[str, Any]:
        """Return verified scientific metadata dictionary for *code* or all parameters if ``None``."""
        return cfg.param_metadata(params=code)

    def get_point_data_from_coordinate(  # noqa: PLR0913
        self,
        coord: GeoCoordinate,
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        params: list[str],
        elevation: float | None = None,
        wind_elevation: float | None = None,
        wind_surface: float | None = None,
        _validate: bool = True,
    ) -> pd.DataFrame:
        """Fetch time-series data for a single point given a :class:`~aidweather.geo.GeoCoordinate`.

        This is the low-level point-fetch method. All other single-point
        fetchers ultimately call this method.

        Args:
            coord: The geographic coordinate to query.
            start: Inclusive start date.
            end: Inclusive end date.
            params: List of NASA POWER parameter codes.
            elevation: Optional site elevation in metres.
            wind_elevation: Optional wind elevation in metres (10–300 m).
            wind_surface: Optional wind surface identifier.
            _validate: If ``True`` (default), validates *params* and the date
                range before fetching. Set to ``False`` in concurrent loops
                where validation has already been performed once.

        Returns:
            A :class:`~pandas.DataFrame` indexed by ``"date"`` with one
            numeric column per parameter. Missing values are
            :data:`pandas.NA`. Returns a NaN-filled frame with the correct
            index if the API returned no data.

        Raises:
            ValueError: If *params* is empty or exceeds API limits.
        """
        if _validate:
            self._validate_inputs(params, start, end)

        if not params:
            raise ValueError("No parameters provided")

        lat, lon = coord.as_decimal()
        payload = self._build_point_payload(
            params=params,
            start=start,
            end=end,
            lon=lon,
            lat=lat,
            elevation=elevation,
            wind_elevation=wind_elevation,
            wind_surface=wind_surface,
        )
        df = self._fetch_data(payload)
        if df.empty:
            req_start = parse_date_strict(start)
            req_end = parse_date_strict(end)
            date_range = pd.date_range(
                start=req_start,
                end=req_end,
                freq="D" if self.temporal_api == "daily" else "h",
                name="date",
            )
            return pd.DataFrame(np.nan, index=date_range, columns=params)

        df = _ensure_all_params_in_df(df, params)
        req_start = parse_date_strict(start)
        req_end = parse_date_strict(end)
        return _filter_df_by_date(df, req_start, req_end)

    # ------------------------------------------------------------------
    # get_multi_point_data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_points_input(
        points: Sequence | pd.DataFrame,
    ) -> list[dict]:
        """Normalise the ``points`` argument into a list of dicts with at
        least ``'lat'`` and ``'lon'`` keys."""
        if not isinstance(points, pd.DataFrame):
            return list(points)  # type: ignore[arg-type]
        # to_dict avoids iterrows() dtype coercion on mixed-type DataFrames
        # (e.g. float lat/lon + string name + float elevation).
        return points.to_dict(orient="records")

    def _submit_point_futures(
        self,
        executor: ThreadPoolExecutor,
        parsed_points: list,
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        params: list[str],
    ) -> dict:
        """Submit one :class:`~concurrent.futures.Future` per point to *executor*.

        Extracts ``lat``, ``lon``, and optional ``elevation`` /
        ``wind_elevation`` / ``wind_surface`` from each point entry (dict or
        sequence), then submits a call to
        :meth:`get_point_data_from_coordinate` for each.

        Args:
            executor: Active :class:`~concurrent.futures.ThreadPoolExecutor`.
            parsed_points: List of point dicts or ``(lat, lon[, elevation])``
                tuples.
            start: Inclusive start date for all points.
            end: Inclusive end date for all points.
            params: Parameter codes for all points.

        Returns:
            A ``{future: point}`` mapping for use with
            :meth:`_collect_futures_results`.
        """
        future_to_point: dict = {}
        for point in parsed_points:
            if isinstance(point, dict):
                lat, lon = point["lat"], point["lon"]
                elevation = point.get("elevation")
                wind_elevation = point.get("wind_elevation")
                wind_surface = point.get("wind_surface")
            else:
                lat, lon, *elev = point
                elevation = elev[0] if elev else None
                wind_elevation = None
                wind_surface = None

            future = executor.submit(
                self.get_point_data_from_coordinate,
                coord=GeoCoordinate.from_decimal(lat, lon),
                start=start,
                end=end,
                params=params,
                elevation=elevation,
                wind_elevation=wind_elevation,
                wind_surface=wind_surface,
                _validate=False,
            )
            future_to_point[future] = point
        return future_to_point

    @staticmethod
    def _collect_futures_results(
        future_to_point: dict,
    ) -> tuple[list[pd.DataFrame], list]:
        """Collect completed futures into result DataFrames and failed-point records.

        Iterates :func:`~concurrent.futures.as_completed` over
        *future_to_point*, appending successful results (with ``lat``/``lon``
        metadata columns injected) to *all_results* and recording exceptions as
        ``(point, error_message)`` pairs in *failed_points*.

        Args:
            future_to_point: ``{future: point}`` mapping produced by
                :meth:`_submit_point_futures`.

        Returns:
            A ``(all_results, failed_points)`` tuple where *all_results* is a
            list of per-point DataFrames (indexed by ``"date"``) and
            *failed_points* is a list of ``(point, error_str)`` pairs.
        """
        all_results: list[pd.DataFrame] = []
        failed_points: list = []

        for future in as_completed(future_to_point):
            point = future_to_point[future]
            try:
                df = future.result()
                if df.empty:
                    continue
                df = df.reset_index()
                if isinstance(point, dict):
                    df["lat"] = point["lat"]
                    df["lon"] = point["lon"]
                    if point.get("name") is not None:
                        df["name"] = point["name"]
                    if point.get("elevation") is not None:
                        df["elevation"] = point["elevation"]
                else:
                    df["lat"] = point[0]
                    df["lon"] = point[1]
                    if len(point) > 2:
                        df["elevation"] = point[2]
                df = df.set_index("date")
                all_results.append(df)
            except Exception as e:
                logger.warning("Failed to fetch data for point %s: %s", point, e)
                failed_points.append((point, str(e)))

        return all_results, failed_points

    def get_multi_point_data(
        self,
        points: (
            Sequence[dict[str, Any] | tuple[float, float] | tuple[float, float, float]]
            | pd.DataFrame
        ),
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        params: list[str],
        max_workers: int = 5,
        _validate: bool = True,
    ) -> tuple[
        pd.DataFrame,
        list[
            tuple[
                dict[str, Any] | tuple[float, float] | tuple[float, float, float],
                str,
            ]
        ],
    ]:
        """Fetch weather data for multiple geographic points in parallel.

        Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to issue
        concurrent calls to :meth:`get_point_data_from_coordinate`. The
        ``max_workers`` count is silently clamped to the configured
        ``api_limits.max_workers`` ceiling (default 5) to comply with NASA
        POWER API guidelines.

        Args:
            points: One of:

                - A :class:`~pandas.DataFrame` with at least ``"lat"`` and
                  ``"lon"`` columns (and optionally ``"name"``, ``"elevation"``).
                - A list of dicts with at least ``"lat"`` and ``"lon"`` keys.
                - A list of ``(lat, lon)`` or ``(lat, lon, elevation)`` tuples.

            start: Inclusive start date for all points.
            end: Inclusive end date for all points.
            params: List of NASA POWER parameter codes.
            max_workers: Maximum number of concurrent fetch threads. Clamped
                to the configured API limit (default 5).
            _validate: If ``True`` (default), validates *params* and the date
                range before dispatching.

        Returns:
            A ``(DataFrame, failed_points)`` tuple where:

            - *DataFrame* has a ``"date"`` column (reset from index), ``"lat"``
              and ``"lon"`` columns, optional ``"name"`` and ``"elevation"``
              columns, and one numeric column per parameter. An empty DataFrame
              is returned if all points failed.
            - *failed_points* is a list of ``(point, error_message)`` pairs for
              any points whose fetch raised an exception.
        """
        if _validate:
            self._validate_inputs(params, start, end)

        parsed_points = self._parse_points_input(points)

        limit = getattr(self, "max_workers_limit", 5)
        if max_workers > limit:
            logger.warning(
                f"Requested max_workers ({max_workers}) exceeds the configured limit of {limit}. "
                f"Enforcing limit of {limit} to conform with NASA POWER API guidelines."
            )
            max_workers = limit

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_point = self._submit_point_futures(
                executor, parsed_points, start, end, params
            )
            all_results, failed_points = self._collect_futures_results(future_to_point)

        if failed_points:
            logger.warning(
                "%d/%d points failed to fetch.", len(failed_points), len(parsed_points)
            )

        if not all_results:
            return pd.DataFrame(), failed_points

        return pd.concat(all_results).reset_index(), failed_points

    # ------------------------------------------------------------------
    # summarize helpers - one method per Rich table
    # ------------------------------------------------------------------

    def _build_profile_table(self, df: pd.DataFrame) -> Table:
        """Builds the Weather Data Profile Rich table."""
        table = Table(
            title="Weather Data Profile", show_header=True, header_style="bold cyan"
        )
        table.add_column("Property", style="dim")
        table.add_column("Value")
        table.add_row("Temporal Resolution", self.temporal_api.capitalize())
        if not df.empty:
            table.add_row("Start Date", str(df.index.min()))
            table.add_row("End Date", str(df.index.max()))
            table.add_row("Data Points", str(len(df)))
            table.add_row("Missing Values", f"{df.isna().sum().sum()} / {df.size}")
            table.add_row("Parameters", ", ".join(df.columns))
        return table

    def _build_perf_table(self) -> Table:
        """Builds the Transfer & Cache Performance Rich table."""
        m = self._metrics
        table = Table(
            title="Transfer & Cache Performance",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        speed = (
            (m["total_downloaded_bytes"] / 1024.0) / m["fetch_duration"]
            if m["fetch_duration"] > 0
            else 0.0
        )
        table.add_row("Network Duration", f"{m['fetch_duration']:.2f} s")
        table.add_row("Total Downloaded", _format_bytes(m["total_downloaded_bytes"]))
        table.add_row("Avg Speed", f"{speed:.2f} KiB/s")
        increm = max(0, m["cache_final_bytes"] - m["cache_initial_bytes"])
        table.add_row("Cache (Initial)", _format_bytes(m["cache_initial_bytes"]))
        table.add_row("Cache (Increment)", _format_bytes(increm))
        table.add_row("Cache (Total)", _format_bytes(m["cache_final_bytes"]))
        return table

    def _build_stats_table(self) -> Table:
        """Builds the Request Statistics Rich table."""
        m = self._metrics
        table = Table(
            title="Request Statistics",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        table.add_row("Total Logical Requests", str(m["total_requests"]))
        table.add_row("Cache Hits (Full)", str(m["cache_hits"]))
        table.add_row("Network API Calls", str(m["api_calls"]))
        hit_rate = (
            (m["cache_hits"] / m["total_requests"]) * 100
            if m["total_requests"] > 0
            else 0.0
        )
        table.add_row("Cache Hit Rate", f"{hit_rate:.1f}%")
        return table

    def _build_conn_table(self) -> Table:
        """Builds the NASA POWER Connection Info Rich table."""
        table = Table(
            title="NASA POWER Connection Info",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Property", style="dim")
        table.add_column("Value")

        table.add_row("User Agent", USER_AGENT.split(" (", maxsplit=1)[0])
        table.add_row("Base URL", self.base_url.split("/temporal")[0])
        return table

    def summarize(self, df: pd.DataFrame) -> None:
        """Print a Rich summary panel with data profile, transfer metrics, and request statistics.

        Renders four :class:`~rich.panel.Panel` blocks to the console:

        - **Weather Data Profile**: temporal resolution, date range, row count,
          missing-value count, and parameter list.
        - **Transfer & Cache Performance**: network duration, bytes downloaded,
          average speed, and cache size delta.
        - **Request Statistics**: total logical requests, cache hits, network
          API calls, and cache hit rate.
        - **NASA POWER Connection Info**: user-agent string and base URL.

        Args:
            df: The result DataFrame returned by any ``get_*_data`` method.
                Used to populate the data profile section.
        """
        console = Console()
        console.print(Panel(self._build_profile_table(df), subtitle="Data Insight"))
        console.print(Panel(self._build_perf_table(), subtitle="Performance"))
        console.print(Panel(self._build_stats_table(), subtitle="Efficiency"))
        console.print(Panel(self._build_conn_table(), subtitle="API Connection"))

    def _fetch_regional_data(
        self,
        payload: dict[str, Any],
    ) -> pd.DataFrame:
        """Fetch data from the regional endpoint and parse the GeoJSON FeatureCollection response."""
        self._metrics["total_requests"] += 1
        start_time = time.perf_counter()

        self.rate_limiter.acquire()

        try:
            resp = self.session.get(self.regional_base_url, params=payload)
            if resp.status_code in (400, 422):
                try:
                    err_data = resp.json()
                    if isinstance(err_data, dict) and "messages" in err_data:
                        err_msg = "; ".join(err_data["messages"])
                        raise APIRequestError(
                            f"NASA POWER API Error ({resp.status_code}): {err_msg}"
                        )
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
            resp.raise_for_status()
            byte_count = len(resp.content)
            data = _parse_json_response(resp)
        except requests.exceptions.RequestException as e:
            resp_obj = getattr(e, "response", None)
            if resp_obj is not None and getattr(resp_obj, "status_code", None) == 429:
                logger.error(
                    "Rate limit exceeded (HTTP 429). Please slow down requests or use an API key."
                )
            logger.error(
                "Regional API request failed for payload %s: %s",
                _safe_payload_repr(payload),
                e,
            )
            raise APIRequestError(f"Regional API request failed: {e}") from e

        self._metrics["api_calls"] += 1
        self._metrics["total_downloaded_bytes"] += byte_count
        self._metrics["fetch_duration"] += time.perf_counter() - start_time

        if "error" in data:
            logger.error(
                "Regional API Error for payload %s: %s",
                _safe_payload_repr(payload),
                data.get("error"),
            )
            return pd.DataFrame()

        return _regional_response_to_dataframe(data)

    def get_regional_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        params: list[str],
        request: RegionalRequest | None = None,
        _validate: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily weather data for a geographic bounding box.

        Returns a long-form DataFrame with one row per (date, grid cell) pair.
        When *request* is provided it takes full precedence over the positional
        coordinate arguments (a :class:`UserWarning` is emitted to make the
        precedence explicit).

        Args:
            lat_min: Southern boundary of the bounding box (latitude).
            lat_max: Northern boundary of the bounding box (latitude).
            lon_min: Western boundary of the bounding box (longitude).
            lon_max: Eastern boundary of the bounding box (longitude).
            start: Inclusive start date.
            end: Inclusive end date.
            params: List with exactly one NASA POWER parameter code (the
                regional API supports only one parameter per request).
            request: Optional :class:`RegionalRequest` that supersedes all
                positional arguments when provided.
            _validate: If ``True`` (default), validates *params* and dates
                before fetching.

        Returns:
            A long-form :class:`~pandas.DataFrame` indexed by ``"date"`` with
            columns ``"lat"``, ``"lon"``, optionally ``"elevation"``, and one
            numeric column for the requested parameter. Missing values are
            :data:`pandas.NA`.

        Raises:
            ValueError: If the bounding box exceeds 4.5° on either axis,
                if ``lat_min >= lat_max`` or ``lon_min >= lon_max``, or if
                more than one parameter is requested.
        """
        if request is not None:
            warnings.warn(
                "Both positional coordinate arguments and 'request' were supplied. "
                "The 'request' object takes precedence; all positional arguments "
                "(lat_min, lat_max, lon_min, lon_max, start, end, params) are ignored.",
                UserWarning,
                stacklevel=2,
            )
            lat_min = request.lat_min
            lat_max = request.lat_max
            lon_min = request.lon_min
            lon_max = request.lon_max
            start = request.start
            end = request.end
            params = request.params

        if _validate:
            self._validate_inputs(params, start, end)

        payload = self._build_regional_payload(
            params=params,
            start=start,
            end=end,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        return self._fetch_regional_data(payload)

    def get_regional_data_from_coordinates(
        self,
        coord_sw: GeoCoordinate,
        coord_ne: GeoCoordinate,
        start: str,
        end: str,
        params: list[str],
    ) -> pd.DataFrame:
        """Fetch regional data using two corner :class:`~aidweather.geo.GeoCoordinate` objects.

        Convenience wrapper over :meth:`get_regional_data` that accepts
        South-West and North-East corner coordinates instead of four separate
        float bounds.

        Args:
            coord_sw: South-West corner of the bounding box.
            coord_ne: North-East corner of the bounding box.
            start: Inclusive start date string.
            end: Inclusive end date string.
            params: List with exactly one NASA POWER parameter code.

        Returns:
            See :meth:`get_regional_data`.
        """
        lat_sw, lon_sw = coord_sw.as_decimal()
        lat_ne, lon_ne = coord_ne.as_decimal()
        return self.get_regional_data(
            lat_min=lat_sw,
            lat_max=lat_ne,
            lon_min=lon_sw,
            lon_max=lon_ne,
            start=start,
            end=end,
            params=params,
        )

    # ------------------------------------------------------------------
    # Transect helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_transect_num_points(
        start_coord: GeoCoordinate,
        end_coord: GeoCoordinate,
        num_points: int | None,
        spacing_km: float | None,
        params: list[str] | None = None,
    ) -> int:
        """Determine the number of sample points for a transect request.

        Resolves the sampling density from *num_points* or *spacing_km*,
        then clamps the result to the minimum effective spacing derived from
        the native grid resolution of the requested parameters. This prevents
        sub-resolution requests that would return duplicate data.

        Clamping logic: the minimum effective spacing equals the latitude step
        of the finest native grid among the requested parameters (MERRA-2:
        0.5° ≈ 55.5 km; CERES: 1.0° ≈ 111.1 km). If the derived spacing is
        finer than the minimum, ``num_points`` is silently clamped and a
        diagnostic log message is emitted.

        Args:
            start_coord: Starting endpoint of the transect.
            end_coord: Ending endpoint of the transect.
            num_points: Explicit number of sample points. Takes priority over
                *spacing_km* when both are provided.
            spacing_km: Approximate distance between samples in kilometres.
                Used to compute *num_points* when that argument is ``None``.
            params: Optional list of parameter codes used to derive the minimum
                grid spacing. Defaults to MERRA-2 resolution when ``None``.

        Returns:
            The resolved (and possibly clamped) number of sample points,
            always at least ``2``.

        Raises:
            ValueError: If neither *num_points* nor *spacing_km* is provided,
                or if *spacing_km* is not positive.
        """
        # Great-circle distance approximation between the two endpoints
        lat1, lon1 = start_coord.as_decimal()
        lat2, lon2 = end_coord.as_decimal()
        dlat_km = (lat2 - lat1) * 111.1
        mid_lat_rad = np.deg2rad((lat1 + lat2) / 2)
        dlon_km = (lon2 - lon1) * 111.32 * np.cos(mid_lat_rad)
        total_km = float(np.hypot(dlat_km, dlon_km))

        # Minimum spacing derived from requested parameter native grid (latitude step)
        min_lat_deg = 0.5
        if params:
            grid_res_list = [cfg.get_native_grid(p)[0] for p in params]
            if grid_res_list:
                min_lat_deg = min(grid_res_list)

        min_spacing_km = min_lat_deg * 111.1

        if num_points is not None and spacing_km is not None:
            effective_spacing = total_km / max(num_points - 1, 1)
            logger.info(
                "Both num_points=%d and spacing_km=%.1f were supplied. "
                "num_points takes priority (effective spacing ≈ %.1f km).",
                num_points,
                spacing_km,
                effective_spacing,
            )
        elif num_points is not None:
            pass  # use as-is; validate below
        elif spacing_km is not None:
            if spacing_km <= 0:
                raise ValueError("spacing_km must be positive.")
            num_points = max(2, int(round(total_km / spacing_km)) + 1)
            logger.info(
                "Derived num_points=%d from spacing_km=%.1f over %.1f km transect.",
                num_points,
                spacing_km,
                total_km,
            )
        else:
            raise ValueError(
                "TransectRequest requires either 'num_points' or 'spacing_km'."
            )

        # Clamp to avoid sub-resolution sampling
        if num_points > 1:
            effective_spacing = total_km / (num_points - 1)
            if effective_spacing < min_spacing_km:
                max_allowed = max(2, int(total_km / min_spacing_km) + 1)
                logger.info(
                    "Requested num_points=%d gives %.1f km spacing, below the "
                    "parameter native spatial resolution (%.2f° lat, ~%.1f km). "
                    "Clamping to %d points.",
                    num_points,
                    effective_spacing,
                    min_lat_deg,
                    min_spacing_km,
                    max_allowed,
                )
                num_points = max_allowed

        return max(2, num_points)

    def get_transect_data(
        self,
        request: TransectRequest | None = None,
        _validate: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Sample points along a 1D transect and fetch data for each in parallel.

        Resolves the sample points via :meth:`_resolve_transect_num_points`
        (which enforces minimum grid-resolution spacing), then delegates to
        :meth:`get_multi_point_data`.

        Args:
            request: A pre-constructed :class:`TransectRequest`. When
                provided, all *kwargs* are ignored.
            _validate: If ``True`` (default), validates params and date range
                before dispatching.
            **kwargs: Forwarded to :class:`TransectRequest` when *request* is
                ``None``. Must include at least ``start_coord``, ``end_coord``,
                ``start``, ``end``, ``params``, and either ``num_points`` or
                ``spacing_km``.

        Returns:
            A :class:`~pandas.DataFrame` with columns ``"date"``, ``"lat"``,
            ``"lon"``, ``"name"`` (``"Point_1"``, ``"Point_2"``, …), and one
            numeric column per parameter. Indexed by row number (not date).

        Raises:
            ValueError: If neither ``num_points`` nor ``spacing_km`` is given.
            OSError: If fetching data for **all** generated transect points
                fails (partial failures are silently dropped).
        """
        if request is None:
            request = TransectRequest(**kwargs)

        if _validate:
            self._validate_inputs(request.params, request.start, request.end)

        n = self._resolve_transect_num_points(
            request.start_coord,
            request.end_coord,
            request.num_points,
            request.spacing_km,
            params=request.params,
        )

        lat1, lon1 = request.start_coord.as_decimal()
        lat2, lon2 = request.end_coord.as_decimal()
        lats = np.linspace(lat1, lat2, n)
        lons = np.linspace(lon1, lon2, n)

        max_workers = request.max_workers
        limit = getattr(self, "max_workers_limit", 5)
        if max_workers > limit:
            logger.warning(
                "Requested max_workers (%d) exceeds the configured limit of %d. "
                "Enforcing limit of %d.",
                max_workers,
                limit,
                limit,
            )
            max_workers = limit

        logger.info(
            "Generated %d transect points from %s to %s.",
            n,
            request.start_coord,
            request.end_coord,
        )

        points_with_metadata: list[dict[str, Any]] = [
            {"lat": round(p_lat, 4), "lon": round(p_lon, 4), "name": f"Point_{i + 1}"}
            for i, (p_lat, p_lon) in enumerate(zip(lats, lons, strict=True))
        ]

        df, failed_points = self.get_multi_point_data(
            points=points_with_metadata,
            start=request.start,
            end=request.end,
            params=request.params,
            max_workers=max_workers,
            _validate=False,
        )

        if df.empty and failed_points:
            sample_errors = ", ".join(
                dict.fromkeys(err for _, err in failed_points[:3])
            )
            raise OSError(
                f"Failed to fetch data for all {len(failed_points)} transect points: "
                f"{sample_errors}"
            )

        return df

    def get_transect_data_from_coordinates(
        self,
        coord_a: GeoCoordinate,
        coord_b: GeoCoordinate,
        start: str,
        end: str,
        params: list[str],
        num_points: int | None = None,
        spacing_km: float | None = None,
        max_workers: int = 5,
    ) -> pd.DataFrame:
        """Fetch transect data using two :class:`~aidweather.geo.GeoCoordinate` endpoints.

        Convenience wrapper over :meth:`get_transect_data` for callers that
        already hold :class:`~aidweather.geo.GeoCoordinate` objects and prefer
        not to construct a :class:`TransectRequest` manually.

        Args:
            coord_a: Starting endpoint of the transect.
            coord_b: Ending endpoint of the transect.
            start: Inclusive start date string.
            end: Inclusive end date string.
            params: List of NASA POWER parameter codes.
            num_points: Explicit number of sample points. Takes priority over
                *spacing_km* when both are provided.
            spacing_km: Approximate spacing between samples in kilometres.
            max_workers: Maximum concurrent fetch threads (clamped to the
                configured API limit).

        Returns:
            See :meth:`get_transect_data`.
        """
        return self.get_transect_data(
            TransectRequest(
                start_coord=coord_a,
                end_coord=coord_b,
                start=start,
                end=end,
                params=params,
                num_points=num_points,
                spacing_km=spacing_km,
                max_workers=max_workers,
            )
        )

    def __repr__(self) -> str:
        return f"<PowerClient(api='{self.temporal_api}', url='{self.base_url}')>"

    def __del__(self) -> None:
        """Close the SQLite cache connection on garbage collection."""
        conn = getattr(self, "db_conn", None)
        if conn is not None:
            conn.close()
