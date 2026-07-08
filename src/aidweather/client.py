# SPDX-License-Identifier: Apache-2.0
"""
aidweather
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides the `PowerClient`, a cache-based client for fetching
meteorological and solar energy data from NASA's POWER API.

The `PowerClient` is designed as a lightweight API wrapper to fetch, validate, and parse
daily and hourly environmental variables. It utilizes a local SQLite cache to avoid redundant
requests and a basic retry loop to manage network transient dropouts.

Please note that this client is bound by NASA POWER API service limits. Standard IP requests
without a registered key or using DEMO_KEY may be rate-limited, and requests are restricted
to a maximum of 20 daily or 15 hourly parameters. Regional bounding-box requests only
support single-parameter downloads and a maximum area of 4.5° x 4.5° at this time.

Key Features:
- Wrapper for NASA's daily and hourly API endpoints.
- Local SQLite cache to persist requested spatial-temporal ranges.
- Retry logic using backoff algorithms for transient network failures.
- Parallel request utilities using safe concurrency levels to prevent IP blocks.

Example:
    >>> from aidweather.client import PowerClient
    >>> client = PowerClient(temporal_api="daily")
    >>> weather_data = client.get_point_data(
    ...     lat=-48.82,
    ...     lon=-21.77,
    ...     start="20220101",
    ...     end="20220131",
    ...     params=["T2M", "WS2M"]
    ... )
    >>> print(weather_data.head())
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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


class RateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls: list[float] = []

    def acquire(self) -> None:
        """Blocks until a call is allowed under the rate limit."""
        if self.max_calls <= 0 or self.period <= 0:
            return

        while True:
            with self.lock:
                now = time.time()
                # Clean up calls older than the sliding window period
                self.calls = [t for t in self.calls if now - t < self.period]

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                # Sleep time calculation
                sleep_time = self.calls[0] + self.period - now

            if sleep_time > 0:
                time.sleep(sleep_time)


# --- Module-level Helpers ---

USER_AGENT = f"aidweather/{__version__} (+https://github.com/matiollipt/aidweather)"


def _session_with_retries(
    total: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Creates a `requests.Session` configured with automatic retries on failures.

    This helper function sets up a session with a `Retry` strategy, making HTTP
    requests more resilient to transient network issues or temporary server errors.
    It uses an exponential backoff algorithm between retries.

    Args:
        total: The total number of retries to attempt.
        backoff_factor: A factor to calculate the delay between retries.
        status_forcelist: A set of HTTP status codes that should trigger a retry.

    Returns:
        A `requests.Session` object with the retry mechanism mounted.
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
    """Raised when a date string's day/month order cannot be inferred safely."""


def parse_date_strict(date_value: Any) -> pd.Timestamp:
    """Parses a date, rejecting slash-separated strings (e.g. "05/03/2023")
    since NASA POWER's day-first users and pandas' month-first default would
    silently disagree on which is the day and which is the month.
    """
    if isinstance(date_value, str) and _AMBIGUOUS_SLASH_DATE_RE.match(date_value.strip()):
        raise AmbiguousDateError(
            f"Ambiguous date '{date_value}': day/month order is not clear from "
            "a slash-separated date. Use an unambiguous format instead, e.g. YYYY-MM-DD."
        )
    return pd.to_datetime(date_value)


def _make_cache_key(payload: dict[str, Any], temporal_api: str = "daily") -> str:
    """Creates a deterministic SHA-256 hash from a request payload dictionary.

    This hash is used as a key for caching. The 'start' and 'end' keys are
    removed from the payload before hashing to ensure that requests for the same
    location and parameters but different time ranges can share a cache entry.

    Args:
        payload: The request payload dictionary.
        temporal_api: The temporal resolution to isolate daily from hourly caches.

    Returns:
        The SHA-256 hash string.
    """
    key_payload = payload.copy()
    key_payload.pop("start", None)
    key_payload.pop("end", None)
    key_payload["_temporal_api"] = temporal_api
    encoded = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    """Determines the date ranges that are missing from the cache.

    Compares the requested date range against the range available in the cached
    DataFrame and returns a list of date range tuples that need to be fetched.

    Args:
        requested_start: The start of the requested date range.
        requested_end: The end of the requested date range.
        cached_df: The DataFrame from the cache, indexed by date.
        temporal_api: The temporal resolution ("daily" or "hourly"), used to
            determine the correct time delta.

    Returns:
        A list of (start, end) tuples representing the date ranges to be fetched.
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
    """Converts a DataFrame to the JSON-like dictionary format for caching.

    This function reverses the parsing process, transforming a DataFrame back into
    the nested dictionary structure similar to the original NASA POWER API response.
    This allows fetched data to be stored in the cache in its native format.

    Args:
        df: The DataFrame to convert. Must have a DatetimeIndex.
        temporal_api: The temporal resolution ("daily" or "hourly").

    Returns:
        A dictionary representing the data in a cacheable format.
    """
    df_copy = df.copy()
    date_format = "%Y%m%d%H" if temporal_api == "hourly" else "%Y%m%d"
    df_copy.index = df_copy.index.strftime(date_format)
    df_copy = df_copy.fillna(-999)
    param_dict = df_copy.to_dict(orient="dict")
    return {"properties": {"parameter": param_dict}}


def _fetch_and_parse(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    temporal_api: Literal["daily", "hourly"],
) -> tuple[pd.DataFrame, int]:
    """Performs a single API request and parses the JSON response into a DataFrame.

    Returns:
        A tuple of (DataFrame, byte_count).
    """
    try:
        resp = session.get(url, params=payload)

        resp.raise_for_status()
        byte_count = len(resp.content)
        data = _parse_json_response(resp)

        if "error" in data:
            logger.error(f"API Error for payload {payload}: {data.get('error')}")
            return pd.DataFrame(), 0

        return _response_to_dataframe(data, temporal_api), byte_count

    except requests.exceptions.RequestException as e:
        resp_obj = getattr(e, "response", None)
        if resp_obj is not None and getattr(resp_obj, "status_code", None) == 429:
            logger.error("Rate limit exceeded (HTTP 429). Please slow down requests.")
        logger.error(f"API request failed for payload {payload}: {e}")
        raise OSError(f"API request failed: {e}") from e


def _merge_and_deduplicate(
    df_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenates, de-duplicates, and sorts a list of DataFrames.

    The DataFrames are combined, and any duplicate index entries are removed,
    keeping the first occurrence. The final DataFrame is sorted by its index.

    Args:
        df_list: A list of DataFrames to merge.

    Returns:
        A single, merged, and sorted DataFrame.
    """
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
    """Filters a DataFrame to a specific inclusive date range.

    Args:
        df: The DataFrame to filter, which must have a DatetimeIndex.
        start: The start date of the filtering range.
        end: The end date of the filtering range.

    Returns:
        The filtered DataFrame.
    """
    if df.empty:
        return df
    return df.loc[start:end]


# --- Parsing Helpers ---


def _parse_json_response(resp: requests.Response) -> dict[str, Any]:
    """Parses JSON from a `requests.Response` object.

    Args:
        resp: The HTTP response object.

    Returns:
        The parsed JSON data as a dictionary.

    Raises:
        ValueError: If the response body cannot be decoded as JSON.
    """
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
    """Parses the raw JSON response from the POWER API into a pandas DataFrame.

    This function extracts the parameter data, formats the date index, and
    converts fill values (-999) to `pd.NA`.

    Args:
        data: The parsed JSON data from the API response.
        temporal_api: The temporal API from which the data was fetched.

    Returns:
        A cleaned DataFrame with a DatetimeIndex and numeric columns.
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
        logger.warning(
            "Unrecognised date key '%s'. Attempting mixed format.", sample_key
        )
        date_format = "mixed"

    try:
        df["date"] = pd.to_datetime(df.index, format=date_format)
    except Exception:
        logger.error(
            "Failed to parse date index with format '%s'. Returning empty.", date_format
        )
        return pd.DataFrame()

    df = df.reset_index(drop=True).set_index("date")
    df = df.replace(-999, pd.NA)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_index()


def _ensure_all_params_in_df(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """Ensures a DataFrame contains all requested parameter columns.

    If a parameter column is missing (e.g., because the API did not return it),
    it is added to the DataFrame and filled with `pd.NA`.

    Args:
        df: The DataFrame to check.
        params: A list of parameter names that should be present.

    Returns:
        The DataFrame with all required columns.
    """
    for param in params:
        if param not in df.columns:
            df[param] = pd.NA
    return df[params]


def _regional_response_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Parses a GeoJSON FeatureCollection from the regional API into a DataFrame.

    The NASA POWER regional endpoint returns a GeoJSON response where each
    Feature represents a 0.5° grid cell with time-series data. This function
    converts the nested structure into a flat, long-form DataFrame.

    Args:
        data: The parsed JSON data from the regional API response.

    Returns:
        A DataFrame with columns: ``date``, ``lat``, ``lon``, ``elevation``,
        and one column per parameter. Indexed by ``date``.
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

__all__ = ["PowerClient"]


class PowerClient:
    """A client for the NASA POWER API with caching and retry mechanisms.

    Attributes:
        temporal_api: The temporal resolution of the API.
        session: The session object used for making HTTP requests.
        db_conn: The connection to the SQLite cache database, if caching is enabled.
    """
    temporal_api: Literal["daily", "hourly"]
    base_url: str
    regional_base_url: str
    params_desc: dict[str, str]
    session: requests.Session
    db_conn: sqlite3.Connection | None
    db_lock: threading.Lock
    cache_cfg: dict[str, Any]
    api_limits: dict[str, Any]
    max_workers_limit: int
    rate_limiter: RateLimiter
    _metrics: dict[str, Any]

    def __init__(
        self,
        temporal_api: Literal["daily", "hourly"] = "daily",
        session: requests.Session | None = None,
    ):
        """Initializes the PowerClient and its caching system.

        Args:
            temporal_api: The temporal API endpoint to use, either "daily" or
                "hourly".
            session: An optional `requests.Session` object to use for API calls.
                If not provided, a new session with retry logic will be created.

        Raises:
            ValueError: If `temporal_api` is not one of "daily" or "hourly".
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
        """Initializes the SQLite database connection and creates the cache table.

        Raises:
            sqlite3.Error: If the database connection or table creation fails.
        """
        db_path = ""
        try:
            cache_dir = str(self.cache_cfg.get("path", "."))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            db_path = os.path.join(cache_dir, "aidweather_cache.db")
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
            logger.error(f"Failed to initialize cache database at {db_path}: {e}")
            self.db_conn = None

    def _validate_inputs(
        self,
        params: list[str],
        start: Any,
        end: Any,
    ) -> None:
        """Validates query inputs (parameters and date ranges)."""
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
        """Loads and decompresses a response from the SQLite cache.

        Args:
            key: The cache key for the desired resource.

        Returns:
            The cached data as a DataFrame, or None if not found or on error.
        """
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
                logger.warning(f"Failed to read from cache database for key {key}: {e}")
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
                    f"Could not decode or decompress cache data for key {key}: {e}"
                )

        return None

    def _write_to_cache_db(self, key: str, data: dict[str, Any]) -> None:
        """Compresses and writes a response dictionary to the SQLite cache.

        Args:
            key: The cache key under which to store the data.
            data: The JSON-like dictionary data to cache.
        """
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
                logger.warning(f"Could not write to cache for key {key}: {e}")

    def _format_date(self, date_str: Any) -> str:
        """Formats a date into the required API string format.

        Args:
            date_str: The date to format.

        Returns:
            The formatted date string (e.g., "YYYYMMDD" or "YYYYMMDDHH").
        """
        dt = parse_date_strict(date_str).to_pydatetime()
        return dt.strftime("%Y%m%d")

    def _validate_request(self, params: list[str], is_regional: bool = False) -> None:
        """Validates NASA POWER API constraints.

        Raises:
            ValueError: If the requested parameters exceed the API's limits.
        """
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
        """Constructs the payload dictionary for a single point API request.

        Args:
            params: List of POWER parameters to request.
            start: The start date in a format parsable by `_format_date`.
            end: The end date in a format parsable by `_format_date`.
            lon: The longitude of the point.
            lat: The latitude of the point.
            elevation: The site elevation in meters.
            wind_elevation: The wind elevation in meters (10 to 300).
            wind_surface: The wind surface parameter.

        Returns:
            The constructed payload dictionary.
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
        """Constructs the payload dictionary for a regional (bounding-box) API request.

        The NASA POWER regional endpoint expects a geographic bounding box defined
        by four edge coordinates. The box must not exceed 4.5° on either axis.

        Args:
            params: List of POWER parameters to request (max 1 for regional).
            start: The start date in a format parsable by ``_format_date``.
            end: The end date in a format parsable by ``_format_date``.
            lat_min: Southern edge of the bounding box (latitude).
            lat_max: Northern edge of the bounding box (latitude).
            lon_min: Western edge of the bounding box (longitude).
            lon_max: Eastern edge of the bounding box (longitude).

        Returns:
            The constructed payload dictionary.

        Raises:
            ValueError: If the bounding box exceeds 4.5° on either axis,
                or if min >= max for latitude or longitude.
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
        """Iterates through a list of date ranges and fetches data for each.

        Args:
            ranges: A list of (start, end) tuples representing the date ranges to fetch.
            base_payload: The base request payload.
            url: The API endpoint URL.

        Returns:
            A list of DataFrames, one for each successfully fetched date range.
        """
        newly_fetched_dfs: list[pd.DataFrame] = []
        if not ranges:
            return newly_fetched_dfs

        total_bytes = 0
        start_time = time.perf_counter()
        logger.info(f"Fetching {len(ranges)} missing date range(s).")

        for start, end in ranges:
            payload = base_payload.copy()
            payload["start"] = self._format_date(start)
            payload["end"] = self._format_date(end)
            if hasattr(self, "rate_limiter") and self.rate_limiter:
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
        """Orchestrates the full data fetching and caching logic.

        Args:
            base_payload: The base payload for the API request.
            url: The specific API URL to use.

        Returns:
            A DataFrame containing the requested data.
        """
        self._metrics["total_requests"] += 1
        fetch_url = url or self.base_url
        use_cache = self.cache_cfg.get("enabled", False) and self.db_conn

        if not use_cache:
            start_time = time.perf_counter()
            if hasattr(self, "rate_limiter") and self.rate_limiter:
                self.rate_limiter.acquire()
            df, b = _fetch_and_parse(
                self.session, fetch_url, base_payload, self.temporal_api
            )
            self._metrics["api_calls"] += 1
            self._metrics["total_downloaded_bytes"] = b
            self._metrics["fetch_duration"] = time.perf_counter() - start_time
            return df

        cache_key = _make_cache_key(base_payload, self.temporal_api)

        def _parse_payload_date(d_str: str, is_end: bool = False) -> pd.Timestamp:
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
            logger.info(f"Retrieved full date range from cache for key {cache_key}.")
            self._metrics["cache_hits"] += 1
            return _filter_df_by_date(cached_df, req_start, req_end)

        fetch_failed = False
        try:
            newly_fetched_dfs = self._fetch_and_parse_ranges(
                ranges_to_fetch, base_payload, fetch_url
            )
        except OSError as e:
            fetch_failed = True
            if cached_df is not None:
                logger.warning(
                    f"API request failed: {e}. Serving stale data from cache."
                )
                return _filter_df_by_date(cached_df, req_start, req_end)
            else:
                raise

        all_dfs = ([cached_df] if cached_df is not None else []) + newly_fetched_dfs
        combined_df = _merge_and_deduplicate(all_dfs)

        if not fetch_failed and not combined_df.empty:
            logger.info(f"Updating cache for key {cache_key} with merged data.")
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
        """Fetches data for a single geographic point.

        Args:
            request: Configuration object.
            **kwargs: Configuration parameters as keyword arguments.

        Returns:
            A DataFrame containing the time-series data for the specified point.
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
        """Fetches data for a single point using a `GeoCoordinate` object.

        Args:
            coord: A `GeoCoordinate` object representing the point.
            start: The start date for the data query (e.g., "YYYYMMDD").
            end: The end date for the data query (e.g., "YYYYMMDD").
            params: A list of POWER API parameter names to fetch.
            elevation: The elevation of the site in meters.
            wind_elevation: The wind elevation in meters (10 to 300).
            wind_surface: The wind surface parameter.
            _validate: Whether to validate inputs.

        Returns:
            A DataFrame containing the time-series data, indexed by date.
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
        points: list | pd.DataFrame,
    ) -> list[dict]:
        """Normalises the `points` argument into a list of dicts with at
        least 'lat' and 'lon' keys."""
        if not isinstance(points, pd.DataFrame):
            return list(points)  # type: ignore[arg-type]

        parsed: list[dict] = []
        for _, row in points.iterrows():
            pt: dict = {"lat": float(row["lat"]), "lon": float(row["lon"])}
            for col, key in [
                ("elevation", "elevation"),
                ("name", "name"),
                ("wind_elevation", "wind_elevation"),
                ("wind_surface", "wind_surface"),
            ]:
                if col in points.columns and not pd.isna(row[col]):
                    pt[key] = float(row[col]) if col != "name" else str(row[col])
            parsed.append(pt)
        return parsed

    def _submit_point_futures(
        self,
        executor: ThreadPoolExecutor,
        parsed_points: list,
        start: datetime | str | timedelta | pd.Timestamp,
        end: datetime | str | timedelta | pd.Timestamp,
        params: list[str],
    ) -> dict:
        """Submits one Future per point and returns future→point mapping."""
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
        """Collects completed futures into result DataFrames and a list of
        ``(point, error_message)`` pairs for points whose fetch raised."""
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
                logger.warning(f"Failed to fetch data for point {point}: {e}")
                failed_points.append((point, str(e)))

        return all_results, failed_points

    def get_multi_point_data(
        self,
        points: (
            list[dict[str, Any] | tuple[float, float] | tuple[float, float, float]]
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
        """Fetches data for multiple geographic points in parallel.

        Args:
            points: A list of points, where each point is a tuple of (latitude,
                longitude) or (latitude, longitude, elevation), or a DataFrame.
            start: The start date for the data query.
            end: The end date for the data query.
            params: A list of POWER API parameters to fetch.
            max_workers: The maximum number of concurrent threads to use.
            _validate: Whether to validate inputs.

        Returns:
            A tuple containing (combined DataFrame, list of ``(point, error_message)``
            pairs for any points whose fetch failed).
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
                f"{len(failed_points)}/{len(parsed_points)} points failed to fetch."
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
        """Prints a rich summary of the retrieved weather data and transfer metrics.

        Args:
            df: The DataFrame to summarize.
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
        """Fetches data from the regional endpoint and parses the GeoJSON response.

        Unlike the point-based ``_fetch_data`` which handles caching and date-range
        merging, regional requests are always dispatched directly because the
        response format (GeoJSON FeatureCollection) is fundamentally different.

        Args:
            payload: The regional request payload.

        Returns:
            A DataFrame with ``lat``, ``lon``, ``elevation``, and parameter columns,
            indexed by date.
        """
        self._metrics["total_requests"] += 1
        start_time = time.perf_counter()

        if hasattr(self, "rate_limiter") and self.rate_limiter:
            self.rate_limiter.acquire()

        try:
            resp = self.session.get(self.regional_base_url, params=payload)
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
                f"Regional API request failed for payload {_safe_payload_repr(payload)}: {e}"
            )
            raise OSError(f"Regional API request failed: {e}") from e

        self._metrics["api_calls"] += 1
        self._metrics["total_downloaded_bytes"] += byte_count
        self._metrics["fetch_duration"] += time.perf_counter() - start_time

        if "error" in data:
            logger.error(
                f"Regional API Error for payload {_safe_payload_repr(payload)}: {data.get('error')}"
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
        """Fetches data for a geographic bounding box using the regional API.

        The NASA POWER regional endpoint returns data on a 0.5° × 0.5° grid
        within the specified bounding box. The box must not exceed 4.5° on
        either axis, and only one parameter may be requested per call.

        Args:
            lat_min: Southern edge of the bounding box (latitude).
            lat_max: Northern edge of the bounding box (latitude).
            lon_min: Western edge of the bounding box (longitude).
            lon_max: Eastern edge of the bounding box (longitude).
            start: The start date for the data query (e.g., ``"20230101"``).
            end: The end date for the data query (e.g., ``"20230131"``).
            params: A list containing **one** POWER API parameter.
            request: Optional ``RegionalRequest`` object. If provided, its
                fields override the positional arguments.
            _validate: Whether to validate inputs.

        Returns:
            A DataFrame with ``lat``, ``lon``, ``elevation`` (when available),
            and parameter columns, indexed by date.

        Raises:
            ValueError: If the bounding box is too large or more than one
                parameter is requested.
        """
        if request is not None:
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
        """Convenience helper to call the regional endpoint with two corner coordinates.

        The two ``GeoCoordinate`` objects define the south-west and north-east
        corners of the bounding box.

        Args:
            coord_sw: South-west corner of the bounding box.
            coord_ne: North-east corner of the bounding box.
            start: The start date for the data query.
            end: The end date for the data query.
            params: A list containing **one** POWER API parameter.

        Returns:
            A DataFrame containing the regional grid data.
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
    ) -> int:
        """Resolves the number of sample points for a transect.

        ``num_points`` takes priority when both arguments are supplied;
        the effective spacing is then derived and logged as INFO.

        The minimum allowed point spacing is 0.5° (~55 km) to match the
        NASA POWER grid resolution. Requesting finer spacing would return
        duplicate data; the count is clamped and an INFO message is emitted.

        Args:
            start_coord: Starting endpoint of the transect.
            end_coord: Ending endpoint of the transect.
            num_points: Explicit number of sample points, or ``None``.
            spacing_km: Approximate spacing between samples in km, or ``None``.

        Returns:
            The resolved (and possibly clamped) number of sample points.

        Raises:
            ValueError: If neither ``num_points`` nor ``spacing_km`` is provided.
        """
        # Great-circle distance approximation between the two endpoints
        lat1, lon1 = start_coord.as_decimal()
        lat2, lon2 = end_coord.as_decimal()
        dlat_km = (lat2 - lat1) * 111.1
        mid_lat_rad = np.deg2rad((lat1 + lat2) / 2)
        dlon_km = (lon2 - lon1) * 111.32 * np.cos(mid_lat_rad)
        total_km = float(np.hypot(dlat_km, dlon_km))

        # Minimum spacing the API can meaningfully resolve
        min_spacing_km = 0.5 * 111.1  # ~55.55 km per 0.5°

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
                    "Requested num_points=%d would give %.1f km spacing, "
                    "below the NASA POWER 0.5° grid resolution (~55 km). "
                    "Clamping to %d points.",
                    num_points,
                    effective_spacing,
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
        """Generates a 1D transect of points and fetches data for them in parallel.

        Points are distributed evenly along the straight-line path from
        ``start_coord`` to ``end_coord``.  Sampling density is controlled
        by ``num_points`` or ``spacing_km`` (see :class:`TransectRequest`).
        The minimum point spacing is enforced at 0.5° (~55 km) to avoid
        fetching duplicate data from the same NASA POWER grid cell.

        Args:
            request: A :class:`TransectRequest` configuration object. When
                omitted, one is constructed from the keyword arguments.
            _validate: Whether to validate inputs.
            **kwargs: Keyword arguments forwarded to :class:`TransectRequest`.

        Returns:
            A combined DataFrame indexed by date with ``lat``, ``lon``, and
            one column per requested parameter.

        Raises:
            ValueError: If neither ``num_points`` nor ``spacing_km`` is given,
                or if the sampling density cannot be resolved.
            OSError: If fetching data for *all* generated points fails.
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

        logger.info("Generated %d transect points from %s to %s.", n, request.start_coord, request.end_coord)

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
            sample_errors = ", ".join(dict.fromkeys(err for _, err in failed_points[:3]))
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
        """Convenience wrapper for :meth:`get_transect_data` using two corner coordinates.

        Mirrors the pattern of :meth:`get_regional_data_from_coordinates`.
        The two ``GeoCoordinate`` objects define the start and end endpoints
        of the transect.

        Args:
            coord_a: Starting endpoint of the transect.
            coord_b: Ending endpoint of the transect.
            start: Start date for the data query (e.g. ``"20230101"``).
            end: End date for the data query (e.g. ``"20230131"``).
            params: List of POWER API parameter names to fetch.
            num_points: Number of sample points (takes priority over
                ``spacing_km`` when both are supplied).
            spacing_km: Approximate spacing between samples in km.
            max_workers: Thread-pool size for concurrent requests.

        Returns:
            A combined DataFrame indexed by date with ``lat``, ``lon``, and
            one column per requested parameter.
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
        """Ensures the database connection is closed when the client is destroyed."""
        if self.db_conn:
            self.db_conn.close()
