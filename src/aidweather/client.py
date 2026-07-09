# SPDX-License-Identifier: Apache-2.0
"""
aidweather.client
~~~~~~~~~~~~~~~~~

NASA POWER API client with SQLite caching and retry logic.

Exposes ``PowerClient`` for fetching daily and hourly meteorological data from
the POWER point and regional endpoints. Caches responses locally by request
hash to avoid redundant network calls.
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
    """Thread-safe sliding-window rate limiter."""

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
    """Return a ``requests.Session`` with a ``Retry`` adapter mounted on both http and https."""
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


def parse_date_strict(date_value: Any) -> pd.Timestamp:
    """Parses a date, rejecting slash-separated strings (e.g. "05/03/2023")
    since NASA POWER's day-first users and pandas' month-first default would
    silently disagree on which is the day and which is the month.
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
    """Return a deterministic SHA-256 hex digest for *payload*, excluding date keys."""
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
    """Return the date ranges missing from *cached_df* relative to the requested range."""
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
    """Perform a single GET request and return a ``(DataFrame, byte_count)`` tuple."""
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
    """Parse the raw POWER JSON response into a DatetimeIndex DataFrame with fill-value NAs."""
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
    """Add any missing *params* columns to *df*, filled with ``pd.NA``."""
    for param in params:
        if param not in df.columns:
            df[param] = pd.NA
    return df[params]


def _regional_response_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Parse a POWER regional GeoJSON FeatureCollection into a long-form DataFrame."""
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
    """NASA POWER API client with SQLite caching, retry logic, and rate limiting."""

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
        """Initialise the client, configure caching, rate limiter, and session.

        Args:
            temporal_api: ``"daily"`` or ``"hourly"``.
            session: Optional pre-configured session; a retry session is created if omitted.

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
        """Validate *params* list and date range; warn on unknown parameter codes."""
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
                logger.warning(f"Could not write to cache for key {key}: {e}")

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
        """Build and return the query payload dict for a single-point API request."""
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
        """Orchestrate cache lookup, gap fetching, merge, and cache update for *base_payload*."""
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
        """Fetch time-series data for a single geographic point."""
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
        """Fetch time-series data for a single point given a ``GeoCoordinate``."""
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
        """Fetch data for multiple geographic points in parallel.

        Returns a ``(DataFrame, failed)`` tuple where *failed* is a list of
        ``(point, error_message)`` pairs for any points whose fetch raised.
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
        """Print a Rich summary panel with data profile, transfer metrics, and request statistics."""
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
        """Fetch data for a geographic bounding box via the regional API.

        Raises:
            ValueError: If the box exceeds 4.5° on either axis or more than one parameter
                is requested.
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
        """Fetch regional data using two corner ``GeoCoordinate`` objects (SW and NE)."""
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
        """Resolve the number of transect sample points from *num_points* or *spacing_km*.

        Raises:
            ValueError: If neither *num_points* nor *spacing_km* is provided.
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
        """Sample *n* points along a transect and fetch data for each in parallel.

        Raises:
            ValueError: If neither ``num_points`` nor ``spacing_km`` is given.
            OSError: If fetching data for all generated points fails.
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
        """Fetch transect data using two ``GeoCoordinate`` endpoints instead of a ``TransectRequest``."""
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
        """Close the SQLite cache connection."""
        if self.db_conn:
            self.db_conn.close()
