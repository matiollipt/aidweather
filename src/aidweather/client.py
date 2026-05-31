# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
aidweather
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides the `PowerClient`, a versatile, cache-based client for fetching
meteorological and solar energy data from NASA's POWER API.

The `PowerClient` is designed as a lightweight API wrapper to fetch, validate, and parse
daily and hourly environmental variables. It utilizes a local SQLite cache to avoid redundant
requests and a basic retry loop to manage network transient dropouts.

Please note that this client is bound by NASA POWER API service limits. Standard IP requests
without a registered key or using DEMO_KEY may be rate-limited, and requests are restricted
to a maximum of 20 daily or 15 hourly parameters. Regional area requests only support
single-parameter downloads at this time.

Key Features:
- Standard wrapper for NASA's daily and hourly API endpoints.
- Basic local SQLite cache to persist requested spatial-temporal ranges.
- Basic retry logic using backoff algorithms for transient network failures.
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

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
from aidweather.config import cfg
from aidweather.geo import GeoCoordinate
from pydantic import BaseModel, ConfigDict
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from urllib3.util.retry import Retry

from aidweather import __version__

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


def _load_env_file(filename: str = ".env") -> None:
    """Parses a .env file from the current directory and sets environment variables.

    This function looks for a file named `.env` in the current working directory.
    If found, it reads each line, splits it into key-value pairs, and adds them
    to `os.environ`. Lines starting with '#' or empty lines are ignored.

    Args:
        filename: The name of the environment file to load.
    """
    env_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, encoding="utf-8") as f:
            for raw_line in f:
                clean_line = raw_line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                if "=" in clean_line:
                    key, value = clean_line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and value:
                        if key not in os.environ:
                            os.environ[key] = value
                            logger.debug(f"Loaded {key} from {filename}")
    except Exception as e:
        logger.warning(f"Failed to load {filename} file: {e}")


def _make_cache_key(payload: dict[str, Any], temporal_api: str = "daily") -> str:
    """Creates a deterministic SHA-256 hash from a request payload dictionary.

    This hash is used as a key for caching. The 'start', 'end', and 'api_key' keys are
    removed from the payload before hashing to ensure that requests for the same
    location and parameters but different time ranges/keys can share a cache entry.

    Args:
        payload: The request payload dictionary.
        temporal_api: The temporal resolution to isolate daily from hourly caches.

    Returns:
        The SHA-256 hash string.
    """
    key_payload = payload.copy()
    key_payload.pop("start", None)
    key_payload.pop("end", None)
    key_payload.pop("api_key", None)
    key_payload["_temporal_api"] = temporal_api
    encoded = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_payload_repr(payload: dict[str, Any]) -> str:
    """Returns a string representation of the payload with secrets redacted."""
    safe_payload = payload.copy()
    if "api_key" in safe_payload:
        key = safe_payload["api_key"]
        if key == "DEMO_KEY":
            pass
        elif isinstance(key, str) and len(key) > 6:
            safe_payload["api_key"] = f"***{key[-6:]}"
        else:
            safe_payload["api_key"] = "***"
    return str(safe_payload)


def _format_bytes(size: float) -> str:
    """Formats a byte count into a human-readable string (e.g., KiB, MiB)."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


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
    temporal_api: str,
) -> tuple[pd.DataFrame, int, dict[str, str]]:
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
            logger.error(
                f"API Error for payload {_safe_payload_repr(payload)}: {data.get('error')}"
            )
            return pd.DataFrame(), 0

        return _response_to_dataframe(data, temporal_api), byte_count

    except requests.exceptions.RequestException as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 429:
            logger.error(
                "Rate limit exceeded (HTTP 429). Please slow down requests or use an API key."
            )
        logger.error(
            f"API request failed for payload {_safe_payload_repr(payload)}: {e}"
        )
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


class ExpandedPointRequest(PointRequest):
    axis: Literal["lat", "lon"] = "lat"
    distance_km: float = 10.0
    num_points: int = 10
    max_workers: int = 8


# --- Main Client Class ---

__all__ = ["PowerClient"]


class PowerClient:
    """A client for the NASA POWER API with caching and retry mechanisms.

    Attributes:
        temporal_api: The temporal resolution of the API.
        session: The session object used for making HTTP requests.
        db_conn: The connection to the SQLite cache database, if caching is enabled.
    """

    def __init__(
        self,
        temporal_api: Literal["daily", "hourly"] = "daily",
        api_key: str | None = None,
        session: requests.Session | None = None,
    ):
        """Initializes the PowerClient and its caching system.

        Args:
            temporal_api: The temporal API endpoint to use, either "daily" or
                "hourly".
            api_key: An optional NASA POWER API key. If not provided, the client
                will attempt to load it from a `.env` file in the current working
                directory or from the `NASA_POWER_API_KEY` environment variable.
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

        # Authentication / API Key Setup — log state, never print to console
        _load_env_file()
        self.api_key = api_key or os.environ.get("NASA_POWER_API_KEY")
        if self.api_key == "DEMO_KEY":
            logger.info("Using NASA POWER DEMO_KEY. Rate limits will be very strict.")
        elif self.api_key:
            masked_key = f"***{self.api_key[-4:]}" if len(self.api_key) > 4 else "***"
            logger.info("NASA POWER API key configured (%s).", masked_key)
        else:
            logger.info("No NASA POWER API key provided. Using IP-based limits.")

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
                self.db_conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        data BLOB NOT NULL
                    )
                """)
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize cache database at {db_path}: {e}")
            self.db_conn = None

    def _read_from_cache_db(self, key: str) -> pd.DataFrame | None:
        """Loads and decompresses a response from the SQLite cache.

        Args:
            key: The cache key for the desired resource.

        Returns:
            The cached data as a DataFrame, or None if not found or on error.
        """
        if not self.db_conn:
            return None

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

    def _format_date(self, date_str: str | datetime) -> str:
        """Formats a date into the required API string format.

        Args:
            date_str: The date to format.

        Returns:
            The formatted date string (e.g., "YYYYMMDD" or "YYYYMMDDHH").
        """
        dt = pd.to_datetime(date_str).to_pydatetime()
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
        start: str,
        end: str,
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
        if self.api_key:
            payload["api_key"] = self.api_key
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
        start: str,
        end: str,
        lon_lat_list: list[tuple[float, float]],
    ) -> dict[str, Any]:
        """Constructs the payload dictionary for a regional API request.

        Args:
            params: List of POWER parameters to request.
            start: The start date in a format parsable by `_format_date`.
            end: The end date in a format parsable by `_format_date`.
            lon_lat_list: A list of (longitude, latitude) tuples.

        Returns:
            The constructed payload dictionary.
        """
        self._validate_request(params, is_regional=True)
        payload = {
            "parameters": ",".join(params),
            "community": "AG",
            "format": "JSON",
            "start": self._format_date(start),
            "end": self._format_date(end),
            "lonlat": ";".join(f"{lon},{lat}" for lon, lat in lon_lat_list),
        }
        if self.api_key:
            payload["api_key"] = self.api_key
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
        newly_fetched_dfs = []
        if not ranges:
            return newly_fetched_dfs

        total_bytes = 0
        start_time = time.perf_counter()
        logger.info(f"Fetching {len(ranges)} missing date range(s).")

        for start, end in ranges:
            payload = base_payload.copy()
            payload["start"] = self._format_date(start)
            payload["end"] = self._format_date(end)
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
            df, b = _fetch_and_parse(
                self.session, fetch_url, base_payload, self.temporal_api
            )
            self._metrics["api_calls"] += 1
            self._metrics["total_downloaded_bytes"] = b
            self._metrics["fetch_duration"] = time.perf_counter() - start_time
            return df

        cache_key = _make_cache_key(base_payload, self.temporal_api)

        def _parse_payload_date(d_str: str) -> pd.Timestamp:
            fmt = "%Y%m%d%H" if len(str(d_str)) == 10 else "%Y%m%d"
            return pd.to_datetime(str(d_str), format=fmt)

        req_start = _parse_payload_date(base_payload["start"])
        req_end = _parse_payload_date(base_payload["end"])

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
        start: str,
        end: str,
        params: list[str],
        elevation: float | None = None,
        wind_elevation: float | None = None,
        wind_surface: float | None = None,
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

        Returns:
            A DataFrame containing the time-series data, indexed by date.
        """
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
            req_start = pd.to_datetime(start)
            req_end = pd.to_datetime(end)
            date_range = pd.date_range(
                start=req_start,
                end=req_end,
                freq="D" if self.temporal_api == "daily" else "h",
            )
            return pd.DataFrame(np.nan, index=date_range, columns=params)

        return _ensure_all_params_in_df(df, params)

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
        start: str,
        end: str,
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
            )
            future_to_point[future] = point
        return future_to_point

    @staticmethod
    def _collect_futures_results(
        future_to_point: dict,
    ) -> tuple[list[pd.DataFrame], list]:
        """Collects completed futures into result DataFrames and failed list."""
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
                failed_points.append(point)

        return all_results, failed_points

    def get_multi_point_data(
        self,
        points: (
            list[dict[str, Any] | tuple[float, float] | tuple[float, float, float]]
            | pd.DataFrame
        ),
        start: str,
        end: str,
        params: list[str],
        max_workers: int = 8,
    ) -> tuple[
        pd.DataFrame,
        list[dict[str, Any] | tuple[float, float] | tuple[float, float, float]],
    ]:
        """Fetches data for multiple geographic points in parallel.

        Args:
            points: A list of points, where each point is a tuple of (latitude,
                longitude) or (latitude, longitude, elevation), or a DataFrame.
            start: The start date for the data query.
            end: The end date for the data query.
            params: A list of POWER API parameters to fetch.
            max_workers: The maximum number of concurrent threads to use.

        Returns:
            A tuple containing (combined DataFrame, list of failed points).
        """
        parsed_points = self._parse_points_input(points)

        if max_workers > 5:
            logger.warning(
                "NASA explicitly warns against more than 5 concurrent requests. "
                "Proceed with caution."
            )

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

        if self.api_key == "DEMO_KEY":
            key_status, key_type = "DEMO_KEY (Strict limits)", "DEMO_KEY"
        elif self.api_key:
            masked = f"***{self.api_key[-4:]}" if len(self.api_key) > 4 else "***"
            key_status, key_type = f"Provided ({masked})", "Personal (Redacted)"
        else:
            key_status, key_type = "Not provided (Using IP-based limits)", "None"

        table.add_row("API Key Status", key_status)
        table.add_row("Auth Mode", key_type)
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

    def get_regional_data(
        self,
        lat_lon_list: list[tuple[float, float]],
        start: str,
        end: str,
        params: list[str],
    ) -> pd.DataFrame:
        """Fetches data for a list of points using the regional API endpoint.

        Args:
            lat_lon_list: A list of (latitude, longitude) tuples.
            start: The start date for the data query.
            end: The end date for the data query.
            params: A list of POWER API parameters to fetch.

        Returns:
            A DataFrame containing the regional data.
        """
        lon_lat_list = [(lon, lat) for (lat, lon) in lat_lon_list]
        payload = self._build_regional_payload(
            params=params,
            start=start,
            end=end,
            lon_lat_list=lon_lat_list,
        )
        return self._fetch_data(payload, url=self.regional_base_url)

    def get_regional_data_from_coordinates(
        self, coords: list[GeoCoordinate], start: str, end: str, params: list[str]
    ) -> pd.DataFrame:
        """Convenience helper to call the regional endpoint with `GeoCoordinate` objects.

        Args:
            coords: A list of `GeoCoordinate` objects.
            start: The start date for the data query.
            end: The end date for the data query.
            params: A list of POWER API parameters to fetch.

        Returns:
            A DataFrame containing the regional data.
        """
        return self.get_regional_data(
            lat_lon_list=[(c.lat, c.lon) for c in coords],
            start=start,
            end=end,
            params=params,
        )

    def get_expanded_point_data(
        self,
        request: ExpandedPointRequest | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generates a line of points and fetches data for them in parallel.

        Args:
            request: Configuration object.
            **kwargs: Configuration parameters as keyword arguments.

        Returns:
            A combined DataFrame with data for all points along the transect.

        Raises:
            IOError: If fetching data for all generated points fails.
        """
        if request is None:
            request = ExpandedPointRequest(**kwargs)

        lat = request.lat
        lon = request.lon
        start = request.start
        end = request.end
        params = request.params
        axis = request.axis
        distance_km = request.distance_km
        num_points = request.num_points
        elevation = request.elevation
        max_workers = request.max_workers
        points = []
        km_per_lat_deg = 111.1
        km_per_lon_deg = 111.32 * np.cos(np.deg2rad(lat))

        if axis == "lat":
            lat_delta_deg = distance_km / km_per_lat_deg
            lat_start, lat_end = lat - lat_delta_deg / 2, lat + lat_delta_deg / 2
            lats = np.linspace(lat_start, lat_end, num_points)
            points = [(round(p_lat, 4), round(lon, 4)) for p_lat in lats]
        else:
            lon_delta_deg = distance_km / km_per_lon_deg
            lon_start, lon_end = lon - lon_delta_deg / 2, lon + lon_delta_deg / 2
            lons = np.linspace(lon_start, lon_end, num_points)
            points = [(round(lat, 4), round(p_lon, 4)) for p_lon in lons]

        if max_workers > 5:
            logger.warning(
                "NASA explicitly warns against more than 5 concurrent requests. "
                "Proceed with caution."
            )

        logger.info(f"Generated {len(points)} points along the {axis} axis.")

        points_with_metadata: list[dict[str, Any]] = []
        for i, (p_lat, p_lon) in enumerate(points):
            pt = {"lat": p_lat, "lon": p_lon, "name": f"Point_{i + 1}"}
            if elevation is not None:
                pt["elevation"] = elevation
            points_with_metadata.append(pt)

        df, failed_points = self.get_multi_point_data(
            points=points_with_metadata,
            start=start,
            end=end,
            params=params,
            max_workers=max_workers,
        )

        # If the operation resulted in a completely empty dataframe because all
        # points failed, raise an error to make the failure explicit.
        if df.empty and failed_points:
            raise OSError(
                f"Failed to fetch data for all {len(failed_points)} points in the expansion. "
            )

        return df

    def __repr__(self) -> str:
        return f"<PowerClient(api='{self.temporal_api}', url='{self.base_url}')>"

    def __del__(self) -> None:
        """Ensures the database connection is closed when the client is destroyed."""
        if self.db_conn:
            self.db_conn.close()
