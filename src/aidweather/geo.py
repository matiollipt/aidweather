# SPDX-License-Identifier: Apache-2.0
"""
aidweather.geo
~~~~~~~~~~~~~

This module provides robust utilities for handling and converting geospatial coordinates.

It defines the `GeoCoordinate` class, a frozen dataclass that serves as a
value object for representing latitude and longitude pairs in decimal degrees.
The module is designed to reliably parse coordinates from various common string
formats—Degrees, Minutes, Seconds (DMS), Degrees, Decimal Minutes (DDM), and
Decimal Degrees (DD)—and format them back into those string representations.

Core Features:
- `GeoCoordinate` class for type-safe coordinate representation.
- Parsers for DMS, DDM, and DD string formats with hemisphere support (N, S, E, W).
- Formatters to convert decimal degrees back into DMS, DDM, and DD strings.
- A high-level `normalize_coord_input` function to unify various input types
  (e.g., tuples, separate arguments, strings, numbers) into a `GeoCoordinate` object.
- Internal validation to ensure latitude and longitude values are within their
  valid ranges (-90 to 90 for latitude, -180 to 180 for longitude).

The internal representation is always decimal degrees, with the convention:
- Latitude:  -90 to +90 (North positive)
- Longitude: -180 to +180 (East positive)

Example:
    >>> from aidweather.geo import GeoCoordinate, normalize_coord_input
    >>> # From decimal degrees
    >>> coord1 = GeoCoordinate.from_decimal(-23.55, -46.63)
    >>> # From a DMS string
    >>> coord2 = GeoCoordinate.from_strings("23°33'0.0\\" S", "46°37'48.0\\" W")
    >>> print(coord1)
    >>> print(coord2.to_dms_str())
    >>> # Normalize mixed input
    >>> normalized = normalize_coord_input((-23.55, "-46°37'48.0\\" W"))
    >>> print(normalized)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

Number = int | float


# ---------------------------------------------------------------------------
# Helpers and validation
# ---------------------------------------------------------------------------


def _validate_lat_lon(lat: float, lon: float) -> None:
    """Validates that latitude and longitude are within their valid ranges.

    Args:
        lat: Latitude to validate.
        lon: Longitude to validate.

    Raises:
        ValueError: If latitude is outside [-90, 90] or longitude is
            outside [-180, 180].
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")


def _hemisphere_from_sign(value: float, is_lat: bool) -> str:
    """Determines the hemisphere (N/S or E/W) from a signed decimal value.

    Args:
        value: The decimal degree value.
        is_lat: True if the value is a latitude, False for longitude.

    Returns:
        The hemisphere character ("N", "S", "E", or "W").
    """
    if is_lat:
        return "N" if value >= 0 else "S"
    return "E" if value >= 0 else "W"


def _apply_hemisphere_sign(value: float, hem: str, is_lat: bool) -> float:
    """Applies the correct sign to a numeric value based on its hemisphere.

    Args:
        value: The numeric value (should be positive).
        hem: The hemisphere character ("N", "S", "E", or "W").
        is_lat: True if the value is a latitude, False for longitude.

    Returns:
        The value with the sign applied (e.g., negative for "S" or "W").

    Raises:
        ValueError: If the hemisphere character is invalid for the given type.
    """
    hem = hem.upper()
    if is_lat and hem not in ("N", "S"):
        raise ValueError("Latitude hemisphere must be 'N' or 'S'.")
    if not is_lat and hem not in ("E", "W"):
        raise ValueError("Longitude hemisphere must be 'E' or 'W'.")

    sign = -1.0 if hem in ("S", "W") else 1.0
    return abs(value) * sign


def _normalize_coord_string(s: str) -> str:
    """Normalizes a coordinate string for consistent regex parsing.

    Args:
        s: The raw coordinate string.

    Returns:
        The cleaned and normalized string.
    """
    s = s.strip()

    # Normalize Unicode variants
    replacements = {
        "º": "°",
        "˚": "°",
        "’": "'",  # noqa: RUF001
        "′": "'",  # noqa: RUF001
        "＇": "'",  # noqa: RUF001
        "”": '"',
        "″": '"',
        "“": '"',
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)

    # Some people write 23°32'51'' S (double apostrophe for seconds)
    s = s.replace("''", '"')

    # Collapse repeated spaces
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Regex patterns for parsing DD / DDM / DMS strings
# ---------------------------------------------------------------------------


_DMS_RE = re.compile(
    r"""
    ^   # Start of string
    (?P<deg>-?\d+)\s*°\s*
    (?P<min>\d+)\s*'\s*
    (?P<sec>\d+(?:\.\d+)?)\s*"?\s*
    (?P<hem>[NnSsEeWw])?
    $   # End of string
    """,
    re.VERBOSE,
)

_DDM_RE = re.compile(
    r"""
    ^   # Start of string
    (?P<deg>-?\d+)\s*°\s*
    (?P<min>\d+(?:\.\d+)?)\s*'\s*
    (?P<hem>[NnSsEeWw])?
    $   # End of string
    """,
    re.VERBOSE,
)

_DD_RE = re.compile(
    r"""
    ^   # Start of string
    (?P<val>-?\d+(?:\.\d+)?)\s*°?\s*
    (?P<hem>[NnSsEeWw])?
    $   # End of string
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Conversions between decimal / DDM / DMS (numeric)
# ---------------------------------------------------------------------------


def decimal_to_ddm_components(value: float, is_lat: bool) -> tuple[int, float, str]:
    """Converts a decimal degree value to its DDM components.

    Args:
        value: The decimal degree value.
        is_lat: True if the value is a latitude, False for longitude.

    Returns:
        A tuple containing (integer degrees, decimal minutes, hemisphere).
    """
    hem = _hemisphere_from_sign(value, is_lat)
    v_abs = abs(value)
    deg = int(v_abs)
    minutes_dec = (v_abs - deg) * 60.0
    return deg, minutes_dec, hem


def decimal_to_dms_components(value: float, is_lat: bool) -> tuple[int, int, float, str]:
    """Converts a decimal degree value to its DMS components.

    Args:
        value: The decimal degree value.
        is_lat: True if the value is a latitude, False for longitude.

    Returns:
        A tuple containing (integer degrees, integer minutes, decimal seconds, hemisphere).
    """
    hem = _hemisphere_from_sign(value, is_lat)
    v_abs = abs(value)
    deg = int(v_abs)
    minutes_full = (v_abs - deg) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    return deg, minutes, seconds, hem


# ---------------------------------------------------------------------------
# String parsers: DD / DDM / DMS
# ---------------------------------------------------------------------------


def parse_dd(s: str, is_lat: bool) -> float:
    """Parses a Decimal Degrees (DD) string to a float value.

    Args:
        s: The DD string (e.g., "45.5° N" or "-45.5").
        is_lat: True if parsing a latitude, False for longitude.

    Returns:
        The coordinate value in decimal degrees.

    Raises:
        ValueError: If the string format is invalid.
    """
    s = _normalize_coord_string(s)
    m = _DD_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid DD coordinate: {s!r}")
    val = float(m.group("val"))
    hem = m.group("hem")
    if hem:
        return _apply_hemisphere_sign(val, hem, is_lat)
    # hemisphere omitted => sign is trusted
    return val


def parse_ddm(s: str, is_lat: bool) -> float:
    """Parses a Degrees, Decimal Minutes (DDM) string to a float value.

    Args:
        s: The DDM string (e.g., "45° 30.5' N").
        is_lat: True if parsing a latitude, False for longitude.

    Returns:
        The coordinate value in decimal degrees.

    Raises:
        ValueError: If the string format is invalid.
    """
    s = _normalize_coord_string(s)
    m = _DDM_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid DDM coordinate: {s!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    hem = m.group("hem") or ("N" if is_lat else "E")
    val = abs(deg) + minutes / 60.0
    return _apply_hemisphere_sign(val, hem, is_lat)


def parse_dms(s: str, is_lat: bool) -> float:
    """Parses a Degrees, Minutes, Seconds (DMS) string to a float value.

    Args:
        s: The DMS string (e.g., "45° 30' 18\\" N").
        is_lat: True if parsing a latitude, False for longitude.

    Returns:
        The coordinate value in decimal degrees.

    Raises:
        ValueError: If the string format is invalid.
    """
    s = _normalize_coord_string(s)
    m = _DMS_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid DMS coordinate: {s!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    seconds = float(m.group("sec"))
    hem = m.group("hem") or ("N" if is_lat else "E")
    val = abs(deg) + minutes / 60.0 + seconds / 3600.0
    return _apply_hemisphere_sign(val, hem, is_lat)


def parse_any_coord_string(s: str, is_lat: bool) -> float:
    """Tries to parse a coordinate string by attempting DMS, DDM, and DD formats.

    Args:
        s: The coordinate string to parse.
        is_lat: True if parsing a latitude, False for longitude.

    Returns:
        The parsed coordinate value in decimal degrees.

    Raises:
        ValueError: If the string cannot be parsed by any of the formats.
    """
    last_error: Exception | None = None
    for parser in (parse_dms, parse_ddm, parse_dd):
        try:
            return parser(s, is_lat=is_lat)
        except ValueError as e:
            last_error = e
    raise ValueError(f"Could not parse coordinate string {s!r}: {last_error}")


# ---------------------------------------------------------------------------
# GeoCoordinate value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoCoordinate:
    """Represents a geographic coordinate in decimal degrees.

    Attributes:
        lat: Latitude in decimal degrees (-90 to 90, N is positive).
        lon: Longitude in decimal degrees (-180 to 180, E is positive).
    """

    lat: float
    lon: float

    def __post_init__(self) -> None:
        """Validates the coordinates after initialization."""
        _validate_lat_lon(self.lat, self.lon)

    @classmethod
    def from_decimal(cls, lat: Number, lon: Number) -> GeoCoordinate:
        """Creates a GeoCoordinate from decimal degree numbers.

        Args:
            lat: Latitude as a float or int.
            lon: Longitude as a float or int.

        Returns:
            A new GeoCoordinate instance.
        """
        return cls(lat=float(lat), lon=float(lon))

    @classmethod
    def from_dd_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Creates a GeoCoordinate from Decimal Degrees (DD) strings.

        Args:
            lat_str: The latitude string in DD format.
            lon_str: The longitude string in DD format.

        Returns:
            A new GeoCoordinate instance.
        """
        lat = parse_dd(lat_str, is_lat=True)
        lon = parse_dd(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_ddm_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Creates a GeoCoordinate from Degrees, Decimal Minutes (DDM) strings.

        Args:
            lat_str: The latitude string in DDM format.
            lon_str: The longitude string in DDM format.

        Returns:
            A new GeoCoordinate instance.
        """
        lat = parse_ddm(lat_str, is_lat=True)
        lon = parse_ddm(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_dms_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Creates a GeoCoordinate from Degrees, Minutes, Seconds (DMS) strings.

        Args:
            lat_str: The latitude string in DMS format.
            lon_str: The longitude string in DMS format.

        Returns:
            A new GeoCoordinate instance.
        """
        lat = parse_dms(lat_str, is_lat=True)
        lon = parse_dms(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_strings(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Creates a GeoCoordinate from strings, auto-detecting the format.

        Args:
            lat_str: The latitude string.
            lon_str: The longitude string.

        Returns:
            A new GeoCoordinate instance.
        """
        lat = parse_any_coord_string(lat_str, is_lat=True)
        lon = parse_any_coord_string(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    def as_decimal(self) -> tuple[float, float]:
        """Returns the coordinate as a (latitude, longitude) tuple.

        Returns:
            A tuple containing the latitude and longitude.
        """
        return self.lat, self.lon

    def to_dd(self) -> tuple[float, float]:
        """Returns the coordinate as a (latitude, longitude) tuple of floats.

        Returns:
            The raw decimal degree values.
        """
        return (self.lat, self.lon)

    def to_dd_str(self, lat_precision: int = 5, lon_precision: int = 5) -> tuple[str, str]:
        """Formats the coordinate into Decimal Degree strings with hemispheres.

        Args:
            lat_precision: The number of decimal places for latitude.
            lon_precision: The number of decimal places for longitude.

        Returns:
            A tuple of formatted (latitude, longitude) strings.
        """
        lat_hem = _hemisphere_from_sign(self.lat, is_lat=True)
        lon_hem = _hemisphere_from_sign(self.lon, is_lat=False)
        lat_fmt = f"{{:.{lat_precision}f}}° {{}}"
        lon_fmt = f"{{:.{lon_precision}f}}° {{}}"
        return (
            lat_fmt.format(abs(self.lat), lat_hem),
            lon_fmt.format(abs(self.lon), lon_hem),
        )

    def to_ddm_str(self, minute_precision: int = 3) -> tuple[str, str]:
        """Formats the coordinate into DDM strings.

        Args:
            minute_precision: The number of decimal places for the minutes part.

        Returns:
            A tuple of formatted (latitude, longitude) strings.
        """
        lat_deg, lat_min, lat_hem = decimal_to_ddm_components(self.lat, is_lat=True)
        lon_deg, lon_min, lon_hem = decimal_to_ddm_components(self.lon, is_lat=False)
        min_fmt = f"{{:.{minute_precision}f}}"
        return (
            f"{lat_deg}°{min_fmt.format(lat_min)}' {lat_hem}",
            f"{lon_deg}°{min_fmt.format(lon_min)}' {lon_hem}",
        )

    def to_dms_str(self, second_precision: int = 0) -> tuple[str, str]:
        """Formats the coordinate into DMS strings.

        Args:
            second_precision: The number of decimal places for the seconds part.

        Returns:
            A tuple of formatted (latitude, longitude) strings.
        """
        lat_deg, lat_min, lat_sec, lat_hem = decimal_to_dms_components(self.lat, is_lat=True)
        lon_deg, lon_min, lon_sec, lon_hem = decimal_to_dms_components(self.lon, is_lat=False)
        sec_fmt = f"{{:.{second_precision}f}}"
        # Use double quote for seconds to avoid escaping hell
        return (
            f"{lat_deg}°{lat_min}'{sec_fmt.format(lat_sec)}\" {lat_hem}",
            f"{lon_deg}°{lon_min}'{sec_fmt.format(lon_sec)}\" {lon_hem}",
        )


# ---------------------------------------------------------------------------
# High-level normalizer
# ---------------------------------------------------------------------------


def _looks_like_number(x: Any) -> bool:
    """Checks if a value can be converted to a float."""
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_two_values(a: Any, b: Any) -> GeoCoordinate:
    """Internal helper to normalize a pair of values into a GeoCoordinate.

    Args:
        a: The first value (typically latitude).
        b: The second value (typically longitude).

    Returns:
        The normalized coordinate object.
    """
    # Case: numbers (or number-like strings) without hemisphere
    if _looks_like_number(a) and _looks_like_number(b):
        # Trust that sign is encoded correctly (negative for S/W)
        return GeoCoordinate.from_decimal(float(a), float(b))

    # Otherwise, treat them as coordinate strings (DD / DDM / DMS)
    return GeoCoordinate.from_strings(str(a), str(b))


def normalize_coord_input(
    lat: Any,
    lon: Any | None = None,
) -> GeoCoordinate:
    """Normalizes various common coordinate input formats into a GeoCoordinate object.

    Accepted forms:
    - A `GeoCoordinate` object: returned as-is.
    - A `(lat, lon)` tuple/list of numbers: treated as decimal degrees.
    - A `(lat_str, lon_str)` tuple/list of strings: parsed as DD/DDM/DMS.
    - Two separate arguments `lat`, `lon`: can be numbers or strings.

    Args:
        lat: The latitude value, or a `GeoCoordinate` object, or a tuple/list
            containing both latitude and longitude.
        lon: The longitude value. Required if `lat` is not a `GeoCoordinate` or
            a tuple/list.

    Returns:
        The standardized coordinate object.

    Raises:
        TypeError: If the input arguments do not match one of the accepted forms.
    """
    # Already normalized
    if isinstance(lat, GeoCoordinate) and lon is None:
        return lat

    # Single tuple/list argument: (lat, lon)
    if lon is None and isinstance(lat, (tuple, list)) and len(lat) == 2:
        a, b = lat
        return _normalize_two_values(a, b)

    # Separate lat, lon arguments
    if lon is not None:
        return _normalize_two_values(lat, lon)

    raise TypeError("normalize_coord_input expects: GeoCoordinate, (lat, lon) pair, or lat+lon.")
