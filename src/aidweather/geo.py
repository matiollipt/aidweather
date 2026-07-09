# SPDX-License-Identifier: Apache-2.0
"""
aidweather.geo
~~~~~~~~~~~~~

Geospatial coordinate utilities.

Defines ``GeoCoordinate``, a frozen dataclass for latitude/longitude pairs in
decimal degrees, and parsers/formatters for DD, DDM, and DMS string formats.
Internal representation: latitude −90 to +90 (N positive), longitude −180 to
+180 (E positive).
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
    """Raise ``ValueError`` if *lat* or *lon* are outside valid degree ranges."""
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")


def _hemisphere_from_sign(value: float, is_lat: bool) -> str:
    """Return the hemisphere letter (N/S or E/W) for a signed decimal value."""
    if is_lat:
        return "N" if value >= 0 else "S"
    return "E" if value >= 0 else "W"


def _apply_hemisphere_sign(value: float, hem: str, is_lat: bool) -> float:
    """Apply the correct sign to *value* based on its hemisphere letter.

    Raises:
        ValueError: If *hem* is not valid for the given axis.
    """
    hem = hem.upper()
    if is_lat and hem not in ("N", "S"):
        raise ValueError("Latitude hemisphere must be 'N' or 'S'.")
    if not is_lat and hem not in ("E", "W"):
        raise ValueError("Longitude hemisphere must be 'E' or 'W'.")

    sign = -1.0 if hem in ("S", "W") else 1.0
    return abs(value) * sign


def _normalize_coord_string(s: str) -> str:
    """Normalise a raw coordinate string for consistent regex parsing."""
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
    """Return the DDM components (degrees, decimal minutes, hemisphere) for *value*."""
    hem = _hemisphere_from_sign(value, is_lat)
    v_abs = abs(value)
    deg = int(v_abs)
    minutes_dec = (v_abs - deg) * 60.0
    return deg, minutes_dec, hem


def decimal_to_dms_components(
    value: float, is_lat: bool
) -> tuple[int, int, float, str]:
    """Return the DMS components (degrees, minutes, seconds, hemisphere) for *value*."""
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
    """Parse a Decimal Degrees (DD) string (e.g. ``"45.5° N"``) to a float.

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
    """Parse a Degrees, Decimal Minutes (DDM) string (e.g. ``"45° 30.5' N"``) to a float.

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
    """Parse a Degrees, Minutes, Seconds (DMS) string (e.g. ``"45° 30' 18\" N"``) to a float.

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
    """Try DMS, then DDM, then DD parsers; raise ``ValueError`` if none match."""
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
    """Immutable geographic coordinate (latitude, longitude) in decimal degrees."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        """Validate lat/lon ranges after dataclass initialization."""
        _validate_lat_lon(self.lat, self.lon)

    @classmethod
    def from_decimal(cls, lat: Number, lon: Number) -> GeoCoordinate:
        """Return a ``GeoCoordinate`` from decimal degree numbers."""
        return cls(lat=float(lat), lon=float(lon))

    @classmethod
    def from_dd_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Return a ``GeoCoordinate`` from Decimal Degrees (DD) strings."""
        lat = parse_dd(lat_str, is_lat=True)
        lon = parse_dd(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_ddm_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Return a ``GeoCoordinate`` from Degrees, Decimal Minutes (DDM) strings."""
        lat = parse_ddm(lat_str, is_lat=True)
        lon = parse_ddm(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_dms_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Return a ``GeoCoordinate`` from Degrees, Minutes, Seconds (DMS) strings."""
        lat = parse_dms(lat_str, is_lat=True)
        lon = parse_dms(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_strings(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Return a ``GeoCoordinate`` from strings, auto-detecting DD/DDM/DMS format."""
        lat = parse_any_coord_string(lat_str, is_lat=True)
        lon = parse_any_coord_string(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    def as_decimal(self) -> tuple[float, float]:
        """Return the coordinate as a ``(latitude, longitude)`` tuple of floats."""
        return self.lat, self.lon

    def to_dd(self) -> tuple[float, float]:
        """Return the coordinate as a ``(latitude, longitude)`` tuple of raw decimal floats."""
        return (self.lat, self.lon)

    def to_dd_str(
        self, lat_precision: int = 5, lon_precision: int = 5
    ) -> tuple[str, str]:
        """Format the coordinate as ``(lat_str, lon_str)`` Decimal Degree strings with hemisphere suffixes."""
        lat_hem = _hemisphere_from_sign(self.lat, is_lat=True)
        lon_hem = _hemisphere_from_sign(self.lon, is_lat=False)
        lat_fmt = f"{{:.{lat_precision}f}}° {{}}"
        lon_fmt = f"{{:.{lon_precision}f}}° {{}}"
        return (
            lat_fmt.format(abs(self.lat), lat_hem),
            lon_fmt.format(abs(self.lon), lon_hem),
        )

    def to_ddm_str(self, minute_precision: int = 3) -> tuple[str, str]:
        """Format the coordinate as ``(lat_str, lon_str)`` DDM strings."""
        lat_deg, lat_min, lat_hem = decimal_to_ddm_components(self.lat, is_lat=True)
        lon_deg, lon_min, lon_hem = decimal_to_ddm_components(self.lon, is_lat=False)
        min_fmt = f"{{:.{minute_precision}f}}"
        return (
            f"{lat_deg}°{min_fmt.format(lat_min)}' {lat_hem}",
            f"{lon_deg}°{min_fmt.format(lon_min)}' {lon_hem}",
        )

    def to_dms_str(self, second_precision: int = 0) -> tuple[str, str]:
        """Format the coordinate as ``(lat_str, lon_str)`` DMS strings."""
        lat_deg, lat_min, lat_sec, lat_hem = decimal_to_dms_components(
            self.lat, is_lat=True
        )
        lon_deg, lon_min, lon_sec, lon_hem = decimal_to_dms_components(
            self.lon, is_lat=False
        )
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
    """Internal helper to normalise a (lat, lon) pair into a ``GeoCoordinate``."""
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
    """Normalise various coordinate inputs into a ``GeoCoordinate``.

    Accepts: an existing ``GeoCoordinate``; a ``(lat, lon)`` tuple/list;
    or two separate *lat* / *lon* arguments (numbers or parseable strings).

    Raises:
        TypeError: If the arguments do not match any accepted form.
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

    raise TypeError(
        "normalize_coord_input expects: GeoCoordinate, (lat, lon) pair, or lat+lon."
    )
