# SPDX-License-Identifier: Apache-2.0
"""
aidweather.geo
~~~~~~~~~~~~~~

Geospatial coordinate utilities for ``aidweather``.

This module provides:

- :class:`GeoCoordinate` — an immutable, validated ``(lat, lon)`` value object
  in decimal degrees (DD), the internal representation used throughout the library.
- :func:`normalize_coord_input` — a flexible high-level intake function that
  accepts plain numbers, tuples, or parseable coordinate strings and returns a
  ``GeoCoordinate``.
- String parsers for Decimal Degrees (:func:`parse_dd`), Degrees Decimal Minutes
  (:func:`parse_ddm`), and Degrees Minutes Seconds (:func:`parse_dms`) notation,
  plus an auto-detecting dispatcher (:func:`parse_any_coord_string`).
- Numeric decomposition helpers (:func:`decimal_to_ddm_components`,
  :func:`decimal_to_dms_components`) for formatting and display use-cases.

**Internal representation**

Latitude: −90.0 to +90.0 — positive values indicate North.
Longitude: −180.0 to +180.0 — positive values indicate East.

This convention matches WGS-84, NASA POWER, and virtually all modern GIS
frameworks. Hemisphere letters (``N``/``S``/``E``/``W``) in input strings are
transparently converted to the signed decimal convention.

**Unicode handling**

Input strings are normalised before parsing: common Unicode variants of the
degree symbol (``º``, ``˚``), prime characters (``′``, ``″``), and curly quotes
are silently replaced with their ASCII equivalents. Double-apostrophe seconds
(``''``) are also accepted.

**Cross-references**

See ``docs/geo_reference.md`` for the full narrative guide, accepted format
examples, ambiguity policies, and usage patterns.
See ``docs/spatial_interpretation.md`` for how NASA POWER maps coordinates to
grid cells.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GeoCoordinate",
    "normalize_coord_input",
    "decimal_to_ddm_components",
    "decimal_to_dms_components",
    "parse_dd",
    "parse_ddm",
    "parse_dms",
    "parse_any_coord_string",
]

# Internal numeric type alias — not part of the public API.
_Number = int | float


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
    """Decompose a signed decimal degree value into Degrees, Decimal Minutes components.

    Useful when you need the raw numeric DDM components for display or further
    computation, without going through the full string-formatting pipeline of
    :meth:`GeoCoordinate.to_ddm_str`.

    Args:
        value: Signed decimal degree value (latitude or longitude).
        is_lat: ``True`` if *value* is a latitude (produces ``'N'``/``'S'`` hemisphere
            letters); ``False`` for longitude (produces ``'E'``/``'W'``).

    Returns:
        A ``(degrees, decimal_minutes, hemisphere)`` tuple where *degrees* is a
        non-negative integer, *decimal_minutes* is a non-negative float in
        ``[0, 60)``, and *hemisphere* is one of ``'N'``, ``'S'``, ``'E'``, ``'W'``.

    Example::

        decimal_to_ddm_components(-23.55, is_lat=True)
        # (23, 33.0, 'S')

        decimal_to_ddm_components(-46.633333, is_lat=False)
        # (46, 37.99980000000033, 'W')
    """
    hem = _hemisphere_from_sign(value, is_lat)
    v_abs = abs(value)
    deg = int(v_abs)
    minutes_dec = (v_abs - deg) * 60.0
    return deg, minutes_dec, hem


def decimal_to_dms_components(
    value: float, is_lat: bool
) -> tuple[int, int, float, str]:
    """Decompose a signed decimal degree value into Degrees, Minutes, Seconds components.

    Useful when you need the raw numeric DMS components for display or further
    computation, without going through the full string-formatting pipeline of
    :meth:`GeoCoordinate.to_dms_str`.

    Args:
        value: Signed decimal degree value (latitude or longitude).
        is_lat: ``True`` if *value* is a latitude (produces ``'N'``/``'S'`` hemisphere
            letters); ``False`` for longitude (produces ``'E'``/``'W'``).

    Returns:
        A ``(degrees, minutes, seconds, hemisphere)`` tuple where *degrees* is a
        non-negative integer, *minutes* is a non-negative integer in ``[0, 60)``,
        *seconds* is a non-negative float in ``[0, 60)``, and *hemisphere* is one
        of ``'N'``, ``'S'``, ``'E'``, ``'W'``.

    Example::

        decimal_to_dms_components(-23.55, is_lat=True)
        # (23, 33, 0.0, 'S')

        decimal_to_dms_components(-46.633333, is_lat=False)
        # (46, 37, 59.99880000001965, 'W')
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
    """Parse a Decimal Degrees (DD) string to a signed float.

    Accepts strings of the form ``"<value>[°] [N|S|E|W]"``. The degree symbol
    and hemisphere letter are both optional; if the hemisphere letter is absent
    the sign of *value* is trusted directly.

    Unicode variants of the degree symbol (``º``, ``˚``) and whitespace
    normalisation are handled automatically by the internal normaliser.

    Args:
        s: A DD coordinate string, e.g. ``"23.55° S"``, ``"-23.55"``,
            or ``"23.55S"``.
        is_lat: ``True`` if parsing a latitude string (validates ``N``/``S``
            hemisphere letters and sign convention); ``False`` for longitude
            (validates ``E``/``W``).

    Returns:
        Signed decimal degree float. Negative values indicate South (latitude)
        or West (longitude).

    Raises:
        ValueError: If *s* does not match the expected DD pattern or carries an
            invalid hemisphere letter for the given axis.

    Example::

        parse_dd("23.55° S", is_lat=True)   # -23.55
        parse_dd("-23.55",   is_lat=True)   # -23.55
        parse_dd("46.63° W", is_lat=False)  # -46.63
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
    """Parse a Degrees, Decimal Minutes (DDM) string to a signed float.

    Accepts strings of the form ``"<deg>° <decimal_min>' [N|S|E|W]"``. The
    hemisphere letter is optional; if absent, the sign of the degrees component
    is used. Unicode prime characters (``′``) are normalised to ASCII
    single-quotes automatically.

    **Ambiguity policy**: If the string carries *both* a leading negative sign
    on the degrees component *and* a hemisphere letter (e.g.
    ``"-23° 33.0' S"``), a :class:`UserWarning` is emitted and the hemisphere
    letter takes precedence. This matches the behaviour of GPS firmware and
    chart-plotting software where the hemisphere letter is the authoritative
    sign specifier.

    Args:
        s: A DDM coordinate string, e.g. ``"23° 33.0' S"`` or
            ``"46° 37.998' W"``.
        is_lat: ``True`` for latitude strings; ``False`` for longitude.

    Returns:
        Signed decimal degree float.

    Raises:
        ValueError: If *s* does not match the expected DDM pattern or carries an
            invalid hemisphere letter for the given axis.

    Warns:
        UserWarning: If *s* contains both a negative degree value and a
            hemisphere letter.

    Example::

        parse_ddm("23° 33.0' S",    is_lat=True)   # -23.55
        parse_ddm("46° 37.998' W",  is_lat=False)  # -46.6333
    """
    s = _normalize_coord_string(s)
    m = _DDM_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid DDM coordinate: {s!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    if m.group("hem") and deg < 0:
        warnings.warn(
            f"Coordinate string {s!r} has both a negative degree value and a hemisphere "
            f"letter '{m.group('hem')}'. The negative sign is ignored; the hemisphere "
            "letter takes precedence.",
            UserWarning,
            stacklevel=2,
        )
    hem = m.group("hem") or ("N" if is_lat else "E")
    val = abs(deg) + minutes / 60.0
    return _apply_hemisphere_sign(val, hem, is_lat)


def parse_dms(s: str, is_lat: bool) -> float:
    """Parse a Degrees, Minutes, Seconds (DMS) string to a signed float.

    Accepts strings of the form
    ``"<deg>° <min>' <sec>[\"\'] [N|S|E|W]"``. The seconds value may include a
    decimal component (e.g. ``"59.88\"``). The trailing double-quote character
    and hemisphere letter are both optional. Unicode prime/double-prime symbols
    (``′``, ``″``) and double-apostrophe seconds (``''``) are normalised
    automatically.

    **Ambiguity policy**: Identical to :func:`parse_ddm` — if both a leading
    negative sign and a hemisphere letter are present, a :class:`UserWarning`
    is emitted and the hemisphere letter takes precedence.

    Args:
        s: A DMS coordinate string, e.g. ``"23° 33' 0\" S"`` or
            ``"46° 37' 59.88\" W"``.
        is_lat: ``True`` for latitude strings; ``False`` for longitude.

    Returns:
        Signed decimal degree float.

    Raises:
        ValueError: If *s* does not match the expected DMS pattern or carries an
            invalid hemisphere letter for the given axis.

    Warns:
        UserWarning: If *s* contains both a negative degree value and a
            hemisphere letter.

    Example::

        parse_dms('23° 33\' 0" S',      is_lat=True)   # -23.55
        parse_dms('46° 37\' 59.88" W',  is_lat=False)  # approx. -46.6333
    """
    s = _normalize_coord_string(s)
    m = _DMS_RE.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid DMS coordinate: {s!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    seconds = float(m.group("sec"))
    if m.group("hem") and deg < 0:
        warnings.warn(
            f"Coordinate string {s!r} has both a negative degree value and a hemisphere "
            f"letter '{m.group('hem')}'. The negative sign is ignored; the hemisphere "
            "letter takes precedence.",
            UserWarning,
            stacklevel=2,
        )
    hem = m.group("hem") or ("N" if is_lat else "E")
    val = abs(deg) + minutes / 60.0 + seconds / 3600.0
    return _apply_hemisphere_sign(val, hem, is_lat)


def parse_any_coord_string(s: str, is_lat: bool) -> float:
    """Auto-detect and parse a coordinate string in DD, DDM, or DMS format.

    Attempts parsers in the order **DMS → DDM → DD**. DMS is tried first
    because it is the most specific format (three numeric groups); DD is
    tried last as it is the most permissive. A string that matches DMS will
    never be mis-parsed as DD.

    This function is used internally by :meth:`GeoCoordinate.from_strings` and
    by :func:`normalize_coord_input` when string inputs are provided.

    Args:
        s: A coordinate string in any of the accepted formats. See
            ``docs/geo_reference.md`` for the full list of accepted variants.
        is_lat: ``True`` for latitude strings (validates ``N``/``S``
            hemisphere letters); ``False`` for longitude (validates ``E``/``W``).

    Returns:
        Signed decimal degree float.

    Raises:
        ValueError: If none of the three parsers can match *s*. The error
            message includes the last parser's failure reason.

    Example::

        parse_any_coord_string('23° 33\' 0" S', is_lat=True)  # DMS → -23.55
        parse_any_coord_string("23° 33.0' S",   is_lat=True)  # DDM → -23.55
        parse_any_coord_string("23.55° S",       is_lat=True)  # DD  → -23.55
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
    """Immutable geographic coordinate (latitude, longitude) in decimal degrees.

    ``GeoCoordinate`` is a **frozen dataclass**: it is immutable after
    construction, hashable, and safe to use as a dictionary key or in sets.
    Latitude/longitude range validation runs automatically on construction.

    Attributes:
        lat: Latitude in decimal degrees. Range: −90.0 (South Pole) to +90.0
            (North Pole). Positive values indicate the Northern Hemisphere.
        lon: Longitude in decimal degrees. Range: −180.0 to +180.0. Positive
            values indicate East of the Prime Meridian.

    Construction classmethods:

    - :meth:`from_decimal` — from plain numbers.
    - :meth:`from_strings` — from strings, auto-detecting DD, DDM, or DMS format.
    - :meth:`from_dd_str` — from explicit Decimal Degrees strings.
    - :meth:`from_ddm_str` — from explicit Degrees Decimal Minutes strings.
    - :meth:`from_dms_str` — from explicit Degrees Minutes Seconds strings.

    Formatting methods:

    - :meth:`as_decimal` / :meth:`to_dd` — ``(lat, lon)`` float tuple.
    - :meth:`to_dd_str` — hemisphere-annotated DD strings.
    - :meth:`to_ddm_str` — DDM strings.
    - :meth:`to_dms_str` — DMS strings.

    Example::

        coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
        coord.lat    # -23.55
        coord.lon    # -46.633333
        coord.to_dd_str()    # ('23.55000° S', '46.63333° W')
        coord.to_ddm_str()   # ('23°33.000\' S', '46°37.998\' W')

    See ``docs/geo_reference.md`` for the full narrative guide and examples.
    """

    lat: float
    lon: float

    def __post_init__(self) -> None:
        """Validate lat/lon ranges after dataclass initialization.

        Raises:
            ValueError: If ``lat`` is outside [−90, 90] or ``lon`` is outside
                [−180, 180].
        """
        _validate_lat_lon(self.lat, self.lon)

    @classmethod
    def from_decimal(cls, lat: _Number, lon: _Number) -> GeoCoordinate:
        """Construct a ``GeoCoordinate`` from decimal degree numbers.

        This is the most direct construction path. Both ``int`` and ``float``
        inputs are accepted; they are cast to ``float`` internally.

        Args:
            lat: Latitude in decimal degrees. Positive = North, negative = South.
            lon: Longitude in decimal degrees. Positive = East, negative = West.

        Returns:
            A validated :class:`GeoCoordinate` instance.

        Raises:
            ValueError: If *lat* or *lon* are outside their valid ranges.

        Example::

            GeoCoordinate.from_decimal(-23.55, -46.633333)
            # GeoCoordinate(lat=-23.55, lon=-46.633333)
        """
        return cls(lat=float(lat), lon=float(lon))

    @classmethod
    def from_dd_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Construct a ``GeoCoordinate`` from Decimal Degrees (DD) strings.

        Use this method when the input is known to be in DD format. It will
        raise ``ValueError`` for DDM or DMS strings; use :meth:`from_strings`
        if the format is uncertain.

        Args:
            lat_str: Latitude DD string, e.g. ``"23.55° S"`` or ``"-23.55"``.
            lon_str: Longitude DD string, e.g. ``"46.63° W"`` or ``"-46.63"``.

        Returns:
            A validated :class:`GeoCoordinate` instance.

        Raises:
            ValueError: If either string does not match the DD pattern.
        """
        lat = parse_dd(lat_str, is_lat=True)
        lon = parse_dd(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_ddm_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Construct a ``GeoCoordinate`` from Degrees, Decimal Minutes (DDM) strings.

        Args:
            lat_str: Latitude DDM string, e.g. ``"23° 33.0' S"``.
            lon_str: Longitude DDM string, e.g. ``"46° 37.998' W"``.

        Returns:
            A validated :class:`GeoCoordinate` instance.

        Raises:
            ValueError: If either string does not match the DDM pattern.

        Warns:
            UserWarning: If a string carries both a negative degree value and
                a hemisphere letter (see :func:`parse_ddm`).
        """
        lat = parse_ddm(lat_str, is_lat=True)
        lon = parse_ddm(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_dms_str(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Construct a ``GeoCoordinate`` from Degrees, Minutes, Seconds (DMS) strings.

        Args:
            lat_str: Latitude DMS string, e.g. ``'23° 33\' 0" S'``.
            lon_str: Longitude DMS string, e.g. ``'46° 37\' 59.88" W'``.

        Returns:
            A validated :class:`GeoCoordinate` instance.

        Raises:
            ValueError: If either string does not match the DMS pattern.

        Warns:
            UserWarning: If a string carries both a negative degree value and
                a hemisphere letter (see :func:`parse_dms`).
        """
        lat = parse_dms(lat_str, is_lat=True)
        lon = parse_dms(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    @classmethod
    def from_strings(cls, lat_str: str, lon_str: str) -> GeoCoordinate:
        """Construct a ``GeoCoordinate`` from strings, auto-detecting DD, DDM, or DMS format.

        Delegates to :func:`parse_any_coord_string` for each axis, which tries
        parsers in the order **DMS → DDM → DD**. Use this method when the
        input coordinate format is not known in advance.

        Args:
            lat_str: Latitude string in any accepted format.
            lon_str: Longitude string in any accepted format.

        Returns:
            A validated :class:`GeoCoordinate` instance.

        Raises:
            ValueError: If either string cannot be parsed by any of the three
                parsers.

        Example::

            # All three are equivalent:
            GeoCoordinate.from_strings('23° 33\' 0" S', '46° 37\' 59.88" W')
            GeoCoordinate.from_strings("23° 33.0' S", "46° 37.998' W")
            GeoCoordinate.from_strings("23.55° S", "46.6333° W")
        """
        lat = parse_any_coord_string(lat_str, is_lat=True)
        lon = parse_any_coord_string(lon_str, is_lat=False)
        return cls(lat=lat, lon=lon)

    def as_decimal(self) -> tuple[float, float]:
        """Return the coordinate as a ``(latitude, longitude)`` tuple of signed floats.

        Returns:
            ``(lat, lon)`` where both values are in signed decimal degrees.
        """
        return self.lat, self.lon

    def to_dd(self) -> tuple[float, float]:
        """Return the coordinate as a ``(latitude, longitude)`` tuple of signed floats.

        This is an alias for :meth:`as_decimal`, provided for naming symmetry
        with :meth:`to_ddm_str` and :meth:`to_dms_str`.

        Returns:
            ``(lat, lon)`` where both values are in signed decimal degrees.
        """
        return self.as_decimal()

    def to_dd_str(
        self, lat_precision: int = 5, lon_precision: int = 5
    ) -> tuple[str, str]:
        """Format the coordinate as hemisphere-annotated Decimal Degree strings.

        Both the latitude and longitude are formatted as absolute values
        (positive) followed by the appropriate hemisphere letter.

        Args:
            lat_precision: Number of decimal places in the latitude component.
                Defaults to ``5``, which gives sub-metre precision (~1.1 m at
                the equator per 0.00001°).
            lon_precision: Number of decimal places in the longitude component.
                Defaults to ``5``.

        Returns:
            ``(lat_str, lon_str)`` tuple, e.g. ``('23.55000° S', '46.63333° W')``.

        Example::

            coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
            coord.to_dd_str()
            # ('23.55000° S', '46.63333° W')

            coord.to_dd_str(lat_precision=2, lon_precision=2)
            # ('23.55° S', '46.63° W')
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
        """Format the coordinate as Degrees, Decimal Minutes (DDM) strings.

        The degrees component is a non-negative integer; the hemisphere letter
        encodes the sign. Commonly used in marine navigation and GPS device
        output formats.

        Args:
            minute_precision: Number of decimal places on the minutes component.
                Defaults to ``3``.

        Returns:
            ``(lat_str, lon_str)`` tuple, e.g. ``("23°33.000' S", "46°37.998' W")``.

        Example::

            coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
            coord.to_ddm_str()
            # ("23°33.000' S", "46°37.998' W")
        """
        lat_deg, lat_min, lat_hem = decimal_to_ddm_components(self.lat, is_lat=True)
        lon_deg, lon_min, lon_hem = decimal_to_ddm_components(self.lon, is_lat=False)
        min_fmt = f"{{:.{minute_precision}f}}"
        return (
            f"{lat_deg}°{min_fmt.format(lat_min)}' {lat_hem}",
            f"{lon_deg}°{min_fmt.format(lon_min)}' {lon_hem}",
        )

    def to_dms_str(self, second_precision: int = 0) -> tuple[str, str]:
        """Format the coordinate as Degrees, Minutes, Seconds (DMS) strings.

        The classical surveying and cartographic notation. The degrees and
        minutes components are non-negative integers; the hemisphere letter
        encodes the sign. Seconds may include a decimal fraction.

        Args:
            second_precision: Number of decimal places on the seconds component.
                Defaults to ``0`` (whole seconds). Increase for sub-second
                precision (e.g. ``second_precision=2`` → ``'59.88"``').

        Returns:
            ``(lat_str, lon_str)`` tuple, e.g.
            ``('23°33\'0" S', '46°37\'60" W')``.

        Example::

            coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
            coord.to_dms_str()
            # ('23°33\'0" S', '46°37\'60" W')

            coord.to_dms_str(second_precision=2)
            # ('23°33\'0.00" S', '46°37\'59.88" W')
        """
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
    """Return ``True`` if *x* can be converted to a float."""
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
    """Normalise any supported coordinate input form into a :class:`GeoCoordinate`.

    This is the primary coordinate intake function used internally by
    :class:`~aidweather.client.PowerClient`. It accepts the widest possible
    range of input forms so that callers do not need to know the specific
    format of their coordinate data in advance.

    Accepted input forms:

    1. **Existing** ``GeoCoordinate`` (returned unchanged)::

           normalize_coord_input(GeoCoordinate.from_decimal(-23.55, -46.63))

    2. **Two separate numeric arguments** (int or float)::

           normalize_coord_input(-23.55, -46.63)

    3. **Two separate parseable string arguments** (DD, DDM, or DMS)::

           normalize_coord_input("23.55° S", "46.63° W")
           normalize_coord_input("23° 33.0' S", "46° 37.998' W")

    4. **A two-element tuple or list** of numbers or parseable strings::

           normalize_coord_input((-23.55, -46.63))
           normalize_coord_input(["23.55° S", "46.63° W"])

    Args:
        lat: A :class:`GeoCoordinate`, a ``(lat, lon)`` tuple/list, or the
            latitude component as a number or parseable string.
        lon: The longitude component (number or parseable string) when *lat*
            is a single value. Must be ``None`` when *lat* is already a
            :class:`GeoCoordinate` or a tuple/list pair.

    Returns:
        A validated :class:`GeoCoordinate` instance.

    Raises:
        TypeError: If the arguments do not match any of the accepted forms
            (e.g. a single non-GeoCoordinate non-sequence argument with no
            *lon*).
        ValueError: If the values are parseable but out of range, or if the
            string format is not recognised.
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
