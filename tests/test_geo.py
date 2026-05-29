# SPDX-License-Identifier: Apache-2.0
# tests/test_geo.py

import pytest
from aidweather.geo import (
    GeoCoordinate,
    _normalize_coord_string,
    _validate_lat_lon,
    normalize_coord_input,
    parse_any_coord_string,
    parse_dd,
    parse_ddm,
    parse_dms,
)

# --- Test internal helper functions ---


def test_validate_lat_lon():
    """
    Tests the internal validator for latitude and longitude ranges.
    - It should pass for valid coordinates.
    - It should raise a ValueError for coordinates outside the valid ranges.
    """
    # Why: This function is critical for ensuring data integrity at the boundary.

    # Test valid cases
    _validate_lat_lon(0, 0)
    _validate_lat_lon(90, 180)
    _validate_lat_lon(-90, -180)

    # Test invalid latitude
    with pytest.raises(ValueError, match="Latitude out of range"):
        _validate_lat_lon(90.1, 0)
    with pytest.raises(ValueError, match="Latitude out of range"):
        _validate_lat_lon(-90.1, 0)

    # Test invalid longitude
    with pytest.raises(ValueError, match="Longitude out of range"):
        _validate_lat_lon(0, 180.1)
    with pytest.raises(ValueError, match="Longitude out of range"):
        _validate_lat_lon(0, -180.1)


def test_normalize_coord_string():
    """
    Tests the string cleaning utility for coordinate strings.
    - It should replace various unicode symbols (degree, quote) with standard ASCII.
    - It should replace double apostrophes with a double quote.
    - It should strip whitespace and collapse internal spaces.
    """
    # Why: Real-world data is messy. This function is the first line of defense.

    # Unicode and symbol variations
    assert _normalize_coord_string("23º 32’ 51” S") == '23° 32\' 51" S'  # noqa: RUF001
    assert _normalize_coord_string("46˚ 38′ 16″ W") == '46° 38\' 16" W'  # noqa: RUF001

    # Double apostrophe for seconds
    assert _normalize_coord_string("23° 32' 51'' S") == '23° 32\' 51" S'

    # Whitespace and padding
    assert _normalize_coord_string("  -46.123   ") == "-46.123"
    assert _normalize_coord_string("45°   30.5' N") == "45° 30.5' N"


# --- Test individual string parsers ---


@pytest.mark.parametrize(
    "dms_str, is_lat, expected",
    [
        # Why: Test DMS parsing, the most complex format, with different variations.
        ('23°33\'0" S', True, -23.55),
        ('46°37\'48" W', False, -46.63),
        ('0°0\'0" N', True, 0.0),
        ('-10°15\'30" E', False, 10.258333),  # Sign should be overridden by hemisphere
    ],
)
def test_parse_dms(dms_str, is_lat, expected):
    """Tests the DMS (Degrees, Minutes, Seconds) parser."""
    result = parse_dms(dms_str, is_lat=is_lat)
    assert result == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "ddm_str, is_lat, expected",
    [
        # Why: Test DDM parsing, common in GPS devices.
        ("23°33.0' S", True, -23.55),
        ("46°38.167' W", False, -46.636117),
        ("0°0.0' N", True, 0.0),
    ],
)
def test_parse_ddm(ddm_str, is_lat, expected):
    """Tests the DDM (Degrees, Decimal Minutes) parser."""
    result = parse_ddm(ddm_str, is_lat=is_lat)
    assert result == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "dd_str, is_lat, expected",
    [
        # Why: Test DD parsing, the simplest and most common computer format.
        ("-23.55", True, -23.55),
        ("23.55° S", True, -23.55),
        ("-46.6361", False, -46.6361),
        ("46.6361° W", False, -46.6361),
    ],
)
def test_parse_dd(dd_str, is_lat, expected):
    """Tests the DD (Decimal Degrees) parser."""
    result = parse_dd(dd_str, is_lat=is_lat)
    assert result == pytest.approx(expected, abs=1e-6)


def test_parse_invalid_formats():
    """Tests that parsers raise ValueError for malformed strings."""
    # Why: Ensure robustness against bad input.
    with pytest.raises(ValueError):
        parse_dms("23 33 0 S", is_lat=True)  # Missing symbols
    with pytest.raises(ValueError):
        parse_ddm("23° S", is_lat=True)  # Missing minutes
    with pytest.raises(ValueError):
        parse_dd("23.55 S W", is_lat=True)  # Gibberish


def test_parse_any_coord_string():
    """
    Tests the universal parser that tries DMS, DDM, and DD formats.
    """
    # Why: This is a key user-facing function that needs to be flexible.
    assert parse_any_coord_string('23°33\'0" S', is_lat=True) == pytest.approx(-23.55)
    assert parse_any_coord_string("23°33.0' S", is_lat=True) == pytest.approx(-23.55)
    assert parse_any_coord_string("-23.55", is_lat=True) == pytest.approx(-23.55)

    with pytest.raises(ValueError, match="Could not parse"):
        parse_any_coord_string("invalid string", is_lat=True)


# --- Test GeoCoordinate class ---


def test_geocoordinate_creation_and_validation():
    """Tests the GeoCoordinate class constructor and its built-in validation."""
    # Why: The core value object must be reliable.

    # Valid creation
    coord = GeoCoordinate(lat=-23.55, lon=-46.63)
    assert coord.lat == -23.55
    assert coord.lon == -46.63

    # Invalid creation
    with pytest.raises(ValueError):
        GeoCoordinate(lat=91, lon=0)
    with pytest.raises(ValueError):
        GeoCoordinate(lat=0, lon=-181)


def test_geocoordinate_from_methods():
    """Tests the various `from_*` class methods for creating GeoCoordinate instances."""
    # Why: These convenience methods are part of the class's public API.

    # from_decimal
    coord1 = GeoCoordinate.from_decimal(-23.55, -46.63)
    assert coord1.lat == -23.55 and coord1.lon == -46.63

    # from_strings (auto-detection)
    coord2 = GeoCoordinate.from_strings('23°33\'0" S', '46°37\'48" W')
    assert coord2.lat == pytest.approx(-23.55)
    assert coord2.lon == pytest.approx(-46.63)

    # from_dms_str
    coord3 = GeoCoordinate.from_dms_str('23°33\'0" S', '46°37\'48" W')
    assert coord2.lat == coord3.lat and coord2.lon == coord3.lon

    # from_ddm_str
    coord4 = GeoCoordinate.from_ddm_str("23°33.0' S", "46°37.8' W")
    assert coord4.lat == pytest.approx(-23.55)
    assert coord4.lon == pytest.approx(-46.63)

    # from_dd_str
    coord5 = GeoCoordinate.from_dd_str("23.55 S", "-46.63")
    assert coord5.lat == pytest.approx(-23.55)
    assert coord5.lon == pytest.approx(-46.63)


def test_geocoordinate_to_methods_roundtrip():
    """
    Tests that converting to a string format and back results in the original value.
    """
    # Why: Ensures that formatting and parsing are inverse operations.
    original_coord = GeoCoordinate(-23.55, -46.63)

    # Test DMS roundtrip
    lat_dms, lon_dms = original_coord.to_dms_str(second_precision=2)
    rt_coord_dms = GeoCoordinate.from_strings(lat_dms, lon_dms)
    assert rt_coord_dms.lat == pytest.approx(original_coord.lat, abs=1e-5)
    assert rt_coord_dms.lon == pytest.approx(original_coord.lon, abs=1e-5)

    # Test DDM roundtrip
    lat_ddm, lon_ddm = original_coord.to_ddm_str(minute_precision=4)
    rt_coord_ddm = GeoCoordinate.from_strings(lat_ddm, lon_ddm)
    assert rt_coord_ddm.lat == pytest.approx(original_coord.lat, abs=1e-6)
    assert rt_coord_ddm.lon == pytest.approx(original_coord.lon, abs=1e-6)

    # Test DD roundtrip
    lat_dd, lon_dd = original_coord.to_dd_str(lat_precision=6, lon_precision=6)
    rt_coord_dd = GeoCoordinate.from_strings(lat_dd, lon_dd)
    assert rt_coord_dd.lat == pytest.approx(original_coord.lat, abs=1e-6)
    assert rt_coord_dd.lon == pytest.approx(original_coord.lon, abs=1e-6)


# --- Test high-level normalize_coord_input function ---


def test_normalize_coord_input_various_forms():
    """
    Tests the main `normalize_coord_input` function with all its accepted input formats.
    """
    # Why: This is the primary entry point for users, so its flexibility is paramount.
    expected = GeoCoordinate(-23.55, -46.63)

    # Form 1: GeoCoordinate object
    assert normalize_coord_input(expected) == expected

    # Form 2: Tuple/list of numbers
    assert normalize_coord_input((-23.55, -46.63)) == expected
    assert normalize_coord_input([-23.55, -46.63]) == expected

    # Form 3: Tuple/list of strings
    assert normalize_coord_input(("23.55 S", "-46.63")) == expected

    # Form 4: Separate arguments (numbers)
    assert normalize_coord_input(-23.55, -46.63) == expected

    # Form 5: Separate arguments (strings)
    assert normalize_coord_input('23°33\'0" S', '46°37\'48.00" W') == expected


def test_normalize_coord_input_invalid():
    """
    Tests that `normalize_coord_input` raises TypeError for unsupported input shapes.
    """
    # Why: Catch misuse of the API early and clearly.
    with pytest.raises(TypeError):
        normalize_coord_input("just one string")  # Missing lon
    with pytest.raises(TypeError):
        normalize_coord_input(("a", "b", "c"))  # Wrong tuple length
