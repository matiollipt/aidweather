# geo — Coordinate utilities

The `geo` module parses, validates, and converts geographic coordinates. It accepts DMS, DDM, and decimal degree string formats, as well as raw numeric values, and normalizes them into a single typed object.

For a complete list of coordinate classes, parsers, formatters, and internal
helpers, see the [API Inventory](api_inventory.md#aidweathergeo).

---

## GeoCoordinate

An immutable value object representing a lat/lon pair in decimal degrees.

```python
from aidweather import GeoCoordinate

# From decimal degrees
coord = GeoCoordinate.from_decimal(-23.55, -46.63)
print(coord.lat, coord.lon)   # -23.55, -46.63

# From a DMS string (auto-detects format)
coord = GeoCoordinate.from_strings("23°33'0\" S", "46°37'48\" W")

# From DDM strings
coord = GeoCoordinate.from_ddm_str("23°33.0' S", "46°37.8' W")

# Convert back to string formats
print(coord.to_dms_str())   # ('23°33\'0" S', '46°37\'48" W')
print(coord.to_ddm_str())   # ('23°33.000\' S', '46°37.800\' W')
print(coord.to_dd_str())    # ('23.55000° S', '46.63000° W')

# Get raw decimal tuple
lat, lon = coord.as_decimal()
```

Coordinates are validated on creation — out-of-range values raise `ValueError` immediately.

---

## normalize_coord_input

High-level helper that accepts almost any reasonable coordinate input and returns a `GeoCoordinate`. Use this when your code needs to handle inputs from different sources.

```python
from aidweather import normalize_coord_input

# From a tuple of floats
coord = normalize_coord_input((-23.55, -46.63))

# From two separate numbers
coord = normalize_coord_input(-23.55, -46.63)

# From a tuple of strings (any format)
coord = normalize_coord_input(("23°33'0\" S", "46°37'48\" W"))

# Already a GeoCoordinate — returned as-is
coord = normalize_coord_input(existing_coord)
```

---

## Supported string formats

| Format | Example |
|---|---|
| Decimal Degrees (DD) | `"-23.55"`, `"23.55° S"` |
| Degrees Decimal Minutes (DDM) | `"23°33.0' S"` |
| Degrees Minutes Seconds (DMS) | `"23°33'0\" S"` |

The parser handles Unicode degree variants (`º`, `˚`), smart quotes, and double-apostrophe seconds (`''`).
