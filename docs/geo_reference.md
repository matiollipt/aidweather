# Geospatial Coordinate Reference — `aidweather.geo`

This document covers the coordinate representation model, format conventions, parsing rules, and the complete public API of the `aidweather.geo` module.

For spatial interpretation of how NASA POWER maps coordinates to grid cells, see the [Spatial Interpretation Guide](spatial_interpretation.md).
For how `GeoCoordinate` fits into the overall request lifecycle, see the [Developer & Architecture Guide](developer_guide.md).
For full Python API surface, see the [API Reference](api_reference.md).

---

## 1. Background: Coordinate Notation Systems

Geographic coordinates locate points on the surface of the Earth using two angles: **latitude** (north–south position) and **longitude** (east–west position). Three notation systems are in common use in scientific and field contexts:

### 1.1 Decimal Degrees (DD)

The canonical numerical form. A single floating-point number expresses the full angular value, with the fractional part representing arc-fractions of a degree. Sign encodes hemisphere: positive = North or East, negative = South or West.

```
-23.55°     → 23.55° South latitude
-46.6333°   → 46.6333° West longitude
```

DD is the format required by most geospatial APIs, databases, and GIS systems, including NASA POWER. It is the **internal representation used throughout `aidweather`**.

### 1.2 Degrees, Decimal Minutes (DDM)

Degrees are expressed as a whole integer; the fractional remainder is carried as decimal minutes (1° = 60 minutes). Commonly used in marine navigation, GPS device outputs, and some regulatory reporting formats.

```
23° 33.0' S     → lat  -23.55
46° 37.998' W   → lon  -46.6333
```

Conversion:
```
DD = sign × (|degrees| + decimal_minutes / 60)
```

### 1.3 Degrees, Minutes, Seconds (DMS)

The classical cartographic and surveying notation. The fractional remainder is split into whole minutes and decimal (or whole) seconds (1 minute = 60 seconds). Found in historical maps, land surveys, astronomical ephemerides, and many field GPS receivers.

```
23° 33' 00" S   → lat  -23.55
46° 37' 59.88" W → lon  -46.6333
```

Conversion:
```
DD = sign × (|degrees| + minutes / 60 + seconds / 3600)
```

> [!NOTE]
> **Why three systems?** Each arose in a different professional tradition. DD emerged with digital computing and GIS. DDM was the natural output of 20th-century marine GPS units and LORAN receivers. DMS traces back to ancient astronomical and survey traditions. Field scientists often receive coordinates from instruments or collaborators in any of these formats; `aidweather.geo` accepts all three transparently.

---

## 2. Internal Coordinate Model

`aidweather.geo` uses a single, unambiguous internal convention:

| Axis | Range | Positive direction |
|:---|:---|:---|
| Latitude | −90.0 to +90.0 | North |
| Longitude | −180.0 to +180.0 | East |

This is the standard mathematical convention used by NASA POWER, WGS-84, and virtually all modern geospatial frameworks. No degree-only, gradian, or radian variants are accepted.

> [!IMPORTANT]
> `aidweather` does **not** accept coordinates in DMS integer tuples (e.g., `(23, 33, 0, "S")`), MGRS grid references, UTM zones, or any other non-geographic projection. Convert to DD, DDM, or DMS string format before passing coordinates to the library.

---

## 3. `GeoCoordinate` — The Canonical Coordinate Object

```python
@dataclass(frozen=True)
class GeoCoordinate:
    lat: float
    lon: float
```

`GeoCoordinate` is a **frozen dataclass**: it is immutable after construction, hashable, and safe to use as a dictionary key or in sets. Range validation (`lat ∈ [−90, 90]`, `lon ∈ [−180, 180]`) runs automatically in `__post_init__`.

### 3.1 Construction

All entry points produce a validated `GeoCoordinate`. Pick the constructor that matches your input format:

#### `from_decimal(lat, lon)` — From plain numbers

The simplest path. Accepts `int` or `float`.

```python
from aidweather import GeoCoordinate

coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
# GeoCoordinate(lat=-23.55, lon=-46.633333)
```

#### `from_strings(lat_str, lon_str)` — Auto-detecting string format

Tries DMS, then DDM, then DD parsers in order. Use when the input format is unknown or mixed.

```python
coord = GeoCoordinate.from_strings("23° 33' 00\" S", "46° 37' 59.88\" W")
coord = GeoCoordinate.from_strings("23° 33.0' S", "46° 37.998' W")
coord = GeoCoordinate.from_strings("23.55° S", "46.6333° W")
```

All three calls produce `GeoCoordinate(lat=-23.55, lon=-46.6333)`.

#### `from_dd_str(lat_str, lon_str)` — Explicit DD string parser

Use when the input is known to be Decimal Degrees. Rejects DDM or DMS formats.

```python
coord = GeoCoordinate.from_dd_str("23.55° S", "46.6333° W")
coord = GeoCoordinate.from_dd_str("-23.55", "-46.6333")  # sign-only, no hemisphere letter
```

#### `from_ddm_str(lat_str, lon_str)` — Explicit DDM string parser

```python
coord = GeoCoordinate.from_ddm_str("23° 33.0' S", "46° 37.998' W")
```

#### `from_dms_str(lat_str, lon_str)` — Explicit DMS string parser

```python
coord = GeoCoordinate.from_dms_str("23° 33' 00\" S", "46° 37' 59.88\" W")
```

### 3.2 Retrieval and Formatting

#### `as_decimal()` / `to_dd()` → `tuple[float, float]`

Returns `(lat, lon)` as a plain Python tuple of floats. `to_dd()` is an alias for `as_decimal()`.

```python
coord.as_decimal()   # (-23.55, -46.633333)
coord.to_dd()        # (-23.55, -46.633333)
```

#### `to_dd_str(lat_precision=5, lon_precision=5)` → `tuple[str, str]`

Formats coordinates as hemisphere-annotated DD strings. Precision controls the number of decimal digits in each component.

```python
coord.to_dd_str()
# ('23.55000° S', '46.63333° W')

coord.to_dd_str(lat_precision=2, lon_precision=2)
# ('23.55° S', '46.63° W')
```

#### `to_ddm_str(minute_precision=3)` → `tuple[str, str]`

Formats as DDM strings. `minute_precision` controls the decimal places on the minutes component.

```python
coord.to_ddm_str()
# ('23°33.000' S', '46°37.998' W')
```

#### `to_dms_str(second_precision=0)` → `tuple[str, str]`

Formats as DMS strings. `second_precision=0` returns whole seconds; increase for sub-second precision.

```python
coord.to_dms_str()
# ('23°33'0" S', '46°37'60" W')

coord.to_dms_str(second_precision=2)
# ('23°33'0.00" S', '46°37'59.88" W')
```

---

## 4. Module-Level Utilities

### 4.1 `normalize_coord_input(lat, lon=None)` — Flexible entry point

The primary intake function used internally by `PowerClient`. Accepts any of:

- an existing `GeoCoordinate` (returned unchanged),
- a `(lat, lon)` tuple or list of two elements (numbers or parseable strings),
- two separate `lat` and `lon` arguments (numbers or parseable strings).

```python
from aidweather.geo import normalize_coord_input, GeoCoordinate

# All equivalent:
c1 = normalize_coord_input(-23.55, -46.63)
c2 = normalize_coord_input((-23.55, -46.63))
c3 = normalize_coord_input(GeoCoordinate.from_decimal(-23.55, -46.63))
c4 = normalize_coord_input("23.55° S", "46.63° W")
c5 = normalize_coord_input(("23° 33.0' S", "46° 37.998' W"))

assert c1 == c2 == c3 == c4 == c5
```

> [!TIP]
> `normalize_coord_input` is the right function to call from downstream code that needs to accept any coordinate form from users or external data sources. You do not need to know the input format in advance.

**Raises `TypeError`** if the arguments do not match any accepted form (e.g., a single string that is not a tuple).

### 4.2 `parse_dd(s, is_lat)` → `float`

Parses a single Decimal Degrees string to a signed float.

```python
from aidweather.geo import parse_dd

parse_dd("23.55° S", is_lat=True)   # -23.55
parse_dd("-23.55", is_lat=True)     # -23.55
parse_dd("46.63° W", is_lat=False)  # -46.63
```

The `is_lat` flag determines which hemisphere letters (`N`/`S` vs. `E`/`W`) are valid and whether the sign convention is applied correctly.

**Raises `ValueError`** if the string does not match the DD pattern.

### 4.3 `parse_ddm(s, is_lat)` → `float`

Parses a single DDM string.

```python
from aidweather.geo import parse_ddm

parse_ddm("23° 33.0' S", is_lat=True)    # -23.55
parse_ddm("46° 37.998' W", is_lat=False) # -46.6333
```

> [!WARNING]
> If a DDM string carries **both** a leading negative sign on the degrees component **and** a hemisphere letter (e.g., `"-23° 33.0' S"`), `parse_ddm` emits a `UserWarning` and uses the hemisphere letter as the authoritative sign indicator, discarding the leading minus sign. This matches the behavior of most GPS devices and chart-plotting software, where the hemisphere letter is the primary sign specifier.

### 4.4 `parse_dms(s, is_lat)` → `float`

Parses a single DMS string. The same ambiguous-sign `UserWarning` policy as `parse_ddm` applies.

```python
from aidweather.geo import parse_dms

parse_dms("23° 33' 0\" S", is_lat=True)      # -23.55
parse_dms("46° 37' 59.88\" W", is_lat=False) # -46.6333
```

### 4.5 `parse_any_coord_string(s, is_lat)` → `float`

Attempts DMS, DDM, then DD parsers in sequence; raises `ValueError` if none match. This is the function called internally by `GeoCoordinate.from_strings`.

```python
from aidweather.geo import parse_any_coord_string

parse_any_coord_string("23° 33' 0\" S", is_lat=True)  # DMS → -23.55
parse_any_coord_string("23° 33.0' S",  is_lat=True)   # DDM → -23.55
parse_any_coord_string("23.55° S",     is_lat=True)   # DD  → -23.55
```

### 4.6 `decimal_to_ddm_components(value, is_lat)` → `tuple[int, float, str]`

Decomposes a signed decimal degree value into `(degrees, decimal_minutes, hemisphere)`. Useful when you need to format or display DDM without the full formatting pipeline of `to_ddm_str`.

```python
from aidweather.geo import decimal_to_ddm_components

decimal_to_ddm_components(-23.55, is_lat=True)
# (23, 33.0, 'S')
```

### 4.7 `decimal_to_dms_components(value, is_lat)` → `tuple[int, int, float, str]`

Decomposes a signed decimal degree value into `(degrees, minutes, seconds, hemisphere)`.

```python
from aidweather.geo import decimal_to_dms_components

decimal_to_dms_components(-23.55, is_lat=True)
# (23, 33, 0.0, 'S')

decimal_to_dms_components(-46.633333, is_lat=False)
# (46, 37, 59.9988, 'W')
```

---

## 5. Accepted String Formats and Unicode Variants

The internal normalizer (`_normalize_coord_string`) handles a wide range of real-world coordinate string variants before regex matching:

| Variant | Accepted? | Notes |
|:---|:---|:---|
| `23.55° S` | ✅ | Standard DD with degree symbol and hemisphere |
| `23.55 S` | ✅ | DD without degree symbol |
| `-23.55` | ✅ | DD sign-only, no hemisphere letter |
| `23° 33.0' S` | ✅ | DDM with single-quote minutes |
| `23° 33′ 0″ S` | ✅ | DMS with Unicode prime/double-prime symbols |
| `23°33'0"S` | ✅ | DMS without spaces |
| `23° 33' 00'' S` | ✅ | Double apostrophe for seconds (converted to `"`) |
| `23º 33' 0" S` | ✅ | Masculine ordinal indicator `º` instead of `°` |
| `23˚ 33' 0" S` | ✅ | Ring above `˚` instead of `°` |
| `23° 33' 0" s` | ✅ | Lowercase hemisphere letter |
| `23°33'0"X` | ❌ | Invalid hemisphere letter |
| `(23, 33, 0, "S")` | ❌ | Tuple form not accepted by string parsers |

> [!NOTE]
> The Unicode normalization step silently replaces visually similar characters (ordinal indicator `º`, ring above `˚`, Unicode primes `′` `″`, full-width apostrophe `＇`, curly quotes) with their ASCII equivalents **before** parsing. This covers common copy-paste artefacts from PDF documents, instrument exports, and web-scraped tables.

---

## 6. Ambiguous Input Policies

### 6.1 Negative degrees with hemisphere letter

Both `parse_ddm` and `parse_dms` accept strings like `"-23° 33.0' S"`. In this case the leading minus is **discarded** and a `UserWarning` is emitted:

```
UserWarning: Coordinate string '"-23° 33.0' S"' has both a negative degree
value and a hemisphere letter 'S'. The negative sign is ignored; the
hemisphere letter takes precedence.
```

The hemisphere letter is the authoritative sign specifier. This policy matches the behavior of GPS firmware and chart-plotting applications and prevents silent double-negation errors.

### 6.2 Hemisphere letter absent

If no hemisphere letter is present, the sign of the numeric component is trusted:

- DD: sign is taken as-is.
- DDM: positive degrees → positive decimal result.
- DMS: positive degrees → positive decimal result.

The `is_lat` parameter is used only to validate that the hemisphere letter, *if present*, is appropriate for the axis (N/S for latitude, E/W for longitude).

### 6.3 Parser priority in `parse_any_coord_string`

The auto-detecting parser tries in the order **DMS → DDM → DD**. This order is deliberate: DMS is the most specific format (three numeric groups plus optional hemisphere), DDM the next, and DD the least specific. A string that matches DMS will never be mis-parsed as DD.

---

## 7. Error Reference

| Situation | Exception | Source |
|:---|:---|:---|
| Latitude outside [−90, 90] | `ValueError` | `GeoCoordinate.__post_init__` |
| Longitude outside [−180, 180] | `ValueError` | `GeoCoordinate.__post_init__` |
| String does not match expected format | `ValueError` | `parse_dd`, `parse_ddm`, `parse_dms` |
| No parser matched in auto-detect | `ValueError` | `parse_any_coord_string` |
| Wrong hemisphere letter for axis | `ValueError` | `_apply_hemisphere_sign` |
| Invalid argument combination | `TypeError` | `normalize_coord_input` |
| Ambiguous sign + hemisphere letter | `UserWarning` | `parse_ddm`, `parse_dms` |

---

## 8. Usage in `PowerClient`

All `PowerClient` methods that accept geographic coordinates route through `normalize_coord_input`. This means you can pass raw numbers, tuples, strings, or a `GeoCoordinate` directly:

```python
from aidweather import PowerClient, GeoCoordinate

client = PowerClient(temporal_api="daily")

# All equivalent ways to specify the same point:
df = client.get_point_data(lat=-23.55, lon=-46.63, ...)
df = client.get_point_data(lat=(-23.55, -46.63), ...)
df = client.get_point_data(lat="23° 33' 0\" S", lon="46° 38' 0\" W", ...)
df = client.get_point_data(lat=GeoCoordinate.from_decimal(-23.55, -46.63), ...)
```

> [!IMPORTANT]
> **Grid-cell resolution reminder**: The coordinate you pass is mapped by NASA POWER to the containing source product grid cell. For MERRA-2 parameters (`T2M`, `PRECTOTCORR`, etc.), the native grid is 0.5° Latitude × 0.625° Longitude. For CERES solar radiation parameters (`ALLSKY_SFC_SW_DWN`), the native grid is 1.0° × 1.0°. Coordinates that differ by less than the native grid spacing will return identical data. See the [Spatial Interpretation Guide](spatial_interpretation.md) for details.

---

## 9. Quick Reference Table

| Task | Recommended call |
|:---|:---|
| Construct from decimal numbers | `GeoCoordinate.from_decimal(lat, lon)` |
| Construct from unknown string format | `GeoCoordinate.from_strings(lat_str, lon_str)` |
| Construct from known DD strings | `GeoCoordinate.from_dd_str(lat_str, lon_str)` |
| Construct from known DDM strings | `GeoCoordinate.from_ddm_str(lat_str, lon_str)` |
| Construct from known DMS strings | `GeoCoordinate.from_dms_str(lat_str, lon_str)` |
| Accept any coordinate form from user/external data | `normalize_coord_input(lat, lon)` |
| Get `(lat, lon)` float tuple | `coord.as_decimal()` |
| Format as DD strings with hemisphere | `coord.to_dd_str()` |
| Format as DDM strings | `coord.to_ddm_str()` |
| Format as DMS strings | `coord.to_dms_str()` |
| Parse a single DD string | `parse_dd(s, is_lat)` |
| Parse a single DDM string | `parse_ddm(s, is_lat)` |
| Parse a single DMS string | `parse_dms(s, is_lat)` |
| Parse a string of unknown format | `parse_any_coord_string(s, is_lat)` |
| Get DDM numeric components | `decimal_to_ddm_components(value, is_lat)` |
| Get DMS numeric components | `decimal_to_dms_components(value, is_lat)` |

---

## 10. Public API Symbol Table

The following symbols are exported from `aidweather.geo` (and re-exported from `aidweather`):

| Symbol | Type | Summary |
|:---|:---|:---|
| `GeoCoordinate` | `dataclass` | Immutable `(lat, lon)` value object in decimal degrees |
| `normalize_coord_input` | `function` | Accepts any coordinate form; returns `GeoCoordinate` |
| `decimal_to_ddm_components` | `function` | Decomposes DD to `(deg, min, hem)` |
| `decimal_to_dms_components` | `function` | Decomposes DD to `(deg, min, sec, hem)` |
| `parse_dd` | `function` | Parses DD string → signed float |
| `parse_ddm` | `function` | Parses DDM string → signed float |
| `parse_dms` | `function` | Parses DMS string → signed float |
| `parse_any_coord_string` | `function` | Auto-detecting string parser → signed float |
