# Geospatial Coordinate Reference — `aidweather.geo`

This reference covers the coordinate representation model, format conventions, parsing rules, and the complete public API surface of the `aidweather.geo` module.

For spatial guidance on how NASA POWER maps coordinates to grid cells, see the [Spatial Interpretation Guide](spatial_interpretation.md).  
For details on how `GeoCoordinate` integrates into internal request lifecycles, see the [Developer & Architecture Guide](developer_guide.md).  
For the top-level Python API overview, see the [API Reference](api_reference.md).

---

## 1. Background: Coordinate Notation Systems

Geographic coordinates identify locations on the surface of the Earth using two angular values: **latitude** (north–south position) and **longitude** (east–west position). Three primary notation systems are widely used across scientific research, spatial data, and field instruments:

### 1.1 Decimal Degrees (DD)

The canonical numerical representation. A single floating-point number expresses the complete angle, where the fractional part represents sub-degree fractions. The sign indicates hemisphere: positive values denote North or East, while negative values denote South or West.

```
-23.55°     → 23.55° South latitude
-46.6333°   → 46.6333° West longitude
```

Decimal Degrees is the required format for most geospatial APIs, spatial databases, and GIS software, including NASA POWER. It serves as the **internal canonical representation across `aidweather`**.

### 1.2 Degrees, Decimal Minutes (DDM)

Degrees are represented as integers, while the fractional remainder is expressed in decimal minutes (where 1° = 60 minutes). DDM is commonly produced by marine GPS receivers, navigational logs, and regulatory spatial reports.

```
23° 33.0' S     → lat  -23.55
46° 37.998' W   → lon  -46.6333
```

Conversion formula:
```
DD = sign × (|degrees| + decimal_minutes / 60)
```

### 1.3 Degrees, Minutes, Seconds (DMS)

The classical cartographic notation system. The fractional remainder is divided into whole minutes and decimal (or integer) seconds (where 1 minute = 60 seconds). DMS is frequently found in historical maps, land surveys, astronomical tables, and handheld GPS displays.

```
23° 33' 00" S   → lat  -23.55
46° 37' 59.88" W → lon  -46.6333
```

Conversion formula:
```
DD = sign × (|degrees| + minutes / 60 + seconds / 3600)
```

> [!NOTE]
> **Why support three notation systems?** Each system developed within a specific technical domain—DD with modern digital GIS, DDM with marine GPS instrumentation, and DMS with traditional surveying. Because field scientists collect coordinates from varied instruments and collaborators, `aidweather.geo` parses and converts all three transparently.

---

## 2. Internal Coordinate Model

`aidweather.geo` uses a single, unambiguous internal representation:

| Axis | Range | Positive Direction |
| :--- | :--- | :--- |
| Latitude | −90.0 to +90.0 | North |
| Longitude | −180.0 to +180.0 | East |

This matches the standard mathematical convention used by NASA POWER, WGS-84, and modern geospatial software. Non-geographic coordinate projections (such as UTM or MGRS) are not accepted directly.

> [!IMPORTANT]
> `aidweather` does **not** accept coordinate tuples in non-standard structures (such as `(23, 33, 0, "S")`), MGRS grid references, or projected coordinate systems. Convert such coordinates to standard DD, DDM, or DMS string formats before initializing `GeoCoordinate`.

---

## 3. `GeoCoordinate` — The Canonical Coordinate Object

```python
@dataclass(frozen=True)
class GeoCoordinate:
    lat: float
    lon: float
```

`GeoCoordinate` is an **immutable frozen dataclass**: once created, its attributes cannot be modified. It is hashable and safe to use as a dictionary key or set element. Coordinate bounds validation (`lat ∈ [−90, 90]`, `lon ∈ [−180, 180]`) runs automatically during initialization (`__post_init__`).

### 3.1 Construction Methods

All constructors return a validated `GeoCoordinate` instance. Select the factory method matching your input data format:

#### `from_decimal(lat, lon)` — From numerical values

The simplest constructor. Accepts numeric `float` or `int` values.

```python
from aidweather import GeoCoordinate

coord = GeoCoordinate.from_decimal(-23.55, -46.633333)
# GeoCoordinate(lat=-23.55, lon=-46.633333)
```

#### `from_strings(lat_str, lon_str)` — Auto-detecting string parser

Attempts DMS, DDM, and DD string parsers in sequence. Ideal when input formats are unknown or heterogeneous.

```python
coord = GeoCoordinate.from_strings("23° 33' 00\" S", "46° 37' 59.88\" W")
coord = GeoCoordinate.from_strings("23° 33.0' S", "46° 37.998' W")
coord = GeoCoordinate.from_strings("23.55° S", "46.6333° W")
```

All three calls resolve to `GeoCoordinate(lat=-23.55, lon=-46.6333)`.

#### `from_dd_str(lat_str, lon_str)` — Explicit DD string parser

Parses Decimal Degrees strings explicitly, rejecting DDM or DMS formats.

```python
coord = GeoCoordinate.from_dd_str("23.55° S", "46.6333° W")
coord = GeoCoordinate.from_dd_str("-23.55", "-46.6333")  # Signed DD without hemisphere letter
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

Returns `(lat, lon)` as a tuple of floating-point numbers. `to_dd()` is a direct alias for `as_decimal()`.

```python
coord.as_decimal()   # (-23.55, -46.633333)
coord.to_dd()        # (-23.55, -46.633333)
```

#### `to_dd_str(lat_precision=5, lon_precision=5)` → `tuple[str, str]`

Formats coordinates as hemisphere-annotated Decimal Degrees strings. Precision controls decimal places.

```python
coord.to_dd_str()
# ('23.55000° S', '46.63333° W')

coord.to_dd_str(lat_precision=2, lon_precision=2)
# ('23.55° S', '46.63° W')
```

#### `to_ddm_str(minute_precision=3)` → `tuple[str, str]`

Formats coordinates as Degrees, Decimal Minutes strings. `minute_precision` controls minute decimal places.

```python
coord.to_ddm_str()
# ('23°33.000' S', '46°37.998' W')
```

#### `to_dms_str(second_precision=0)` → `tuple[str, str]`

Formats coordinates as Degrees, Minutes, Seconds strings. `second_precision=0` rounds to whole seconds.

```python
coord.to_dms_str()
# ('23°33'0" S', '46°37'60" W')

coord.to_dms_str(second_precision=2)
# ('23°33'0.00" S', '46°37'59.88" W')
```

---

## 4. Module-Level Utilities

### 4.1 `normalize_coord_input(lat, lon=None)` — Flexible coordinate intake

The central coordinate normalization helper used internally by `PowerClient`. Accepts any of the following:

- An existing `GeoCoordinate` instance (returned unchanged),
- A `(lat, lon)` tuple or two-element list of numbers or parseable strings,
- Two separate `lat` and `lon` arguments (numbers or parseable strings).

```python
from aidweather.geo import normalize_coord_input, GeoCoordinate

# All produce identical GeoCoordinate objects:
c1 = normalize_coord_input(-23.55, -46.63)
c2 = normalize_coord_input((-23.55, -46.63))
c3 = normalize_coord_input(GeoCoordinate.from_decimal(-23.55, -46.63))
c4 = normalize_coord_input("23.55° S", "46.63° W")
c5 = normalize_coord_input(("23° 33.0' S", "46° 37.998' W"))

assert c1 == c2 == c3 == c4 == c5
```

> [!TIP]
> Use `normalize_coord_input` in external applications to accept flexible user coordinate inputs without needing prior knowledge of the input format.

Raises `TypeError` if the argument combination cannot be resolved to a valid coordinate.

### 4.2 `parse_dd(s, is_lat)` → `float`

Parses a single Decimal Degrees string into a signed floating-point number.

```python
from aidweather.geo import parse_dd

parse_dd("23.55° S", is_lat=True)   # -23.55
parse_dd("-23.55", is_lat=True)     # -23.55
parse_dd("46.63° W", is_lat=False)  # -46.63
```

The `is_lat` flag specifies axis context to validate appropriate hemisphere letters (`N`/`S` vs. `E`/`W`).

Raises `ValueError` if the string pattern does not match Decimal Degrees format.

### 4.3 `parse_ddm(s, is_lat)` → `float`

Parses a single DDM string into a signed float.

```python
from aidweather.geo import parse_ddm

parse_ddm("23° 33.0' S", is_lat=True)    # -23.55
parse_ddm("46° 37.998' W", is_lat=False) # -46.6333
```

> [!WARNING]
> If a DDM string contains **both** a leading negative sign and a hemisphere letter (e.g. `"-23° 33.0' S"`), `parse_ddm` issues a `UserWarning` and prioritizes the hemisphere letter as the authoritative sign indicator, discarding the negative sign.

### 4.4 `parse_dms(s, is_lat)` → `float`

Parses a single DMS string into a signed float, applying the same hemisphere sign precedence policy as `parse_ddm`.

```python
from aidweather.geo import parse_dms

parse_dms("23° 33' 0\" S", is_lat=True)      # -23.55
parse_dms("46° 37' 59.88\" W", is_lat=False) # -46.6333
```

### 4.5 `parse_any_coord_string(s, is_lat)` → `float`

Tries DMS, DDM, and DD parsers in sequence, returning the parsed float value or raising `ValueError` if no format matches.

```python
from aidweather.geo import parse_any_coord_string

parse_any_coord_string("23° 33' 0\" S", is_lat=True)  # DMS → -23.55
parse_any_coord_string("23° 33.0' S",  is_lat=True)   # DDM → -23.55
parse_any_coord_string("23.55° S",     is_lat=True)   # DD  → -23.55
```

### 4.6 `decimal_to_ddm_components(value, is_lat)` → `tuple[int, float, str]`

Decomposes a signed decimal degree number into `(degrees, decimal_minutes, hemisphere)`.

```python
from aidweather.geo import decimal_to_ddm_components

decimal_to_ddm_components(-23.55, is_lat=True)
# (23, 33.0, 'S')
```

### 4.7 `decimal_to_dms_components(value, is_lat)` → `tuple[int, int, float, str]`

Decomposes a signed decimal degree number into `(degrees, minutes, seconds, hemisphere)`.

```python
from aidweather.geo import decimal_to_dms_components

decimal_to_dms_components(-23.55, is_lat=True)
# (23, 33, 0.0, 'S')

decimal_to_dms_components(-46.633333, is_lat=False)
# (46, 37, 59.9988, 'W')
```

---

## 5. Accepted String Formats and Unicode Variants

The internal normalizer (`_normalize_coord_string`) handles a wide variety of real-world string variations before regex evaluation:

| Variant | Accepted? | Notes |
| :--- | :--- | :--- |
| `23.55° S` | ✅ | Standard DD with degree symbol and hemisphere letter |
| `23.55 S` | ✅ | DD without degree symbol |
| `-23.55` | ✅ | DD signed value without hemisphere letter |
| `23° 33.0' S` | ✅ | DDM with single-quote minute symbol |
| `23° 33′ 0″ S` | ✅ | DMS with Unicode prime/double-prime symbols |
| `23°33'0"S` | ✅ | DMS without spaces |
| `23° 33' 00'' S` | ✅ | Double apostrophes for seconds (converted to `"`) |
| `23º 33' 0" S` | ✅ | Masculine ordinal indicator `º` instead of `°` |
| `23˚ 33' 0" S` | ✅ | Ring above symbol `˚` instead of `°` |
| `23° 33' 0" s` | ✅ | Lowercase hemisphere letter |
| `23°33'0"X` | ❌ | Invalid hemisphere letter |
| `(23, 33, 0, "S")` | ❌ | Tuple structures not accepted by string parsers |

> [!NOTE]
> Prior to parsing, visual character variations (such as ordinal indicators `º`, ring symbols `˚`, Unicode primes `′` `″`, and curly quotes) are normalized to standard ASCII characters. This ensures reliable copy-pasting from PDFs, tables, and web sources.

---

## 6. Ambiguous Input Policies

### 6.1 Negative degrees with hemisphere letter

When given strings like `"-23° 33.0' S"`, `parse_ddm` and `parse_dms` ignore the leading minus sign, issue a `UserWarning`, and use the hemisphere letter (`S`) to determine orientation. This prevents accidental double-negation errors.

### 6.2 Omitted hemisphere letters

When hemisphere letters are omitted, the numerical sign is preserved directly:
- DD: Numeric sign is used as-is.
- DDM: Positive numeric degrees yield positive results.
- DMS: Positive numeric degrees yield positive results.

The `is_lat` argument validates that hemisphere letters, when present, match the axis (N/S for latitude, E/W for longitude).

### 6.3 Parser priority in `parse_any_coord_string`

The multi-format string parser evaluates in strict order: **DMS → DDM → DD**. Evaluating from most specific to least specific guarantees that DMS strings are never partially matched as DD values.

---

## 7. Error Reference

| Situation | Exception | Source |
| :--- | :--- | :--- |
| Latitude outside [−90, 90] | `ValueError` | `GeoCoordinate.__post_init__` |
| Longitude outside [−180, 180] | `ValueError` | `GeoCoordinate.__post_init__` |
| Unrecognized coordinate string format | `ValueError` | `parse_dd`, `parse_ddm`, `parse_dms` |
| No parser match in auto-detect | `ValueError` | `parse_any_coord_string` |
| Mismatched hemisphere letter for axis | `ValueError` | `_apply_hemisphere_sign` |
| Invalid input types | `TypeError` | `normalize_coord_input` |
| Ambiguous sign with hemisphere letter | `UserWarning` | `parse_ddm`, `parse_dms` |

---

## 8. Usage in `PowerClient`

All `PowerClient` functions accepting geographic coordinates pass inputs through `normalize_coord_input`. You can supply coordinates as numbers, tuples, strings, or `GeoCoordinate` objects:

```python
from aidweather import PowerClient, GeoCoordinate

client = PowerClient(temporal_api="daily")

# All equivalent coordinate inputs:
df = client.get_point_data(lat=-23.55, lon=-46.63, start="2023-01-01", end="2023-01-31", params=["T2M"])
df = client.get_point_data(lat=(-23.55, -46.63), start="2023-01-01", end="2023-01-31", params=["T2M"])
df = client.get_point_data(lat="23° 33' 0\" S", lon="46° 38' 0\" W", start="2023-01-01", end="2023-01-31", params=["T2M"])
df = client.get_point_data(lat=GeoCoordinate.from_decimal(-23.55, -46.63), start="2023-01-01", end="2023-01-31", params=["T2M"])
```

> [!IMPORTANT]
> **Spatial Grid Resolution Note**: NASA POWER maps requested coordinates to the enclosing source product grid cell. For MERRA-2 meteorological parameters (`T2M`, `PRECTOTCORR`), native cells are 0.5° × 0.625°. For CERES solar radiation (`ALLSKY_SFC_SW_DWN`), native cells are 1.0° × 1.0°. Coordinates within the same source cell return identical series. See the [Spatial Interpretation Guide](spatial_interpretation.md) for details.

---

## 9. Quick Reference Table

| Task | Recommended Call |
| :--- | :--- |
| Construct from decimal numbers | `GeoCoordinate.from_decimal(lat, lon)` |
| Construct from unknown string format | `GeoCoordinate.from_strings(lat_str, lon_str)` |
| Construct from known DD strings | `GeoCoordinate.from_dd_str(lat_str, lon_str)` |
| Construct from known DDM strings | `GeoCoordinate.from_ddm_str(lat_str, lon_str)` |
| Construct from known DMS strings | `GeoCoordinate.from_dms_str(lat_str, lon_str)` |
| Accept any coordinate form from user/external source | `normalize_coord_input(lat, lon)` |
| Extract `(lat, lon)` float tuple | `coord.as_decimal()` |
| Format as DD strings with hemisphere | `coord.to_dd_str()` |
| Format as DDM strings | `coord.to_ddm_str()` |
| Format as DMS strings | `coord.to_dms_str()` |
| Parse a single DD string | `parse_dd(s, is_lat)` |
| Parse a single DDM string | `parse_ddm(s, is_lat)` |
| Parse a single DMS string | `parse_dms(s, is_lat)` |
| Parse an unknown coordinate string | `parse_any_coord_string(s, is_lat)` |
| Extract DDM numeric components | `decimal_to_ddm_components(value, is_lat)` |
| Extract DMS numeric components | `decimal_to_dms_components(value, is_lat)` |

---

## 10. Public API Symbol Table

| Symbol | Type | Summary |
| :--- | :--- | :--- |
| `GeoCoordinate` | `dataclass` | Immutable `(lat, lon)` value object in decimal degrees |
| `normalize_coord_input` | `function` | Accepts any coordinate form; returns `GeoCoordinate` |
| `decimal_to_ddm_components` | `function` | Decomposes DD to `(deg, min, hem)` |
| `decimal_to_dms_components` | `function` | Decomposes DD to `(deg, min, sec, hem)` |
| `parse_dd` | `function` | Parses DD string → signed float |
| `parse_ddm` | `function` | Parses DDM string → signed float |
| `parse_dms` | `function` | Parses DMS string → signed float |
| `parse_any_coord_string` | `function` | Auto-detecting string parser → signed float |
