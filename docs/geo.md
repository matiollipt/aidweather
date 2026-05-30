# geo

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## Purpose
Provides robust utilities for handling, parsing, and formatting geospatial coordinates. It ensures coordinates are always represented as valid decimal degrees.

## Key responsibilities
- Parsing coordinates from various string formats (DMS, DDM, DD).
- validating latitude (-90 to 90) and longitude (-180 to 180).
- Formatting coordinates back to strings.
- Normalizing mixed input types into a standard object.

## Public API

### Classes
- `GeoCoordinate` (Frozen Dataclass):
  - Attributes: `lat` (float), `lon` (float).
  - `from_decimal(lat, lon)`: Factory method.
  - `from_strings(lat_str, lon_str)`: Auto-detects format.
  - `to_dd()`, `to_dms_str()`, `to_ddm_str()`: Formatters.

### Functions
- `normalize_coord_input(lat, lon=None) -> GeoCoordinate`:
  - Accepts tuples, separate args, strings, or existing objects and returns a `GeoCoordinate`.

## Data flow and dependencies
- **External dependencies**: `re`, `math`, `dataclasses`.
- **Downstream**: Used heavily by `client.PowerClient` and `cli`.

## Error handling and edge cases
- **Validation**: Raises `ValueError` immediately upon creation if coordinates are out of bounds.
- **Parsing**: Raises `ValueError` if string formats do not match expected patterns.

## Minimal usage example
```python
from aidweather.geo import GeoCoordinate, normalize_coord_input

# From strings
c1 = normalize_coord_input("23°33'0.0\" S", "46°37'48.0\" W")

# From numbers
c2 = normalize_coord_input(-23.55, -46.63)

print(c1.lat, c1.lon)
```
