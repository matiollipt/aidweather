# API Reference — `aidweather`

This document details the public Python API surface of `aidweather`.

---

## `aidweather.client.PowerClient`

```python
class PowerClient:
    def __init__(
        self,
        temporal_api: Literal["daily", "hourly"] = "daily",
        cache_config: dict[str, Any] | None = None,
        logging_config: dict[str, Any] | None = None,
        api_limits: dict[str, Any] | None = None,
    ) -> None:
        ...
```

### Methods

#### `get_point_data`
```python
def get_point_data(
    self,
    request: PointRequest | None = None,
    **kwargs,
) -> pd.DataFrame:
```
Fetches time-series weather data for a single geographic coordinate.

#### `get_point_data_from_coordinate`
```python
def get_point_data_from_coordinate(
    self,
    coord: GeoCoordinate,
    start: datetime | str | timedelta | pd.Timestamp,
    end: datetime | str | timedelta | pd.Timestamp,
    params: list[str],
    elevation: float | None = None,
    wind_elevation: float | None = None,
    wind_surface: float | None = None,
) -> pd.DataFrame:
```
Fetches single-point data using a `GeoCoordinate` instance. Returns a DataFrame indexed by `date` with `df.attrs["spatial_provenance"]` populated.

#### `get_multi_point_data`
```python
def get_multi_point_data(
    self,
    points: list[dict[str, Any] | tuple[float, float]] | pd.DataFrame,
    start: datetime | str | timedelta | pd.Timestamp,
    end: datetime | str | timedelta | pd.Timestamp,
    params: list[str],
    max_workers: int = 5,
) -> tuple[pd.DataFrame, list[tuple[Any, str]]]:
```
Fetches data concurrently for multiple points. Returns `(df, failed_points)`.

#### `get_transect_data`
```python
def get_transect_data(
    self,
    request: TransectRequest | None = None,
    **kwargs,
) -> pd.DataFrame:
```
Samples points along a straight 1D path. Automatically clamps sample points if spacing is sub-resolution.

#### `get_regional_data`
```python
def get_regional_data(
    self,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: datetime | str | timedelta | pd.Timestamp,
    end: datetime | str | timedelta | pd.Timestamp,
    params: list[str],
) -> pd.DataFrame:
```
Queries a 2D regional bounding box (max 4.5° × 4.5°, 1 parameter).

#### `get_parameter_metadata`
```python
def get_parameter_metadata(self, code: str | None = None) -> dict[str, Any]:
```
Returns structured parameter metadata (short name, units, source family, native grid, availability) for `code` or all parameters if `None`.

#### `summarize`
```python
def summarize(self, df: pd.DataFrame) -> None:
```
Prints a styled Rich report with transfer metrics, profile statistics, and connection diagnostic information.

---

## `aidweather.geo.GeoCoordinate`

```python
@dataclass(frozen=True)
class GeoCoordinate:
    lat: float
    lon: float
```

### Classmethods
- `from_decimal(lat: float, lon: float) -> GeoCoordinate`
- `from_strings(lat_str: str, lon_str: str) -> GeoCoordinate` (auto-detects DD, DDM, DMS)
- `from_dd_str(lat_str, lon_str)`
- `from_ddm_str(lat_str, lon_str)`
- `from_dms_str(lat_str, lon_str)`

### Instance Methods
- `as_decimal() -> tuple[float, float]`
- `to_dd_str(lat_precision: int = 5, lon_precision: int = 5) -> tuple[str, str]`
- `to_ddm_str(minute_precision: int = 3) -> tuple[str, str]`
- `to_dms_str(second_precision: int = 0) -> tuple[str, str]`

---

## `aidweather.config._Config`

- `cfg.get(key_path, default=None)`
- `cfg.params(group="default") -> dict[str, str]`
- `cfg.param_metadata(code=None) -> dict[str, Any]`
- `cfg.get_native_grid(code: str) -> tuple[float, float]`
- `cfg.cache_config() -> dict[str, Any]`
