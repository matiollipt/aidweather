# Spatial Interpretation Guide — `aidweather`

Understanding the spatial properties of NASA POWER parameters is essential for sound environmental, agricultural, and climatological analyses.

---

## 1. Grid Cell Support vs. Weather Station Observations

NASA POWER does not provide point observations from ground-based weather stations. Instead, parameters are generated from global atmospheric reanalysis models and satellite observations.

When a coordinate (e.g., `-23.55° lat, -46.63° lon`) is requested:
1. NASA POWER maps the requested coordinate to the containing source product grid cell.
2. The value returned represents the spatial average over that entire source cell.
3. Coordinates located within the same source grid cell will return identical weather values.

---

## 2. Non-Uniform Parameter Resolutions

NASA POWER is composed of several underlying satellite and model data streams, each with its own native resolution:

| Parameter Family | Primary Source Product | Native Grid | Spatial Cell Size (~Eq.) |
| :--- | :--- | :--- | :--- |
| Meteorological (`T2M`, `PRECTOTCORR`, `RH2M`, `WS10M`, `PS`, etc.) | NASA GMAO MERRA-2 / GEOS-IT | 0.50° Lat × 0.625° Lon | ~55.5 km × 69.4 km |
| Solar Radiation (`ALLSKY_SFC_SW_DWN`, `ALLSKY_SFC_PAR_TOT`) | NASA LaRC CERES SYN1deg / SRB | 1.00° Lat × 1.00° Lon | ~111.1 km × 111.1 km |

> [!IMPORTANT]
> Do not assume all NASA POWER parameters share a 0.5° × 0.5° grid. When mixing meteorological (`T2M`) and solar radiation (`ALLSKY_SFC_SW_DWN`) parameters in a single point query, the solar values reflect a coarser 1.0° cell than the meteorological values.

---

## 3. Spatial Transect Sampling & Clamping Policy

When generating points along a 1D transect path, requesting point spacing finer than the native grid cell produces redundant requests that return identical cell values.

To prevent unnecessary server load and misleading spatial sampling:
- `aidweather` derives the minimum effective transect spacing from the finest native resolution of the requested parameters.
- For MERRA-2 meteorological parameters (`T2M`), the minimum spacing threshold is ~0.5° (~55.5 km).
- For CERES solar radiation parameters (`ALLSKY_SFC_SW_DWN`), the minimum spacing threshold is ~1.0° (~111.1 km).
- Requests specifying finer spacing are logged with transparent warnings and clamped to `max_allowed` distinct points.

---

## 4. Regional Bounding-Box Boundaries

The regional endpoint returns all grid cell centers falling within the requested bounding box:
- Bounds are inclusive (`latitude-min`, `latitude-max`, `longitude-min`, `longitude-max`).
- Maximum allowable span per request: 4.5° latitude × 4.5° longitude.
- Coordinates crossing the antimeridian (180° longitude) or with reversed bounds (`lat_min >= lat_max`) are rejected with explicit `ValueError` messages.
