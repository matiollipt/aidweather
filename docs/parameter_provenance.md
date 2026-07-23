# Parameter Provenance Guide — `aidweather`

This document details the scientific lineage, units, temporal availability, and primary applications for all 18 core parameters supported by `aidweather`.

---

## Parameter Catalogue & Matrix

### 1. Air Temperature & Thermal Indices

#### `T2M` — Temperature at 2 Meters
- **Short Name**: Temperature at 2 Meters
- **Source Product**: NASA GMAO MERRA-2 / GEOS-IT Reanalysis
- **Native Resolution**: 0.50° Latitude × 0.625° Longitude
- **Temporal Coverage**: Daily (1981-01-01 to Present), Hourly (2001-01-01 to Present)
- **Units**: Daily: °C, Hourly: °C
- **Time Standards**: LST, UTC
- **Provisional Data Tail**: Yes (recent ~2–3 months provisional prior to final MERRA-2 reprocessing)

#### `T2M_MAX` — Maximum Air Temperature at 2 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: Thermal stress modeling, heat wave detection in crop phenology.

#### `T2M_MIN` — Minimum Air Temperature at 2 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: Frost risk assessment, chilling requirement tracking, growing degree-day (GDD) calculations.

#### `T2M_RANGE` — Diurnal Temperature Range at 2 Meters
- **Source Product**: Derived from MERRA-2 T2M_MAX - T2M_MIN
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: Day-night thermal amplitude analysis for crop development models.

#### `T2MWET` — Wet Bulb Temperature at 2 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: Livestock heat stress assessment, evaporative cooling limits.

#### `T2MDEW` — Dew Point Temperature at 2 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: Foliar disease germination forecasting, relative humidity derivations.

---

### 2. Surface & Moisture Parameters

#### `TS` — Surface Skin Temperature
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: °C
- **Application**: SEBAL/METRIC remote sensing evapotranspiration, soil boundary conditions.

#### `RH2M` — Relative Humidity at 2 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: %
- **Application**: Atmospheric moisture tracking, crop disease risk index modeling.

#### `PRECTOTCORR` — Corrected Total Precipitation
- **Source Product**: MERRA-2 / GEOS-IT (Bias-Corrected Liquid + Frozen Water)
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Daily Unit**: mm/day
- **Hourly Unit**: mm/hr
- **Application**: Crop water budget, drought indices, rainfall estimation.

#### `GWETTOP` — Surface Soil Wetness
- **Source Product**: MERRA-2 / GEOS-IT (land-surface model)
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Temporal Coverage**: Daily only (1981-01-01 to Present). No hourly variant is published by NASA POWER; `aidweather` raises `ValueError` if requested via the hourly endpoint.
- **Units**: 0–1 (fraction of saturation, top ~0-5 cm soil layer)
- **Application**: Irrigation timing, seedling emergence risk, topsoil-driven disease modeling (e.g. damping-off).

#### `GWETROOT` — Root Zone Soil Wetness
- **Source Product**: MERRA-2 / GEOS-IT (land-surface model)
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Temporal Coverage**: Daily only (1981-01-01 to Present). No hourly variant is published by NASA POWER; `aidweather` raises `ValueError` if requested via the hourly endpoint.
- **Units**: 0–1 (fraction of saturation, root zone layer)
- **Application**: Irrigation scheduling, water-stress modeling for established crops.

#### `GWETPROF` — Profile Soil Moisture
- **Source Product**: MERRA-2 / GEOS-IT (land-surface model)
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Temporal Coverage**: Daily only (1981-01-01 to Present). No hourly variant is published by NASA POWER; `aidweather` raises `ValueError` if requested via the hourly endpoint.
- **Units**: 0–1 (fraction of saturation, full soil column)
- **Application**: Drought monitoring, long-term water-balance modeling.

---

### 3. Solar Radiation & Photosynthesis

#### `ALLSKY_SFC_SW_DWN` — All Sky Surface Shortwave Downward Irradiance
- **Source Product**: NASA LaRC CERES SYN1deg / FLASHFlux / SRB
- **Native Resolution**: 1.00° Latitude × 1.00° Longitude
- **Daily Unit**: kWh/m²/day
- **Hourly Unit**: W/m²
- **Temporal Coverage**: Daily (1984-01-01 to Present), Hourly (2001-01-01 to Present)
- **Application**: Reference Evapotranspiration (ETo Penman-Monteith), solar energy yield.

#### `ALLSKY_SFC_PAR_TOT` — All Sky Surface PAR Total
- **Source Product**: NASA LaRC CERES SYN1deg / FLASHFlux / SRB
- **Native Resolution**: 1.00° Latitude × 1.00° Longitude
- **Daily Unit**: MJ/m²/day
- **Hourly Unit**: W/m²
- **Application**: Photosynthetically Active Radiation (400–700 nm) for biomass crop models (DSSAT, APSIM).

---

### 4. Wind & Surface Pressure

#### `WS10M` — Wind Speed at 10 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: m/s
- **Application**: Wind energy estimation, pesticide drift forecasting, evapotranspiration.

#### `WS10M_MAX` — Maximum Wind Speed at 10 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: m/s
- **Application**: Extreme storm event detection, crop lodging risk.

#### `WD10M` — Wind Direction at 10 Meters
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: degrees
- **Application**: Wind rose construction, atmospheric dispersion modeling.

#### `PS` — Surface Pressure
- **Source Product**: MERRA-2 / GEOS-IT
- **Native Resolution**: 0.50° Lat × 0.625° Lon
- **Units**: kPa
- **Application**: Atmospheric density correction in evapotranspiration equations.
