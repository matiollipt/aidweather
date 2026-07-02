# Weather Data Source Comparison

This document helps you make an informed decision about which weather data source — or combination of sources — fits your study.

`aidweather` is built on top of **NASA POWER**, a satellite-derived reanalysis product. Other tools pull from different data origins, each with its own strengths, caveats, and appropriate use cases. Understanding these differences is essential before designing any meteorological analysis.

---

## Contents

- [How to read this guide](#how-to-read-this-guide)
- [aidweather / NASA POWER](#aidweather--nasa-power)
- [Meteostat](#meteostat)
- [Side-by-side comparison](#side-by-side-comparison)
- [Decision guide](#decision-guide)
- [Using multiple sources together](#using-multiple-sources-together)
- [Validation workflow](#validation-workflow)

---

## How to read this guide

Each tool is presented with four sections:

1. **Data source** — where the numbers come from, who produces them, and what the physical basis is.
2. **The library** — what the Python package does, how it works, and what its key capabilities are.
3. **Installation** — how to add it to your project.
4. **Known limitations** — what the tool cannot do or does poorly.

The [Decision guide](#decision-guide) at the end translates these characteristics into concrete recommendations.

---

## aidweather / NASA POWER

### Data source

**NASA POWER** (Prediction Of Worldwide Energy Resources) is a reanalysis product maintained by NASA's Langley Research Center. It is derived from satellite remote sensing observations and modelled using the **MERRA-2 atmospheric reanalysis** system — a global climate model assimilated with observational satellite data.

| Attribute | Details |
|---|---|
| **Producer** | NASA Langley Research Center (LaRC), Applied Science Program |
| **Physical basis** | Satellite-derived + MERRA-2 atmospheric reanalysis |
| **Data type** | Modelled/reanalysis — not direct station readings |
| **Spatial coverage** | Global, every coordinate on Earth |
| **Spatial resolution** | 0.5° × 0.5° grid (~55 km per cell) |
| **Temporal resolution** | Daily (from 1981-01-01) and Hourly (from 2001-01-01) |
| **Near real-time lag** | ~5–7 days behind present |
| **Missing data** | Effectively none — the global model always produces a value |
| **Agro-community parameters** | Solar radiation, evapotranspiration, humidity, wind speed, precipitation (corrected), soil temperature, and more |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free, including commercial use; attribution required |
| **Citation** | *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program."* |
| **Homepage** | [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/) |

> [!IMPORTANT]
> NASA POWER data are **modelled estimates**, not direct measurements. They represent the best global gridded estimate for a given cell, but they are smoothed over ~55 km and may not capture fine-scale terrain effects or local microclimates.

### The library

`aidweather` is a Python library that wraps the NASA POWER REST API with a local SQLite cache, retry logic, and coordinate normalization. It is part of the `aidbio` toolchain and is specifically designed for agricultural and environmental pipelines.

Key capabilities:

- **Single-point queries** — daily or hourly data for one lat/lon.
- **Multi-point batch queries** — fetches dozens of sites in parallel with concurrency control.
- **Spatial transects** — evenly-spaced points along a great-circle path between two endpoints.
- **Regional grid queries** — returns data for every 0.5° cell within a bounding box (up to 4.5° × 4.5°).
- **Smart caching** — results are stored in a shared SQLite database compressed with gzip. Repeated queries for the same location are served from disk instantly, with zero new API requests.
- **Interval splitting** — if you previously cached Jan–Jun and now request Jan–Dec, only Jul–Dec is fetched from the API.
- **Rate limiting** — a client-side sliding window rate limiter caps requests at 30/minute (configurable), protecting the NASA service and your IP.
- **Clean output** — returns a `pandas` DataFrame with a timezone-naive `DatetimeIndex` and numeric columns. NASA's `-999` fill values are coerced to standard `NaN`.

See the [Client documentation](client.md) for the full API reference.

### Installation

```bash
# Via the official install script (Linux / macOS):
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash

# Or as a uv tool (system-wide, isolated):
uv tool install git+https://github.com/matiollipt/aidweather.git

# Or with pip inside an existing environment:
pip install git+https://github.com/matiollipt/aidweather.git
```

```python
from aidweather import PowerClient

client = PowerClient(temporal_api="daily")
df = client.get_point_data(
    lat=-15.7975,
    lon=-47.8919,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR", "RH2M"],
)
```

### Known limitations

- **Spatial resolution is coarse** (~55 km). Local microclimates, valley effects, and urban heat islands are invisible at this scale.
- **Precipitation is a modelled estimate** — daily totals are often smoothed relative to what a rain gauge would record, particularly for convective events.
- **Not a substitute for in-situ observation** — use it where no station exists, not as a replacement for quality-controlled station records where they are available.
- **Near real-time lag** — approximately 5–7 days behind present; unsuitable for operational forecasting.
- **Regional endpoint** — limited to one parameter per request and a bounding box of 4.5° × 4.5°.

---

## Meteostat

### Data source

**Meteostat** is a free weather and climate API that aggregates **historical observations from national weather services** (NOAA, DWD, ECCC, INMET, and others). Data comes from actual weather stations, not models. Meteostat normalizes these records, performs quality control, and makes them available through a unified API and Python library.

| Attribute | Details |
|---|---|
| **Producer** | Meteostat project (community-maintained, open source) |
| **Physical basis** | Ground-based weather station observations |
| **Data type** | Observed — direct sensor readings from certified stations |
| **Spatial coverage** | Wherever weather stations exist (uneven; dense in Europe/North America) |
| **Spatial resolution** | Station-level; spatial interpolation available (degrades with distance) |
| **Temporal resolution** | Daily and Hourly |
| **Historical depth** | Varies by station; some records go back to the early 20th century |
| **Near real-time lag** | ~1–2 days for most stations |
| **Missing data** | Common — stations go offline, change instruments, or have QC gaps |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Citation** | Cite the data source as "Meteostat" and include the station WMO IDs when possible |
| **Homepage** | [https://dev.meteostat.net/](https://dev.meteostat.net/) |

> [!NOTE]
> Because Meteostat relies on real station observations, data quality is high near active, well-maintained stations — but degrades rapidly as you move away from them. In remote regions or areas with sparse networks (common in tropical and developing countries), nearby stations may be 50–200 km away, and interpolation introduces substantial error.

### The library

`meteostat` is a Python package that provides access to historical weather observations via the Meteostat API, using a local Pandas-based workflow.

Key capabilities:

- **Point data with interpolation** — use `ms.Point(lat, lon)` and the library will find nearby stations and spatially interpolate to your coordinate.
- **Station-level data** — query raw data from a specific WMO station by ID.
- **Station search** — find nearby stations for a given coordinate.
- **Daily and hourly resolution**.
- **Standard variables** — temperature (mean, min, max), precipitation, wind speed and direction, pressure, humidity, snow depth.

The library **does not** provide caching, transects, regional grids, rate limiting, or batch multi-point workflows.

### Installation

```bash
pip install meteostat
```

```python
from meteostat import Point, Daily
from datetime import datetime

start = datetime(2023, 1, 1)
end   = datetime(2023, 12, 31)

# Create a Point for the location
location = Point(-15.7975, -47.8919)  # Brasília

# Get daily data
data = Daily(location, start, end)
df = data.fetch()
print(df.head())
```

> [!NOTE]
> **Station availability** — the `Point` object transparently finds and interpolates from nearby stations. Call `Stations().nearby(lat, lon).fetch()` to inspect which stations are used and how far they are from your coordinate. Large distances (>50 km) signal that interpolation may introduce significant errors.

### Known limitations

- **Station-dependent coverage** — in regions with sparse networks (most of sub-Saharan Africa, the Amazon basin, remote parts of Brazil, Oceania), there may simply be no station within a usable distance.
- **Data gaps are common** — stations go offline, change sensors, or are temporarily excluded from feeds. Gaps are not filled.
- **Interpolation degrades with distance** — beyond ~50–100 km, spatial interpolation from neighboring stations introduces meaningful bias.
- **No bulk spatial API** — fetching data for many coordinates requires looping; there is no native batch or transect mode.
- **No built-in caching** — every call hits the API. High-frequency or multi-site queries will be slow.
- **Not reproducible over time** — station re-processing or additions can silently alter historical values on re-fetch.
- **Community-maintained** — API availability and library maintenance depend on the Meteostat project. Occasional downtime has been observed.

---

## Side-by-side comparison

| Dimension | aidweather (NASA POWER) | Meteostat |
|---|---|---|
| **Data origin** | Satellite reanalysis (MERRA-2) | Ground station observations |
| **Global coverage** | ✅ Any coordinate on Earth | ⚠️ Depends on station proximity |
| **Spatial resolution** | 0.5° × 0.5° (~55 km grid) | Station-level (variable) |
| **Missing data** | Effectively none | Common (station gaps) |
| **Precipitation quality** | Smoothed / modelled | Observed (more realistic peaks) |
| **Temperature quality** | Good; slight warm bias possible in urban areas | High (when near a station) |
| **Solar radiation** | ✅ Rich set of solar parameters | ❌ Not available |
| **Evapotranspiration** | ✅ (several ET variants) | ❌ Not available |
| **Historical depth** | Daily: 1981+; Hourly: 2001+ | Varies by station (decades) |
| **Near real-time lag** | ~5–7 days | ~1–2 days |
| **Hourly data** | ✅ | ✅ |
| **Spatial transects** | ✅ (native in aidweather) | ❌ |
| **Regional grid queries** | ✅ (native in aidweather) | ❌ |
| **Multi-point batch** | ✅ (parallel, with rate limiting) | ❌ (manual loop) |
| **Built-in caching** | ✅ SQLite + gzip | ❌ |
| **Python library** | `aidweather` | `meteostat` |
| **Installation** | `pip install` / install script / uv tool | `pip install meteostat` |
| **License** | Library: Apache-2.0; Data: CC BY 4.0 | Library: MIT; Data: CC BY 4.0 |
| **Reproducibility** | High (model is stable) | Moderate (station data can be re-processed) |
| **Best for** | Spatial coverage, agro-pipelines, remote sites | Observed baselines, validation, dense-network regions |

---

## Decision guide

### Use aidweather (NASA POWER) when…

- Your study site is **remote, rural, or lacks nearby weather stations** — which is most of the world's agricultural land.
- You need **solar radiation** (`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`) or **evapotranspiration** parameters — these are simply not available in Meteostat.
- You need **spatial data**: transects across a watershed, a grid over a region, or dozens of sites fetched in a single pipeline call.
- **Data completeness is critical** — a complete, gap-free time series is more important than the marginal accuracy gain from station observations.
- You are building a **reproducible pipeline** that must return the same result when re-run months later.
- You are integrating with other `aidbio` tools that consume `aidweather` output.

### Use Meteostat when…

- Your site is in a **region with dense, high-quality station coverage** (e.g., European countries, major Brazilian cities) and a nearby station is within ~30 km.
- The analysis requires **observed extremes** — peak precipitation events, temperature anomalies — that reanalysis tends to smooth.
- You are performing **formal validation** of another data source (including NASA POWER) against observed records.
- Your study specifically concerns **what was actually measured** at a station (e.g., a regulatory or insurance context).

### Use both together when…

- You are **calibrating or validating** a model built on NASA POWER data — use Meteostat as the observational reference.
- You want to **quantify the bias** of the reanalysis at your specific site before relying on it for predictions.
- You are doing an **uncertainty analysis** that requires comparing modelled vs. observed series.
- You need **full spatial coverage** (NASA POWER) but also want to **anchor selected sites** to observed station data (Meteostat).

> [!TIP]
> A bias correction factor derived from a Meteostat vs. NASA POWER comparison at one well-monitored reference site can be used to adjust NASA POWER estimates at nearby remote sites — giving you the spatial coverage of the model with improved local accuracy.

---

## Using multiple sources together

When combining sources, three practical rules apply:

1. **Align temporal resolution** — both must be at the same frequency (e.g., both daily) before merging. Use `pandas` `.resample()` if needed.
2. **Normalize units** — NASA POWER returns temperature in °C (`T2M`), Meteostat also uses °C (`tavg`, `tmin`, `tmax`), but precipitation needs attention: NASA POWER returns `PRECTOTCORR` in mm/day, Meteostat returns `prcp` in mm/day. Always verify before computing differences.
3. **Use an inner join** — Meteostat may have gaps. Use `how="inner"` when merging DataFrames to avoid computing metrics on NaN-contaminated rows.

```python
import pandas as pd
from aidweather import PowerClient
from meteostat import Point, Daily
from datetime import datetime

lat, lon = -15.7975, -47.8919
start, end = datetime(2023, 1, 1), datetime(2023, 12, 31)

# --- aidweather ---
client = PowerClient(temporal_api="daily")
df_aid = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=["T2M"])
df_aid = df_aid.rename(columns={"T2M": "T2M_NASA"})

# --- Meteostat ---
df_ms = Daily(Point(lat, lon), start, end).fetch()[["tavg"]].rename(
    columns={"tavg": "T2M_METEOSTAT"}
)

# --- Merge and compare ---
comparison = df_ms.join(df_aid, how="inner")
bias = (comparison["T2M_METEOSTAT"] - comparison["T2M_NASA"]).mean()
print(f"Mean bias (Meteostat - NASA POWER): {bias:.3f} °C")
```

---

## Validation workflow

`aidweather` ships a validation scratchpad at [`scratchpad/external_validation.py`](../scratchpad/external_validation.py) that implements a full comparison workflow for a single location (Brasília, Brazil, 2025):

1. **Fetches NASA POWER** directly via the REST API and via `aidweather` (to verify they match exactly).
2. **Fetches Meteostat** for the same coordinate and period.
3. **Merges** all three series on a common date index.
4. **Computes MAE and RMSE** of aidweather vs. Meteostat.
5. **Produces three plots**:
   - Time-series overlay (all three sources on the same axes).
   - Residual error profile over time (bias drift).
   - Error distribution histogram.
   - Pearson correlation heatmap.

### Interpretation guidance

| Metric | Acceptable threshold (daily T2M) |
|---|---|
| MAE | < 1.5 °C |
| RMSE | < 2.0 °C |
| Pearson r | > 0.95 |

If your site passes these thresholds, NASA POWER data from `aidweather` is fit for purpose in agricultural and environmental modeling. If not, consider applying a site-specific bias correction anchored to the Meteostat (or ERA5) reference, or review whether a closer Meteostat station exists.

> [!NOTE]
> These thresholds are pragmatic guidelines for agricultural modeling, not universal scientific standards. The appropriate tolerance depends on your specific application — crop yield models, for example, are more sensitive to temperature biases than rainfall trend analyses.

---

*More data sources will be added to this comparison as they are validated against the `aidweather` workflow. Candidates include CHIRPS (precipitation-only).*
