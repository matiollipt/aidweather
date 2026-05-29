# %% [markdown]
# # Masterclass in Agroclimatic Analytics: Harnessing NASA POWER with `aidweather`
#
# ### *A Unified Framework for Scalable, Cache-Backed Spatial-Temporal Analysis in Agronomy & Crop Modeling*
#
# **Author:** Cleverson Matiolli, PhD & Gemini
# **Ecosystem:** `aidweather` (Base Package) -> `aidviz` (Visualization) -> `aidfarm` (Ecosystem Analytics)
#
# ---
#
# ## Executive Overview & Biophysical Rationale
#
# Quantitative agronomy and precision agriculture require high-resolution, robust meteorological feeds to parameterize dynamic crop growth models (e.g., DSSAT, APSIM) and machine learning models for yield forecasting. However, raw environmental APIs present significant operational friction:
# 1. **High Latency & Rate Limits:** Repetitive spatial-temporal queries can throttle IP addresses or violate API service level agreements (SLAs).
# 2. **Inconsistent Geocoordinate Syntax:** Parsing Degrees-Minutes-Seconds (DMS), Degrees-Decimal Minutes (DDM), or raw Decimal Degrees (DD) from heterogeneous fieldwork logs introduces human and numeric errors.
# 3. **Data Quality & Schema Drift:** Handling missing values, timezone misalignments, and irregular formats requires boilerplate data cleaning pipelines.
#
# The `aidweather` package solves these issues. This masterclass tutorial explores all aspects of `aidweather`:
# - **Centralized Configuration:** Bundled metadata catalogues and agronomic dictionaries.
# - **Geospatial Coordinate Normalization:** Value objects for converting, validating, and displaying geolocations.
# - **SQLite Caching Engine:** Intelligent split-and-merge database caching that reduces network overhead.
# - **Parallel Concurrency:** Thread-pool based spatial querying across coordinate grids.
# - **Data Lineage:** Preprocessing utilities to ensure robust Pandas time-series alignments.
#
# ---
#
# ## Biophysical & STEM-Level Weather Parameters
#
# The NASA POWER API utilizes the **MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)** data assimilation system and satellite observations to estimate key parameters:
#
# *   **Photosynthetically Active Radiation ($\text{PAR}$, code `ALLSKY_SFC_PAR_TOT`):**
#     $$\text{PAR} = \int_{400\text{ nm}}^{700\text{ nm}} I(\lambda) d\lambda \quad \left[\text{MJ}\cdot\text{m}^{-2}\cdot\text{day}^{-1}\right]$$
#     Drives the crop carbon accumulation rate. Crop growth engines utilize $\text{PAR}$ coupled with a Radiation Use Efficiency ($\text{RUE}$) parameter to calculate daily dry matter production: $\Delta\text{Biomass} = \text{PAR} \times \text{RUE}$.
#
# *   **Air Temperature at 2 meters ($T_{2\text{m}}$, code `T2M`, `T2M_MAX`, `T2M_MIN`):**
#     Controls the phenological clock via Growing Degree Days ($\text{GDD}$):
#     $$\text{GDD} = \max\left(\frac{T_{2\text{m, max}} + T_{2\text{m, min}}}{2} - T_{\text{base}}, 0\right)$$
#     Large diurnal temperature ranges ($T_{\text{range}} = T_{2\text{m, max}} - T_{2\text{m, min}}$) alter the ratio of crop photosynthesis to respiration.
#
# *   **Dew Point Temperature at 2 meters ($T_{\text{dew}}$, code `T2MDEW`):**
#     Serves as an operational proxy for Leaf Wetness Duration (LWD). Fungal spore germination (e.g., *Phakopsora pachyrhizi* or Soybean Rust) requires liquid water on the leaf surface, which occurs when the ambient temperature approaches the dew point:
#     $$T_{2\text{m}} - T_{\text{dew}} \leq 2^{\circ}\text{C}$$
#
# *   **Corrected Total Precipitation ($P_{\text{corr}}$, code `PRECTOTCORR`):**
#     A bias-adjusted estimate of rainfall and snowmelt, critical for calculating the field soil water balance:
#     $$\Delta S = P_{\text{corr}} - \text{ET}_0 - D - R$$
#     where $\text{ET}_0$ is the Reference Evapotranspiration, $D$ is deep drainage, and $R$ is surface runoff.
#
# Let's initialize our workspace and begin the demonstration!

# %%
import os
import sqlite3
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Set up logging for interactive execution
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Import all core elements from aidweather
from aidweather import (
    PowerClient,
    GeoCoordinate,
    normalize_coord_input,
    cfg,
    get_config,
    ensure_date_column,
)

print("Packages loaded. Python environment and aidweather library are ready!")

# %% [markdown]
# ## Part 1: Centralized Configuration System
#
# The `aidweather` package implements a centralized configuration architecture. It loads options from an internal, bundled `config.json` using `importlib.resources`. This mechanism provides a single source of truth across downstream applications (like `aidviz` and `aidfarm`), while falling back to safe defaults if the configuration file is modified or absent.
#
# Let's inspect the `cfg` singleton configuration values:

# %%
# Retrieve the configuration singleton instance
config = get_config()

# 1. Fetch default base URLs for NASA POWER API
print("=== API Endpoints ===")
print("Daily Point URL:", config.get_url("daily", "point"))
print("Hourly Point URL:", config.get_url("hourly", "point"))
print("Daily Regional URL:", config.get_url("daily", "regional"))

# 2. Access default agronomic parameter sets
print("\n=== Parameter Groups ===")
default_params = config.params("default")
print("Default Parameters ({}):".format(len(default_params)))
for code, name in default_params.items():
    print(f"  - {code:18s}: {name}")

# %% [markdown]
# Each parameter comes with a detailed biophysical description. Let's inspect the descriptions for agricultural features:

# %%
print("=== STEM Biophysical Descriptions ===")
descriptions = config.param_descriptions()
for param in ["T2M", "ALLSKY_SFC_PAR_TOT", "PRECTOTCORR", "T2MDEW"]:
    print(f"\n[{param}]:")
    print(descriptions.get(param, "No description found."))

# %% [markdown]
# The configuration system also maps parameters to distinct hex colors to ensure consistent visualizations throughout the downstream ecosystem (e.g., `aidviz`).
#
# Let's view the hex map:

# %%
print("=== Parameter Color Map ===")
color_map = config.color_map()
for param, hex_color in color_map.items():
    print(f"  - {param:18s}: {hex_color}")

# %% [markdown]
# ## Part 2: High-Precision Geospatial Coordinate Management
#
# Field notebooks and historical datasets often represent geographic locations in diverse formats:
# - **Decimal Degrees (DD):** `"-23.55"`
# - **Degrees, Decimal Minutes (DDM):** `"23° 33.0' S"`
# - **Degrees, Minutes, Seconds (DMS):** `"23° 33' 0\" S"`
#
# To guarantee type safety, value-object integrity, and error-free execution, `aidweather` provides the `GeoCoordinate` dataclass and the `normalize_coord_input` function. Let's test the robust parsing, validation, and conversion features:

# %%
# 1. Parsing diverse string representations
coords = [
    # Latitude in DDM, Longitude in DMS
    GeoCoordinate.from_strings("23° 33.0' S", "46° 37' 48\" W"),
    # Negative decimal numbers represented as strings
    GeoCoordinate.from_strings("-23.55°", "-46.63°"),
    # Raw numeric float values (South / West are negative)
    GeoCoordinate.from_decimal(-23.55, -46.63),
]

print("=== Parsed Coordinates ===")
for idx, c in enumerate(coords, 1):
    print(f"Coord {idx}: Lat: {c.lat:.5f}, Lon: {c.lon:.5f}")

# 2. Normalizing mixed inputs into a single GeoCoordinate object
mixed_input = normalize_coord_input((-23.55, "46°37'48.0\" W"))
print("\nNormalized Mixed Input (Tuple):", mixed_input)

# %% [markdown]
# The `GeoCoordinate` object also lets you output coordinates back into standardized strings for publication-grade metadata files:

# %%
print("=== Exporting Formatted Strings ===")
print("Decimal Degrees (DD) :", mixed_input.to_dd_str(lat_precision=5))
print("Decimal Minutes (DDM):", mixed_input.to_ddm_str(minute_precision=3))
print("Minutes/Seconds (DMS):", mixed_input.to_dms_str(second_precision=2))

# %% [markdown]
# The value object performs input range checks on instantiation to prevent submission of invalid coordinates to the NASA API, raising descriptive exceptions. Let's verify this validation:

# %%
try:
    # Latitude must reside between -90 and +90
    invalid_coord = GeoCoordinate.from_decimal(95.0, -46.63)
except ValueError as e:
    print("Caught expected validation error:", e)

# %% [markdown]
# ## Part 3: Production-Grade Weather Data Ingestion & Caching
#
# The `PowerClient` wraps the NASA POWER API with a robust SQLite database caching layer located at `~/.aidweather_cache/` by default (or configured inside `config.json`).
#
# When a weather query is submitted:
# 1. The client checks the local cache database using a hashed representation of the request parameters and coordinates.
# 2. It identifies whether the requested temporal bounds are already covered.
# 3. If there is a missing sub-range, it fetches *only* the missing dates from the NASA API.
# 4. It merges and deduplicates the cached and newly retrieved datasets, writes the merged data back to the database as a Gzip-compressed BLOB, and returns the requested range to the user.
#
# Let's demonstrate a full caching round-trip, measuring cold vs. hot request latencies.
#
# *Note: We will use a mock session or real requests depending on network status. Because `PowerClient` uses a local sqlite database, we will clear the cache to ensure we measure a true "cold" request.*

# %%
# Initialize PowerClient
client = PowerClient(temporal_api="daily")

# Check if cache is enabled and find the cache path
cache_info = client.cache_cfg
print("Cache Path:", cache_info.get("path"))
print("Cache Enabled:", cache_info.get("enabled"))

# Ensure cache is clear before running the benchmark
db_path = Path(cache_info.get("path", ".")) / "aidweather_cache.db"
if db_path.exists():
    print("Clearing historical cache to start benchmark...")
    try:
        db_path.unlink()
        print("Cache deleted.")
    except Exception as e:
        print(f"Could not delete cache file: {e}")

# Re-initialize client to create a clean database
client = PowerClient(temporal_api="daily")

# %% [markdown]
# Let's perform a **Cold Request** (fetching data for the first time). The client will fetch the weather metrics from the NASA servers.

# %%
# Define parameters for an agricultural parcel (e.g., Londrina, PR, Brazil - major agricultural hub)
lat, lon = -23.31, -51.16
start_date = "2023-01-01"
end_date = "2023-01-15"
params = ["T2M", "PRECTOTCORR", "ALLSKY_SFC_PAR_TOT"]

print(
    f"Executing Cold Request for coordinates ({lat}, {lon}) from {start_date} to {end_date}..."
)
t0 = time.perf_counter()

df_cold = client.get_point_data(
    lat=lat, lon=lon, start=start_date, end=end_date, params=params
)

t_cold = time.perf_counter() - t0
print(f"Cold Request completed in {t_cold:.3f} seconds.")
print("Shape of returned DataFrame:", df_cold.shape)
print(df_cold.head())

# %% [markdown]
# Now, let's execute the exact same request again. Since the data is stored in the local cache, this will be a **Hot Request**, served entirely from the SQLite database.

# %%
print(f"Executing Hot Request for same coordinates ({lat}, {lon}) and dates...")
t0 = time.perf_counter()

df_hot = client.get_point_data(
    lat=lat, lon=lon, start=start_date, end=end_date, params=params
)

t_hot = time.perf_counter() - t0
print(f"Hot Request completed in {t_hot:.3f} seconds.")
print(f"Speedup factor: {t_cold / t_hot:.1f}x faster!")

# %% [markdown]
# Let's execute the `summarize` function. This prints a formatted dashboard containing data insights, request statistics, network transfer performance, and connection states.

# %%
client.summarize(df_hot)

# %% [markdown]
# Let's demonstrate the **Split-and-Merge Cache Logic**.
# We will query a date range that *partially* overlaps with the cached range: `"2023-01-10"` to `"2023-01-25"`.
# The client should identify that `2023-01-10` to `2023-01-15` is already cached, fetch *only* `2023-01-16` to `2023-01-25` from NASA, merge it, and cache the updated, continuous dataset.

# %%
extended_start = "2023-01-10"
extended_end = "2023-01-25"

print(f"Executing Overlapping Request from {extended_start} to {extended_end}...")
t0 = time.perf_counter()

df_extended = client.get_point_data(
    lat=lat, lon=lon, start=extended_start, end=extended_end, params=params
)

t_extended = time.perf_counter() - t0
print(f"Overlapping Request completed in {t_extended:.3f} seconds.")
print("Shape of extended DataFrame:", df_extended.shape)
print(
    "Index ranges returned: {} to {}".format(
        df_extended.index.min().date(), df_extended.index.max().date()
    )
)

# %% [markdown]
# ## Part 4: Advanced Concurrency & Spatial Transects
#
# Agricultural assessments often span multiple geographic points representing distinct treatment plots or an environmental gradient (e.g., altitude variation, distance from coastal humidity sources).
#
# The `PowerClient` offers two parallelized fetching routines:
# 1. **`get_multi_point_data`:** Submits concurrent API queries across an arbitrary list of coordinate dictionaries or a Pandas DataFrame.
# 2. **`get_expanded_point_data`:** Generates a linear sampling transect centered at a specific coordinate and fetches data along the spatial gradient.
#
# *NASA cautions against opening more than 5 concurrent HTTP connections from a single IP to prevent transient throttling. `PowerClient` handles thread execution limits safely and flags warnings if the concurrency pool exceeds standard guidelines.*
#
# Let's build a spatial transect running along the Latitude axis across a length of 20 kilometers:

# %%
center_lat, center_lon = -23.31, -51.16
distance_km = 20.0
num_points = 3  # Keep count low to respect NASA guidelines and speed up demo
transect_start = "2023-01-01"
transect_end = "2023-01-05"

print(
    f"Generating a {distance_km} km transect along the LATITUDE axis centered at ({center_lat}, {center_lon})..."
)

df_transect = client.get_expanded_point_data(
    lat=center_lat,
    lon=center_lon,
    start=transect_start,
    end=transect_end,
    params=["T2M", "PRECTOTCORR"],
    axis="lat",
    distance_km=distance_km,
    num_points=num_points,
    max_workers=3,
)

# Preview the returned multi-point spatial dataset
print("\nMulti-point transect dataframe shape:", df_transect.shape)
print("\nReturned DataFrame Head:")
print(df_transect.head(10))

# Notice that the returned DataFrame contains columns for 'lat' and 'lon', identifying each point!
print("\nUnique coordinates in dataset:")
print(df_transect[["lat", "lon"]].drop_duplicates().reset_index(drop=True))

# %% [markdown]
# ## Part 5: Data Preprocessing & Alignment Lineage
#
# Raw weather datasets can contain inconsistencies (e.g., date formats under column labels such as `obs_date`, `time`, `RecordDate`, or date records set in DatetimeIndexes).
# To provide a reliable, clean transition from raw inputs to predictive ML pipelines or crop models, the `aidweather.utils` module includes the `ensure_date_column` function.
#
# Let's inspect how it robustly:
# 1. Identifies date fields from a list of candidate names.
# 2. Normalizes date times to timezone-naive midnight strings.
# 3. Falls back to a DatetimeIndex when no date column is explicitly present.

# %%
# 1. Scenario A: DataFrame with a non-standard column name and string dates
raw_data_a = pd.DataFrame(
    {
        "obs_date": [
            "2023-05-15 08:30:00",
            "2023-05-16 12:45:00",
            "2023-05-17 19:15:00",
        ],
        "T2M": [18.5, 20.2, 17.8],
    }
)
print("Raw DataFrame A:")
print(raw_data_a)

# Standardize date column
clean_df_a = ensure_date_column(
    raw_data_a, name="date", candidates=["obs_date", "measurement_time"]
)
print("\nStandardized DataFrame A:")
print(clean_df_a)
print("Dtype of date column:", clean_df_a["date"].dtype)

# %% [markdown]
# Let's demonstrate the fallback when the date is stored in the Index as a DatetimeIndex:

# %%
# 2. Scenario B: DataFrame with a DatetimeIndex and no standard columns
datetime_idx = pd.date_range("2023-06-01", periods=3, freq="D")
raw_data_b = pd.DataFrame({"RH2M": [65.0, 72.0, 58.0]}, index=datetime_idx)
print("Raw DataFrame B (DatetimeIndex):")
print(raw_data_b)

# Generate unified date column from index fallback
clean_df_b = ensure_date_column(raw_data_b, name="date", index_fallback=True)
print("\nStandardized DataFrame B:")
print(clean_df_b)

# %% [markdown]
# ---
#
# ## Conclusion & Operational Guidelines
#
# We have completed our masterclass in weather data ingestion and preprocessing using `aidweather`!
#
# ### Key Architecture Takeaways:
# *   **Performance Optimization:** Always leverage the local caching layer. For production servers, set the `path` option in `config.json` or through environment variables to persistent disk storage to prevent cold startup API lags.
# *   **Scientific Credibility:** Coordinate handling must preserve spatial precision. Use `GeoCoordinate` to parse input types cleanly and avoid geographic offsets.
# *   **Parallel Query Control:** Keep parallel worker threads under the recommended safety maximum ($N=5$) when submitting requests to NASA to prevent rate limits.
# *   **Ecosystem Integration:** Standardized date columns using `ensure_date_column` represent a critical interface boundary. Clean dataframes can be directly fed into the downstream `aidviz` visualization routines or downstream machine learning tasks in `aidfarm`.

# %%
