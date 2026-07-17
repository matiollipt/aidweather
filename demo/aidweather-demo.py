# %% [markdown]
# # `aidweather` — Practical Guide
#
# **Author:** Cleverson Matiolli, PhD
#
# ---
#
# `aidweather` retrieves daily or hourly weather data from
# [NASA POWER](https://power.larc.nasa.gov/) and caches it locally so repeated
# requests don't hit the network again.
#
# This notebook walks through every public component of the package:
#
# | Section | What you'll learn |
# |---------|-------------------|
# | 1 · Configuration | Inspect API endpoints and available weather parameters |
# | 2 · Coordinates | Parse, validate and convert geographic coordinates |
# | 3 · Fetching data | Download weather data for a single location |
# | 4 · Caching | See how the cache speeds up repeated queries |
# | 5 · Multi-point | Fetch data for several locations in parallel |
# | 6 · Data cleanup | Standardise the date column in any DataFrame |
# | 7 · Logging | Configure and enable logging to a local log file |
# | 8 · One-liner bonus | Point and regional queries in a single call |


# %%
import logging
import time
from pathlib import Path

import pandas as pd

from aidweather import (
    GeoCoordinate,
    PowerClient,
    ensure_date_column,
    get_config,
    normalize_coord_input,
)

# Configure logging using built-in configuration settings
log_cfg = get_config().logging_config()
log_handlers = [logging.StreamHandler()]  # Console logger is always active

if log_cfg.get("enabled", False):
    log_file = log_cfg.get("filename")
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file_path))

logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
    force=True,
)

print("aidweather ready.")

# %% [markdown]
# ---
# ## 1 · Configuration
#
# `cfg` is a module-level singleton that reads `assets/config.json` once and
# exposes helpers for API URLs and parameter metadata.
# `get_config()` returns the same object — both refer to the same instance.

# %%
config = get_config()

# API base URLs
print("Daily point URL  :", config.get_url("daily", "point"))
print("Hourly point URL :", config.get_url("hourly", "point"))


# %% [markdown]
# ### Available parameter groups
#
# Parameters are grouped in `config.json`.
# The `"default"` group is a curated set of variables useful for most agronomic work.

# %%
default_params = config.params("default")
print(f"Default group — {len(default_params)} parameters:\n")
for code, label in default_params.items():
    print(f"  {code:<22} {label}")

# %% [markdown]
# The "all" group contains all available parameters.

# %%
all_params = config.params("all")
print(f"All group — {len(all_params)} parameters:\n")
for code, label in all_params.items():
    print(f"  {code:<22} {label}")

# %% [markdown]
# ### Parameter descriptions
#
# Each parameter has a longer description stored in the config.

# %%
descriptions = config.param_descriptions()
for code in all_params:
    print(f"[{code}]\n  {descriptions.get(code, 'n/a')}\n")

# %% [markdown]
# ---
# ## 2 · Coordinates
#
# `GeoCoordinate` accepts coordinates in **any common format** and converts
# between them.  It also validates ranges before you ever touch the API.
#
# Supported input styles:
# - Decimal degrees as a float: `-23.55`
# - Decimal degrees as a string: `"-23.55°"`
# - Degrees and decimal minutes: `"23° 33.0' S"`
# - Degrees, minutes, seconds: `"23° 33' 0\" S"`

# %%
# From two strings (any mix of formats)
c1 = GeoCoordinate.from_strings("23° 33.0' S", "46° 37' 48\" W")

# From two floats (south/west are negative)
c2 = GeoCoordinate.from_decimal(-23.55, -46.63)

for i, c in enumerate([c1, c2], 1):
    print(f"Coord {i}:  lat={c.lat:.5f}  lon={c.lon:.5f}")

# %% [markdown]
# ### Mixed-format input via `normalize_coord_input`
#
# If you have a tuple where each element could be a float or a string,
# `normalize_coord_input` handles it in one call.

# %%
coord = normalize_coord_input((-23.55, "46°37'48.0\" W"))
print("Normalised:", coord)

# %% [markdown]
# ### Export to formatted strings

# %%
print("DD  :", coord.to_dd_str(lat_precision=5))
print("DDM :", coord.to_ddm_str(minute_precision=3))
print("DMS :", coord.to_dms_str(second_precision=2))

# %% [markdown]
# ### Validation
#
# Invalid coordinates raise a `ValueError` immediately — before any API call is made.

# %%
try:
    GeoCoordinate.from_decimal(95.0, -46.63)  # latitude > 90 is invalid
except ValueError as e:
    print("Caught:", e)

# %% [markdown]
# ---
# ## 3 · Fetching weather data
#
# `PowerClient` wraps the NASA POWER API.
# Initialise it once and reuse it across queries.
#
# ```python
# client = PowerClient(temporal_api="daily")   # or "hourly"
# ```
#
# ### `get_point_data` — single location
#
# Returns a **pandas DataFrame** indexed by date, one column per parameter.

# %%
client = PowerClient(temporal_api="daily")

lat, lon = -23.31, -51.16  # Londrina, PR, Brazil
start, end = "2023-01-01", "2023-01-15"
params = ["T2M", "PRECTOTCORR", "ALLSKY_SFC_PAR_TOT"]

df = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=params)

print(df.shape)  # (days, parameters)
print(df.head())

# %% [markdown]
# ### `summarize` — quick data overview
#
# Prints a formatted table with row counts, date range and per-column statistics.

# %%
client.summarize(df)

# %% [markdown]
# ### A quick look at the data
#
# Once you have a DataFrame, standard pandas and matplotlib workflows apply.

# %%
# Daily mean temperature and rainfall over the fetched period
print("Mean temperature (°C):", df["T2M"].mean().round(2))
print("Total rainfall (mm)  :", df["PRECTOTCORR"].sum().round(2))

# Growing Degree Days (base 10 °C) — an example of a derived variable
if {"T2M"}.issubset(df.columns):
    df["GDD"] = (df["T2M"] - 10).clip(lower=0)
    print("\nGrowing Degree Days per day:")
    print(df[["T2M", "GDD"]].to_string())

# %% [markdown]
# ---
# ## 4 · Caching
#
# The client saves each response to a local SQLite database
# (`~/.aidweather_cache/aidweather_cache.db` by default).
# On the next request for the same location and date range, data is read
# from disk instead of the network.

# %%
cache_cfg = client.cache_cfg
print("Cache enabled:", cache_cfg.get("enabled"))
print("Cache path   :", cache_cfg.get("path"))

# %% [markdown]
# ### Cold vs. hot request timing

# %%
# Clear cache to get a true cold read
db_path = Path(cache_cfg.get("path", ".")) / "aidweather_cache.db"
if db_path.exists():
    db_path.unlink()

client = PowerClient(temporal_api="daily")  # re-init on fresh DB

t0 = time.perf_counter()
df_cold = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=params)
t_cold = time.perf_counter() - t0

t0 = time.perf_counter()
df_hot = client.get_point_data(lat=lat, lon=lon, start=start, end=end, params=params)
t_hot = time.perf_counter() - t0

print(f"Cold request : {t_cold:.3f} s")
print(f"Hot request  : {t_hot:.3f} s  ({t_cold / t_hot:.0f}× faster)")

# %% [markdown]
# ### Partial overlap — only the missing days are fetched
#
# If you extend the date range beyond what's already cached, the client
# fetches **only the new dates** and merges them with the cached data.

# %%
df_ext = client.get_point_data(
    lat=lat,
    lon=lon,
    start="2023-01-10",
    end="2023-01-25",  # overlaps with cached range
    params=params,
)

print("Extended range:", df_ext.index.min().date(), "→", df_ext.index.max().date())
print("Rows returned :", len(df_ext))

# %% [markdown]
# ---
# ## 5 · Multi-point fetching
#
# ### `get_transect_data` — 1D spatial transect
#
# Fetches data for evenly-spaced points along a straight-line path between two
# `GeoCoordinate` endpoints. Each point is resolved in parallel via the standard
# point API, so you can request multiple parameters.
#
# The minimum point spacing is **0.5° (~55 km)** to match the NASA POWER native
# grid resolution. If the requested density exceeds this, `num_points` is clamped
# automatically and an `INFO` message is logged.

# %%
coord_a = GeoCoordinate.from_decimal(-25.0, -51.16)  # southern end
coord_b = GeoCoordinate.from_decimal(-20.0, -51.16)  # northern end (~555 km)

df_transect = client.get_transect_data(
    start_coord=coord_a,
    end_coord=coord_b,
    start="2023-01-01",
    end="2023-01-05",
    params=["T2M", "PRECTOTCORR"],
    num_points=3,  # 3 evenly-spaced points along the path
    max_workers=3,
)

print("Transect shape:", df_transect.shape)
print("\nUnique locations:")
print(df_transect[["lat", "lon"]].drop_duplicates().reset_index(drop=True))
print()
print(df_transect.head(9))

# %% [markdown]
# You can also use `spacing_km` instead of `num_points` to control density,
# or use the convenience wrapper `get_transect_data_from_coordinates`:

# %%
df_transect_spacing = client.get_transect_data_from_coordinates(
    coord_a=coord_a,
    coord_b=coord_b,
    start="2023-01-01",
    end="2023-01-05",
    params=["T2M"],
    spacing_km=200,  # one point every ~200 km
    max_workers=3,
)
print("Points fetched (spacing_km=200):", df_transect_spacing["lat"].nunique())

# %% [markdown]
# ### `get_multi_point_data` — arbitrary list of locations
#
# Pass a list of dicts with `lat` and `lon` keys (plus optional `name` and
# `elevation`). Returns a combined DataFrame and a list of any failed points.

# %%
locations = [
    {"lat": -23.31, "lon": -51.16, "name": "Londrina"},
    {"lat": -15.78, "lon": -47.93, "name": "Brasília"},
    {"lat": -30.03, "lon": -51.23, "name": "Porto Alegre"},
]

df_multi, failed = client.get_multi_point_data(
    points=locations,
    start="2023-01-01",
    end="2023-01-05",
    params=["T2M"],
    max_workers=3,
)

print("Multi-point shape:", df_multi.shape)
if failed:
    print("Failed points:", failed)

mean_temperatures = df_multi.groupby(["name"])["T2M"].mean()

print(mean_temperatures)
# %% [markdown]
# ---
# ## 6 · Data cleanup — `ensure_date_column`
#
# Raw datasets don't always have a column called `"date"`.
# `ensure_date_column` searches a list of candidate column names and
# normalises whatever it finds to a plain date string (`YYYY-MM-DD`).
# If no column matches it can fall back to the DataFrame's index.

# %%
# --- Scenario A: non-standard column name with timestamp strings ---
raw_a = pd.DataFrame(
    {
        "obs_date": [
            "2023-05-15 08:30:00",
            "2023-05-16 12:45:00",
            "2023-05-17 19:15:00",
        ],
        "T2M": [18.5, 20.2, 17.8],
    }
)

clean_a = ensure_date_column(
    raw_a, name="date", candidates=["obs_date", "measurement_time"], normalize=True
)
print("Before:")
print(raw_a)
print("\nAfter :")
print(clean_a)
print("dtype :", clean_a["date"].dtype)

# %%
# --- Scenario B: date lives in the DatetimeIndex, not in a column ---
raw_b = pd.DataFrame(
    {"RH2M": [65.0, 72.0, 58.0]},
    index=pd.date_range("2023-06-01", periods=3, freq="D"),
)

clean_b = ensure_date_column(raw_b, name="date", index_fallback=True, normalize=True)
print("Before:")
print(raw_b)
print("\nAfter :")
print(clean_b)

# %% [markdown]
# ---
# ## 7 · Logging
#
# `aidweather` logs internal operations (such as HTTP requests, cache hits/misses,
# rate limits, etc.) using Python's standard `logging` library.
# By default, a `NullHandler` is attached to prevent unwanted warning messages.
# You can check the default logging configuration, override directories, and
# easily enable logging to a file.

# %%
# Retrieve the resolved log configurations
log_cfg = config.logging_config()
print("Logging enabled in config :", log_cfg.get("enabled"))
print("Log file destination      :", log_cfg.get("filename"))
print("Log level                 :", log_cfg.get("level"))

# Fetching some data to trigger log messages
print("\nFetching data to trigger log entries...")
df_log = client.get_point_data(
    lat=lat,
    lon=lon,
    start="2023-01-01",
    end="2023-01-02",
    params=["T2M"],
)

# Show that the log file has been created and populated
log_path = Path(log_cfg.get("filename"))
if log_path.exists():
    print(f"Log file created/updated at: {log_path}")
    print("\nLast 3 lines of the log file:")
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines[-3:]:
            print("  " + line.strip())
else:
    print(f"Log file not found at: {log_path}")

# %% [markdown]
# ---
# ## 8 · One-liner bonus
#
# Get NASA POWER API data into a tidy dataframe in one line — point or regional:

# %%

# Grab a year of weather for any point on Earth:
df_oneliner = PowerClient().get_point_data(
    lat=-23.55,
    lon=-46.63,
    start="2023-01-01",
    end="2023-12-31",
    params=["T2M", "PRECTOTCORR"],
)

print("\nShape:", df_oneliner.shape)
print("\nHead:")
print(df_oneliner.head())
print("\nDescribe:")
print(df_oneliner.describe())

# %% [markdown]
# Or a regional grid over a bounding box (one parameter per request,
# ≤ 4.5° × 4.5°, returned as a 0.5° × 0.5° grid):

# %%
df_regional = client.get_regional_data(
    lat_min=-23.5,
    lat_max=-20.0,
    lon_min=-47.0,
    lon_max=-44.0,
    start="2023-01-01",
    end="2023-01-05",
    params=["T2M"],
)

print("\nRegional shape:", df_regional.shape)
print(df_regional.head())


# %% [markdown]
# ---
# ## Summary

#
# | Component | What it does |
# |-----------|--------------|
# | `cfg` / `get_config()` | Read API endpoint URLs and parameter metadata from config |
# | `GeoCoordinate` | Parse, validate and convert geographic coordinates |
# | `normalize_coord_input` | One-call parsing for mixed float/string coordinate tuples |
# | `PowerClient.get_point_data` | Fetch daily or hourly data for one location |
# | `PowerClient.get_multi_point_data` | Fetch for multiple locations in parallel |
# | `PowerClient.get_transect_data` | Fetch a 1D transect between two `GeoCoordinate` endpoints |
# | `PowerClient.get_transect_data_from_coordinates` | Convenience wrapper for transects using two corner coords |
# | `PowerClient.get_regional_data` | Fetch a 0.5° grid within a bounding box (1 param, daily only) |
# | `PowerClient.get_regional_data_from_coordinates` | Regional fetch using two corner `GeoCoordinate` objects |
# | `PowerClient.summarize` | Print a formatted overview of any fetched DataFrame |
# | `ensure_date_column` | Normalise date columns in any DataFrame |
#
# All fetched data is cached locally. Subsequent calls for the same
# location and date range return immediately from disk.

# %%
