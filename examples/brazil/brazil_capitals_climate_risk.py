# %% [markdown]
# # Climate-Risk Screening for Brazilian State Capitals
#
# This flagship example demonstrates how to retrieve NASA POWER daily environmental parameters, calculate annual agroclimatic and extreme precipitation indicators, perform robust long-term trend analysis, and calculate comparative climate risk screening metrics for all 27 Brazilian state capitals.
#
# > [!WARNING]
# > The state capitals analyzed here serve as a national demonstration sample for the `aidweather` package. This analysis does **not** constitute a definitive diagnosis of desertification or flooding vulnerability. State capitals are highly localized points and do not represent the geographic diversity of their respective states.

# %%
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aidweather import PowerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("brazil_capitals_risk")

# Attempt scipy import for trend computation
try:
    from scipy.stats import theilslopes

    HAS_SCIPY = True
    print("Scipy available: Theil-Sen estimator will be used for trends.")
except ImportError:
    HAS_SCIPY = False
    print("Scipy not available: Falling back to numpy.polyfit for trends.")

# NASA POWER Daily parameters to fetch
PARAMS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2M_RANGE",
    "T2MDEW",
    "T2MWET",
    "RH2M",
    "QV2M",
    "TS",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WS10M",
    "WS50M",
    "WS50M_DIR",
]

# %% [markdown]
# ## 1. Geolocation Data Setup
#
# We load the coordinates of the 27 state capitals of Brazil from our CSV catalog.
# The table contains the `state_code`, `state`, `capital`, `region`, `lat`, and `lon` for each location.

# %%
if __name__ == "__main__":
    # Read state capitals metadata
    script_dir = Path(__file__).resolve().parent
    capitals_path = script_dir / "data" / "brazil_state_capitals.csv"

    capitals_df = pd.read_csv(capitals_path)
    print(f"Loaded {len(capitals_df)} state capitals.")
    print("\nFirst 5 locations:")
    print(capitals_df.head())

# %% [markdown]
# ## 2. Fetching Daily Weather
#
# We use the `PowerClient` daily point API to fetch data in parallel. We choose a set of parameters that capture temperature, humidity, wind, and solar radiation profiles.
# We will fetch data from 1981-01-01 to near real-time for all 27 capitals.

# %%
if __name__ == "__main__":
    start_date = "1981-01-01"
    end_date = (date.today() - timedelta(days=5)).isoformat()

    points_input = []
    for _, row in capitals_df.iterrows():
        points_input.append(
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "name": str(row["capital"]),
            }
        )

    print(
        f"\nFetching daily weather data from {start_date} to {end_date} for {len(points_input)} capitals..."
    )

    client = PowerClient(temporal_api="daily")

    # For the actual API call, map WS50M_DIR to WD50M as required by NASA POWER daily point API
    api_params = [p if p != "WS50M_DIR" else "WD50M" for p in PARAMS]

    # Fetch multi-point in parallel
    fetched_df, failed = client.get_multi_point_data(
        points=points_input,
        start=start_date,
        end=end_date,
        params=api_params,
        max_workers=5,
    )

    if not fetched_df.empty and "WD50M" in fetched_df.columns:
        fetched_df = fetched_df.rename(columns={"WD50M": "WS50M_DIR"})

    print(f"Fetch complete. Failed points: {len(failed)}")

    # Merge daily data with original capital metadata to preserve all requested metadata
    daily_df = fetched_df.merge(
        capitals_df,
        left_on=["lat", "lon"],
        right_on=["lat", "lon"],
        suffixes=("_fetched", ""),
    )

    # Reorder columns
    desired_order = [
        "date",
        "state_code",
        "state",
        "capital",
        "region",
        "lat",
        "lon",
    ] + PARAMS
    daily_df = daily_df[desired_order].sort_values(["capital", "date"])

    print("\nDaily Dataset Sample:")
    print(daily_df.head())

# %% [markdown]
# ## 3. Scientific Methodology & Calculations
#
# ### Daily Vapor Pressure Deficit (VPD)
# Vapor Pressure Deficit (VPD, in kPa) is a key agroclimatic parameter that indicates the drying capacity of the air. It is calculated daily using the Tetens formula:
#
# 1. **Saturation Vapor Pressure ($e_s$, in kPa)** as a function of air temperature ($T$ in °C):
#    $$e_s(T) = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
# 2. **Actual Vapor Pressure ($e_a$, in kPa)** as a function of relative humidity ($RH$ in %):
#    $$e_a = e_s(T) \times \frac{RH}{100}$$
# 3. **Vapor Pressure Deficit (VPD, in kPa)**:
#    $$VPD = e_s(T) - e_a = e_s(T) \times \left(1 - \frac{RH}{100}\right)$$
#
# ### Annual Precipitation Metrics
# To screen for drought and deluge hazards, the script compiles:
# - **`annual_precip_mm`**: The sum of daily corrected precipitation.
# - **`rainy_days_ge_1mm`**: Count of days with daily precipitation $\ge 1.0\text{ mm}$.
# - **`heavy_rain_days_ge_20mm`**: Count of days with daily precipitation $\ge 20.0\text{ mm}$.
# - **`extreme_rain_days_ge_50mm`**: Count of days with daily precipitation $\ge 50.0\text{ mm}$.
# - **`dry_days_lt_1mm`**: Count of days with daily precipitation $< 1.0\text{ mm}$.
# - **`longest_dry_spell_days`**: The maximum number of consecutive days with daily precipitation $< 1.0\text{ mm}$ within the calendar year.
# - **`max_1day_precip_mm`**: Maximum daily precipitation amount observed.
# - **`max_5day_precip_mm`**: Maximum 5-day rolling sum of daily precipitation.
# - **`rainfall_concentration`**: Ratio of maximum 5-day rainfall to total annual rainfall.


# %%
def compute_longest_dry_spell(precip_series: pd.Series) -> int:
    is_dry = (precip_series < 1.0) & precip_series.notna()
    consec = is_dry.groupby((~is_dry).cumsum()).sum()
    if consec.empty:
        return 0
    return int(consec.max())


def compute_vpd(t2m: pd.Series, rh2m: pd.Series) -> pd.Series:
    es = 0.61078 * np.exp((17.27 * t2m) / (t2m + 237.3))
    ea = es * (rh2m / 100.0)
    vpd = es - ea
    return vpd.clip(lower=0.0)


def compute_annual_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["datetime"] = pd.to_datetime(df["date"])
    df["year"] = df["datetime"].dt.year

    df["vpd_daily"] = np.nan
    has_vpd_inputs = df["T2M"].notna() & df["RH2M"].notna()
    if has_vpd_inputs.any():
        df.loc[has_vpd_inputs, "vpd_daily"] = compute_vpd(
            df.loc[has_vpd_inputs, "T2M"], df.loc[has_vpd_inputs, "RH2M"]
        )

    df = df.sort_values(["capital", "datetime"])
    df["roll_5d_precip"] = df.groupby("capital")["PRECTOTCORR"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )

    group_cols = ["state_code", "state", "capital", "region", "lat", "lon", "year"]
    records = []

    for keys, group in df.groupby(group_cols):
        precip = group["PRECTOTCORR"]
        annual_precip = precip.sum()
        max_5day = group["roll_5d_precip"].max()

        if pd.isna(max_5day) or annual_precip <= 0:
            rainfall_concat = 0.0
        else:
            rainfall_concat = float(max_5day / annual_precip)

        rec = {
            "state_code": keys[0],
            "state": keys[1],
            "capital": keys[2],
            "region": keys[3],
            "lat": keys[4],
            "lon": keys[5],
            "year": int(keys[6]),
            "annual_precip_mm": float(annual_precip),
            "rainy_days_ge_1mm": int((precip >= 1.0).sum()),
            "heavy_rain_days_ge_20mm": int((precip >= 20.0).sum()),
            "extreme_rain_days_ge_50mm": int((precip >= 50.0).sum()),
            "dry_days_lt_1mm": int((precip < 1.0).sum()),
            "longest_dry_spell_days": compute_longest_dry_spell(precip),
            "max_1day_precip_mm": (
                float(precip.max()) if not precip.isna().all() else np.nan
            ),
            "max_5day_precip_mm": float(max_5day) if not pd.isna(max_5day) else np.nan,
            "rainfall_concentration": rainfall_concat,
            "mean_t2m_c": (
                float(group["T2M"].mean()) if not group["T2M"].isna().all() else np.nan
            ),
            "mean_t2m_max_c": (
                float(group["T2M_MAX"].mean())
                if not group["T2M_MAX"].isna().all()
                else np.nan
            ),
            "mean_t2m_min_c": (
                float(group["T2M_MIN"].mean())
                if not group["T2M_MIN"].isna().all()
                else np.nan
            ),
            "mean_t2m_range_c": (
                float(group["T2M_RANGE"].mean())
                if not group["T2M_RANGE"].isna().all()
                else np.nan
            ),
            "mean_rh2m_pct": (
                float(group["RH2M"].mean())
                if not group["RH2M"].isna().all()
                else np.nan
            ),
            "mean_solar_kwh_m2_day": (
                float(group["ALLSKY_SFC_SW_DWN"].mean())
                if not group["ALLSKY_SFC_SW_DWN"].isna().all()
                else np.nan
            ),
            "mean_ws10m_m_s": (
                float(group["WS10M"].mean())
                if not group["WS10M"].isna().all()
                else np.nan
            ),
            "mean_ws50m_m_s": (
                float(group["WS50M"].mean())
                if not group["WS50M"].isna().all()
                else np.nan
            ),
            "mean_vpd_kpa": (
                float(group["vpd_daily"].mean())
                if not group["vpd_daily"].isna().all()
                else np.nan
            ),
        }
        records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("\nComputing annual metrics...")
    annual_df = compute_annual_metrics(daily_df)

    print("\nAnnual Metrics Sample:")
    print(
        annual_df[
            [
                "capital",
                "year",
                "annual_precip_mm",
                "longest_dry_spell_days",
                "mean_vpd_kpa",
            ]
        ].head()
    )

# %% [markdown]
# ## 4. Long-Term Trend Fitting
#
# For each capital and annual metric, we calculate a linear slope over the years (from 1981 onward).
# - **Theil-Sen Estimator**: A non-parametric estimator that computes the median of all slopes between pairs of points. It is highly robust to outliers.
# - **OLS Linear Regression**: Falls back to standard Ordinary Least Squares if scipy is unavailable.


# %%
def compute_trend(years: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    mask = ~np.isnan(values)
    x = years[mask]
    y = values[mask]

    n_years = len(x)
    if n_years < 2:
        return {
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "intercept": np.nan,
            "method": "none",
            "n_years": n_years,
            "first_year": int(years[0]) if len(years) > 0 else np.nan,
            "last_year": int(years[-1]) if len(years) > 0 else np.nan,
        }

    first_year = int(x[0])
    last_year = int(x[-1])

    if HAS_SCIPY:
        try:
            res = theilslopes(y, x)
            slope = float(res[0])
            intercept = float(res[1])
            method = "theilslopes"
        except Exception as e:
            logger.debug(f"Theil-Sen estimation failed, falling back to polyfit: {e}")
            slope, intercept = np.polyfit(x, y, 1)
            method = "polyfit"
    else:
        slope, intercept = np.polyfit(x, y, 1)
        method = "polyfit"

    return {
        "slope_per_year": slope,
        "slope_per_decade": slope * 10.0,
        "intercept": intercept,
        "method": method,
        "n_years": n_years,
        "first_year": first_year,
        "last_year": last_year,
    }


def compute_trends_dataframe(annual_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "annual_precip_mm",
        "rainy_days_ge_1mm",
        "heavy_rain_days_ge_20mm",
        "extreme_rain_days_ge_50mm",
        "dry_days_lt_1mm",
        "longest_dry_spell_days",
        "max_1day_precip_mm",
        "max_5day_precip_mm",
        "rainfall_concentration",
        "mean_t2m_c",
        "mean_t2m_max_c",
        "mean_t2m_min_c",
        "mean_t2m_range_c",
        "mean_rh2m_pct",
        "mean_solar_kwh_m2_day",
        "mean_ws10m_m_s",
        "mean_ws50m_m_s",
        "mean_vpd_kpa",
    ]

    records = []
    group_cols = ["state_code", "state", "capital", "region", "lat", "lon"]
    for keys, group in annual_df.groupby(group_cols):
        group = group.sort_values("year")
        years = group["year"].values
        for metric in metrics:
            values = group[metric].values
            trend_info = compute_trend(years, values)
            rec = {
                "state_code": keys[0],
                "state": keys[1],
                "capital": keys[2],
                "region": keys[3],
                "lat": keys[4],
                "lon": keys[5],
                "metric": metric,
                **trend_info,
            }
            records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("\nComputing trend metrics for all indicators...")
    trends_df = compute_trends_dataframe(annual_df)

    print("\nTrends Dataset Sample (Slope per Decade):")
    sample_trends = trends_df[["capital", "metric", "slope_per_decade", "method"]].head(
        10
    )
    print(sample_trends)

# %% [markdown]
# ## 5. Risk Screening Scores
#
# To construct comparative risk indexes, trend slopes are normalized across all capitals using **Z-scores**:
# $$Z(C, M) = \frac{\text{Slope}(C, M) - \mu_M}{\sigma_M}$$
# where $\mu_M$ is the average slope of metric $M$ across all capitals, and $\sigma_M$ is the standard deviation.
#
# Two auditable indices are computed for each capital:
# 1. **`desertification_pressure_score`**: Calculates relative pressure toward drying conditions.
#    $$\text{desertification\_pressure\_score} = Z_{\text{mean\_t2m\_c}} + Z_{\text{mean\_t2m\_max\_c}} + Z_{\text{dry\_days\_lt\_1mm}} + Z_{\text{longest\_dry\_spell\_days}} - Z_{\text{annual\_precip\_mm}} - Z_{\text{mean\_rh2m\_pct}}$$
# 2. **`flooding_pressure_score`**: Calculates relative pressure toward extreme precipitation/deluge.
#    $$\text{flooding\_pressure\_score} = Z_{\text{heavy\_rain\_days\_ge\_20mm}} + Z_{\text{extreme\_rain\_days\_ge\_50mm}} + Z_{\text{max\_1day\_precip\_mm}} + Z_{\text{max\_5day\_precip\_mm}} + Z_{\text{rainfall\_concentration}}$$


# %%
def compute_risk_screening_scores(trends_df: pd.DataFrame) -> pd.DataFrame:
    slopes_df = trends_df.pivot(
        index=["state_code", "state", "capital", "region", "lat", "lon"],
        columns="metric",
        values="slope_per_year",
    ).reset_index()

    des_pos = [
        "mean_t2m_c",
        "mean_t2m_max_c",
        "dry_days_lt_1mm",
        "longest_dry_spell_days",
    ]
    des_neg = ["annual_precip_mm", "mean_rh2m_pct"]
    flood_pos = [
        "heavy_rain_days_ge_20mm",
        "extreme_rain_days_ge_50mm",
        "max_1day_precip_mm",
        "max_5day_precip_mm",
        "rainfall_concentration",
    ]

    z_cols = {}
    for m in des_pos + des_neg + flood_pos:
        if m in slopes_df.columns:
            series = slopes_df[m]
            mean_val = series.mean()
            std_val = series.std(ddof=0)

            if pd.isna(std_val) or std_val == 0:
                z_scores = pd.Series(0.0, index=series.index)
            else:
                z_scores = (series - mean_val) / std_val

            z_cols[f"z_{m}"] = z_scores
            slopes_df[f"slope_{m}"] = series
        else:
            warnings.warn(
                f"Metric '{m}' is missing from trend calculations. Skipped in risk scoring.",
                UserWarning,
                stacklevel=2,
            )
            z_cols[f"z_{m}"] = pd.Series(np.nan, index=slopes_df.index)
            slopes_df[f"slope_{m}"] = np.nan

    for col, z_series in z_cols.items():
        slopes_df[col] = z_series

    # Calculate Desertification Pressure Score: + (positive indices) - (negative indices)
    des_terms = []
    for m in des_pos:
        des_terms.append(slopes_df[f"z_{m}"])
    for m in des_neg:
        des_terms.append(-slopes_df[f"z_{m}"])
    des_df = pd.concat(des_terms, axis=1)
    slopes_df["desertification_pressure_score"] = des_df.sum(axis=1, min_count=1)

    # Calculate Flooding Pressure Score: sum of positive heavy rain indices
    flood_terms = []
    for m in flood_pos:
        flood_terms.append(slopes_df[f"z_{m}"])
    flood_df = pd.concat(flood_terms, axis=1)
    slopes_df["flooding_pressure_score"] = flood_df.sum(axis=1, min_count=1)

    output_cols = [
        "state_code",
        "state",
        "capital",
        "region",
        "lat",
        "lon",
        "desertification_pressure_score",
        "flooding_pressure_score",
    ]
    for m in des_pos + des_neg + flood_pos:
        output_cols.append(f"slope_{m}")
        output_cols.append(f"z_{m}")

    return slopes_df[output_cols]


if __name__ == "__main__":
    print("\nComputing Climate Risk Screening Scores...")
    scores_df = compute_risk_screening_scores(trends_df)

    print("\n=======================================================")
    print("FINAL RISK SCORES FOR ALL BRAZILIAN CAPITALS")
    print("=======================================================")
    final_output = scores_df[
        [
            "capital",
            "state_code",
            "region",
            "desertification_pressure_score",
            "flooding_pressure_score",
        ]
    ].sort_values("desertification_pressure_score", ascending=False)
    print(final_output.to_string(index=False))

# %% [markdown]
# ---
# **End of Tutorial**
# This script fetched actual daily parameters, computed long-term trends from 1981, and generated auditable scores for risk assessment without persisting any intermediate files.

# %%
