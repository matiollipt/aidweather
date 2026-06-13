# SPDX-License-Identifier: Apache-2.0
# Load 03_brazil_capitals_climate_risk dynamically to bypass the invalid identifier issue (starting with a number)
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

example_path = Path(__file__).parent / "brazil_capitals_climate_risk.py"
spec = importlib.util.spec_from_file_location("bcr", str(example_path))
bcr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcr)


def test_capitals_csv_integrity():
    """
    Test that the state capitals CSV exists, contains exactly 27 rows,
    has no missing coordinates, and coordinates are within Brazil's broad bounding box.
    """
    csv_path = Path(__file__).parent / "data" / "brazil_state_capitals.csv"
    assert csv_path.exists(), f"Capitals CSV not found at {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 27, f"Expected 27 capitals, found {len(df)}"

    required_cols = {"state_code", "state", "capital", "region", "lat", "lon"}
    assert required_cols.issubset(df.columns), f"Missing required columns in CSV: {df.columns}"

    # Verify no missing lat/lon values
    assert df["lat"].isna().sum() == 0, "Latitude contains null values"
    assert df["lon"].isna().sum() == 0, "Longitude contains null values"

    # Brazil's broad bounding box: Latitudes in [-35.0, 6.0], Longitudes in [-75.0, -34.0]
    for _, row in df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        capital = row["capital"]
        assert -35.0 <= lat <= 6.0, f"{capital} Latitude {lat} is outside Brazilian boundaries (-35 to +6)"
        assert -75.0 <= lon <= -34.0, f"{capital} Longitude {lon} is outside Brazilian boundaries (-75 to -34)"


def test_longest_dry_spell():
    """Test longest dry spell computation on known daily precipitation patterns."""
    # Pattern: 2 dry, 1 wet, 3 dry, 1 wet, 1 dry
    precip = pd.Series([0.0, 0.5, 5.0, 0.0, 0.0, 0.2, 12.0, 0.0])
    # Dry days are < 1.0. Longest consecutive is 3 days (0.0, 0.0, 0.2)
    assert bcr.compute_longest_dry_spell(precip) == 3

    # All dry days
    assert bcr.compute_longest_dry_spell(pd.Series([0.0, 0.0, 0.0])) == 3

    # All wet days
    assert bcr.compute_longest_dry_spell(pd.Series([2.0, 10.0, 1.5])) == 0

    # Handling NaNs: NaNs should not be counted as dry
    precip_nan = pd.Series([0.0, 0.0, np.nan, 0.0, 2.0])
    assert bcr.compute_longest_dry_spell(precip_nan) == 2


def test_vpd_calculation():
    """Test Tetens Vapor Pressure Deficit calculation against known analytical values."""
    # For T = 25 C and RH = 60%:
    # es = 0.61078 * exp(17.27 * 25 / (25 + 237.3)) = 3.1671 kPa
    # ea = 3.1671 * 0.6 = 1.9003 kPa
    # VPD = 3.1671 - 1.9003 = 1.2668 kPa
    t = pd.Series([25.0])
    rh = pd.Series([60.0])
    vpd = bcr.compute_vpd(t, rh)
    assert len(vpd) == 1
    assert pytest.approx(vpd.iloc[0], rel=1e-3) == 1.2668

    # RH >= 100% should lead to 0 VPD (clipped)
    t_sat = pd.Series([20.0])
    rh_sat = pd.Series([105.0])
    vpd_sat = bcr.compute_vpd(t_sat, rh_sat)
    assert vpd_sat.iloc[0] == 0.0


def test_annual_metrics_aggregation():
    """Test daily aggregations into annual metrics for a synthetic year."""
    # Create synthetic daily data for 1 capital for 1 year (365 days)
    dates = pd.date_range("2020-01-01", "2020-12-31")
    n_days = len(dates)
    
    # 365 days of base parameters
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "state_code": ["SP"] * n_days,
        "state": ["São Paulo"] * n_days,
        "capital": ["São Paulo"] * n_days,
        "region": ["Sudeste"] * n_days,
        "lat": [-23.55] * n_days,
        "lon": [-46.63] * n_days,
        "T2M": [20.0] * n_days,
        "T2M_MAX": [25.0] * n_days,
        "T2M_MIN": [15.0] * n_days,
        "T2M_RANGE": [10.0] * n_days,
        "T2MDEW": [12.0] * n_days,
        "T2MWET": [15.0] * n_days,
        "RH2M": [50.0] * n_days,
        "QV2M": [0.007] * n_days,
        "TS": [21.0] * n_days,
        "PRECTOTCORR": [0.5] * n_days,  # Base: dry day (< 1mm)
        "ALLSKY_SFC_SW_DWN": [5.0] * n_days,
        "WS10M": [3.0] * n_days,
        "WS50M": [4.0] * n_days,
        "WS50M_DIR": [180.0] * n_days
    })
    
    # Inject heavy and extreme rainfall events
    # Day 100: 25.0 mm (heavy rain >= 20)
    # Day 200: 60.0 mm (extreme rain >= 50)
    df.loc[100, "PRECTOTCORR"] = 25.0
    df.loc[200, "PRECTOTCORR"] = 60.0
    
    annual = bcr.compute_annual_metrics(df)
    assert len(annual) == 1
    row = annual.iloc[0]
    
    assert row["year"] == 2020
    assert row["state_code"] == "SP"
    # Total rain: 364 * 0.5 + 25.0 + 60.0 = 267.0
    assert row["annual_precip_mm"] == pytest.approx(267.0)
    # Rainy days (>=1.0): day 100 (25.0) and day 200 (60.0) -> 2 days
    assert row["rainy_days_ge_1mm"] == 2
    assert row["heavy_rain_days_ge_20mm"] == 2
    assert row["extreme_rain_days_ge_50mm"] == 1
    assert row["dry_days_lt_1mm"] == 364
    
    # Check max 1-day precip
    assert row["max_1day_precip_mm"] == 60.0
    
    # Check mean temperature aggregates
    assert row["mean_t2m_c"] == 20.0
    assert row["mean_t2m_max_c"] == 25.0
    assert row["mean_t2m_min_c"] == 15.0


def test_trend_fallback_computation():
    """Test trend slope fitting and fallback path when scipy is absent."""
    years = np.array([2000, 2001, 2002, 2003, 2004])
    values = np.array([10.0, 12.0, 14.0, 16.0, 18.0])  # slope = 2.0
    
    # Standard computation (with or without scipy)
    res = bcr.compute_trend(years, values)
    assert res["slope_per_year"] == pytest.approx(2.0)
    assert res["slope_per_decade"] == pytest.approx(20.0)
    assert res["intercept"] == pytest.approx(-3990.0)
    assert res["n_years"] == 5
    assert res["first_year"] == 2000
    assert res["last_year"] == 2004
    
    # Force numpy fallback path by patching bcr.HAS_SCIPY = False
    original_has_scipy = bcr.HAS_SCIPY
    try:
        bcr.HAS_SCIPY = False
        res_fallback = bcr.compute_trend(years, values)
        assert res_fallback["slope_per_year"] == pytest.approx(2.0)
        assert res_fallback["method"] == "polyfit"
    finally:
        bcr.HAS_SCIPY = original_has_scipy

    # Test with fewer than 2 years of valid data
    res_short = bcr.compute_trend(np.array([2000]), np.array([10.0]))
    assert np.isnan(res_short["slope_per_year"])
    assert res_short["method"] == "none"


def test_risk_screening_scores():
    """Test Z-score scaling and pressure index calculations on synthetic slopes."""
    # Create synthetic trends for 3 capitals (DF, SP, AM)
    # The component lists
    metrics = [
        "mean_t2m_c", "mean_t2m_max_c", "dry_days_lt_1mm", "longest_dry_spell_days",
        "annual_precip_mm", "mean_rh2m_pct", "heavy_rain_days_ge_20mm",
        "extreme_rain_days_ge_50mm", "max_1day_precip_mm", "max_5day_precip_mm",
        "rainfall_concentration"
    ]
    
    records = []
    # Capital 1: DF
    for m in metrics:
        # positive indicators high for desertification, negative high (meaning dry/warming)
        val = 1.0 if m in ["mean_t2m_c", "longest_dry_spell_days"] else -1.0
        records.append({
            "state_code": "DF", "state": "Distrito Federal", "capital": "Brasília",
            "region": "Centro-Oeste", "lat": -15.79, "lon": -47.88,
            "metric": m, "slope_per_year": val, "slope_per_decade": val * 10,
            "intercept": 0.0, "method": "polyfit", "n_years": 10,
            "first_year": 2000, "last_year": 2009
        })
    # Capital 2: SP
    for m in metrics:
        val = 0.0
        records.append({
            "state_code": "SP", "state": "São Paulo", "capital": "São Paulo",
            "region": "Sudeste", "lat": -23.55, "lon": -46.63,
            "metric": m, "slope_per_year": val, "slope_per_decade": val * 10,
            "intercept": 0.0, "method": "polyfit", "n_years": 10,
            "first_year": 2000, "last_year": 2009
        })
    # Capital 3: AM
    for m in metrics:
        val = -1.0 if m in ["mean_t2m_c", "longest_dry_spell_days"] else 1.0
        records.append({
            "state_code": "AM", "state": "Amazonas", "capital": "Manaus",
            "region": "Norte", "lat": -3.12, "lon": -60.02,
            "metric": m, "slope_per_year": val, "slope_per_decade": val * 10,
            "intercept": 0.0, "method": "polyfit", "n_years": 10,
            "first_year": 2000, "last_year": 2009
        })
        
    trends_df = pd.DataFrame(records)
    scores = bcr.compute_risk_screening_scores(trends_df)
    
    assert len(scores) == 3
    assert "desertification_pressure_score" in scores.columns
    assert "flooding_pressure_score" in scores.columns
    
    # Brasília should be the highest desertification score due to positive warming/dry spells and negative rain trends
    df_row = scores[scores["capital"] == "Brasília"].iloc[0]
    am_row = scores[scores["capital"] == "Manaus"].iloc[0]
    assert df_row["desertification_pressure_score"] > am_row["desertification_pressure_score"]
    
    # Manaus should be the highest flooding score due to positive heavy rainfall trends
    assert am_row["flooding_pressure_score"] > df_row["flooding_pressure_score"]
