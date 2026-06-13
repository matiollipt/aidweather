# Climate-Risk Screening for Brazilian State Capitals

This flagship example demonstrates how to retrieve NASA POWER daily environmental parameters, calculate annual agroclimatic and extreme precipitation indicators, perform robust long-term trend analysis, and calculate comparative climate risk screening metrics for all 27 Brazilian state capitals.

The implementation is located at:
- CLI Executable Script: [brazil_capitals_climate_risk.py](file:///home/clever/aidbio/dev/aidweather/examples/brazil/brazil_capitals_climate_risk.py)
- Interactive Notebook-style Walkthrough: [03_brazil_capitals_climate_risk.py](file:///home/clever/aidbio/dev/aidweather/notebooks/brazil/03_brazil_capitals_climate_risk.py)
- Geographic Coordinates Dataset: [brazil_state_capitals.csv](file:///home/clever/aidbio/dev/aidweather/examples/brazil/data/brazil_state_capitals.csv)

---

## 1. Scientific Methodology & Calculations

### Daily Vapor Pressure Deficit (VPD)
Vapor Pressure Deficit (VPD, in kPa) is a key agroclimatic parameter that indicates the drying capacity of the air. It is calculated daily using the Tetens formula:

1. **Saturation Vapor Pressure ($e_s$, in kPa)** as a function of air temperature ($T$ in °C):
   $$e_s(T) = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
2. **Actual Vapor Pressure ($e_a$, in kPa)** as a function of relative humidity ($RH$ in %):
   $$e_a = e_s(T) \times \frac{RH}{100}$$
3. **Vapor Pressure Deficit (VPD, in kPa)**:
   $$VPD = e_s(T) - e_a = e_s(T) \times \left(1 - \frac{RH}{100}\right)$$

Daily VPD values are computed for every day containing valid temperature (`T2M`) and relative humidity (`RH2M`) measurements, and then averaged over each calendar year to compute the annual metric `mean_vpd_kpa`.

### Annual Precipitation Metrics
To screen for drought and deluge hazards, the script compiles:
- **`annual_precip_mm`**: The sum of daily corrected precipitation (`PRECTOTCORR`).
- **`rainy_days_ge_1mm`**: Count of days with daily precipitation $\ge 1.0\text{ mm}$.
- **`heavy_rain_days_ge_20mm`**: Count of days with daily precipitation $\ge 20.0\text{ mm}$.
- **`extreme_rain_days_ge_50mm`**: Count of days with daily precipitation $\ge 50.0\text{ mm}$.
- **`dry_days_lt_1mm`**: Count of days with daily precipitation $< 1.0\text{ mm}$.
- **`longest_dry_spell_days`**: The maximum number of consecutive days with daily precipitation $< 1.0\text{ mm}$ within the calendar year.
- **`max_1day_precip_mm`**: Maximum daily precipitation amount observed.
- **`max_5day_precip_mm`**: Maximum 5-day rolling sum of daily precipitation (calculated continuously per capital to prevent year-boundary bleeding).
- **`rainfall_concentration`**: Ratio of maximum 5-day rainfall to total annual rainfall:
  $$\text{rainfall\_concentration} = \frac{\max\text{ 5-day rainfall}}{\text{annual rainfall}}$$

### Long-Term Trend Fitting
For each capital and annual metric, the script calculates a linear slope over the years (from 1981 onward).
- **Theil-Sen Estimator**: The script attempts to use `scipy.stats.theilslopes` if `scipy` is installed. Theil-Sen is a non-parametric estimator that computes the median of all slopes between pairs of points. It is highly robust to outliers and extreme climatic anomalies.
- **OLS Linear Regression**: If `scipy` is not available, the script falls back to standard Ordinary Least Squares using `numpy.polyfit(..., degree=1)`.

### Risk Screening Scores
To construct comparative risk indexes, trend slopes are normalized across all capitals using **Z-scores**:
$$Z(C, M) = \frac{\text{Slope}(C, M) - \mu_M}{\sigma_M}$$
where $\mu_M$ is the average slope of metric $M$ across all capitals, and $\sigma_M$ is the standard deviation.

Two auditable indices are computed for each capital:
1. **`desertification_pressure_score`**: Calculates relative pressure toward drying conditions.
   $$\text{desertification\_pressure\_score} = Z_{\text{mean\_t2m\_c}} + Z_{\text{mean\_t2m\_max\_c}} + Z_{\text{dry\_days\_lt\_1mm}} + Z_{\text{longest\_dry\_spell\_days}} - Z_{\text{annual\_precip\_mm}} - Z_{\text{mean\_rh2m\_pct}}$$
2. **`flooding_pressure_score`**: Calculates relative pressure toward extreme precipitation/deluge.
   $$\text{flooding\_pressure\_score} = Z_{\text{heavy\_rain\_days\_ge\_20mm}} + Z_{\text{extreme\_rain\_days\_ge\_50mm}} + Z_{\text{max\_1day\_precip\_mm}} + Z_{\text{max\_5day\_precip\_mm}} + Z_{\text{rainfall\_concentration}}$$

> [!TIP]
> If any component trend is missing or fails to compute, the script emits a warning and skips the component, calculating the score over the remaining components (using pandas' `sum(min_count=1)`).

---

## 2. Command Line Interface (CLI) Usage

The script is a self-contained CLI. Executing it retrieves data from NASA POWER and runs the full risk assessment pipeline.

```bash
python examples/brazil/brazil_capitals_climate_risk.py [OPTIONS]
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--start` | Start date for query (format YYYY-MM-DD). | `"1981-01-01"` |
| `--end` | End date for query (format YYYY-MM-DD). | Today minus 5 days |
| `--out-dir` | Directory to save generated datasets and reports. | `"outputs/brazil_capitals"` |
| `--format` | Output format for daily/annual tables: `parquet` or `csv`. | `"parquet"` |
| `--max-workers` | Maximum concurrent threads to speed up fetches. | `5` |
| `--sample` | Fetch only 3 capitals (Brasília, São Paulo, Manaus) and only for the year 2020. Useful for fast validation. | `False` |
| `--force` | Force overwrite of daily output files, querying the API again. | `False` |
| `--no-fetch` | Recompute annual metrics, trends, and risk scores using the existing daily output file. | `False` |

### Sample Execution

To test the installation quickly without making massive API queries:
```bash
python examples/brazil/brazil_capitals_climate_risk.py --sample --format csv
```

To run the full diagnostic from 1981 to near real-time:
```bash
python examples/brazil/brazil_capitals_climate_risk.py --format parquet
```

---

## 3. Scientific Caveats & Future Directions

> [!WARNING]
> The state capitals analyzed here serve as a national demonstration sample for the `aidweather` package. This analysis does **not** constitute a definitive diagnosis of desertification or flooding vulnerability. State capitals are highly localized points and do not represent the geographic diversity of their respective states.
>
> Later versions of regional assessments should target:
> - **Semiarid Polygons**: Specific arid regions in Northeast Brazil.
> - **Drought-Prone Municipalities**: Rural farming districts sensitive to water supply failure.
> - **Flood-Prone Watersheds & River Basins**: Bounding polygons that capture runoff accumulation.
> - **NASA POWER Regional Grids**: Using regional grid-bounding operations rather than single coordinate points.
