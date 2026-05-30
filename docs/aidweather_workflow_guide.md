# AidWeather Workflow Guide

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## From geographical points to validated weather data

**Starting point:** agricultural analysis requires robust meteorological data.

Examples:
* Enriching farm harvest records with corresponding temperature and precipitation data.
* Collecting historical weather baselines for specific agricultural plots.
* Building a reliable foundational weather dataset for downstream ML or agronomic modeling.

AidWeather is built to turn geographic coordinates and time ranges into a structured dataset. The package is modular and focused: the `PowerClient` handles weather acquisition, the `geo` module normalizes location data, and the `utils` module provides time series standardization.

## Workflow by module

### 1. Standardize coordinates and configuration

Before weather fetching, AidWeather normalizes location and shared settings. The `geo` module provides robust parsing and validation for latitude and longitude in multiple formats, ensuring all requests are perfectly formed. The `config` module centralizes API URLs, parameter mappings, and cache settings. This reduces friction and inconsistency when scaling across many requests.

### 2. Fetch external weather with NASA POWER

The `PowerClient` retrieves daily or hourly weather from NASA POWER, with retries, local SQLite caching, gzip compression, and support for point and multi-point retrieval. This allows users to easily pull historical data in a robust, production-ready manner, avoiding transient network issues.

**Typical weather variables**
* temperature (T2M, T2M_MAX, T2M_MIN)
* relative humidity (RH2M)
* precipitation (PRECTOTCORR)
* solar radiation (ALLSKY_SFC_PAR_TOT)
* wind (WS10M)
* other POWER parameters relevant to crop development.

### 3. Provide standardized output for downstream systems

The data is returned as a clean, standardized `pandas` DataFrame with a validated datetime index (managed by `ensure_date_column` in the `utils` module). This ensures that the dataset is ready to be merged directly with internal farm data or passed along to feature engineering, exploratory data analysis (EDA), and machine learning pipelines in other packages.

## End-to-end flow

**Coordinates & Date Range → Coordinate Normalization → NASA POWER Fetch (w/ Caching & Retries) → Standardized Time Series DataFrame**

A representative pipeline script in the project uses the `PowerClient` to fetch weather, leveraging the `geo` module to normalize points of interest.

## Practical outcome

AidWeather is best understood as a **foundational data ingestion package** for agricultural analytics:

* **Technical-facing**: provides a resilient, caching, parallelized client to acquire NASA POWER data effortlessly.
* **Business-facing**: ensures that downstream analysis and models are built upon consistent and validated meteorological baselines.
