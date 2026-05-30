# Test Coverage

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

This document summarizes the test coverage observed in the `tests/` directory. The test suite uses `pytest` and relies heavily on mocking for external dependencies (API calls, file I/O).

## Module Coverage

### 1. Client (`tests/test_client.py`, `tests/test_client_caching.py`, `tests/test_client_validation.py`, `tests/test_client_advanced.py`, `tests/test_client_additional.py`)
- **Coverage**: High.
- **Scope**:
  - **API Interaction**: Mocks `requests` to test daily/hourly parsing, error handling (500 errors), and partial data availability (latency handling).
  - **Caching**: Verification of SQLite caching hit/miss logic and edge cases.
  - **Validation & Advanced**: Edge case validation and integration of more complex requests.

### 2. Configuration (`tests/test_config.py`, `tests/test_config_extra.py`)
- **Coverage**: High.
- **Scope**:
  - `_Config` logic: Nested key retrieval, default fallbacks, parameter grouping.
  - **File I/O**: Tests loading of `config.json` and model configurations.
  - **Error Handling**: Tests behavior when config files are missing or invalid (corrupted JSON).

### 3. Geo (`tests/test_geo.py`, `tests/test_geo_extra.py`)
- **Coverage**: Complete.
- **Scope**:
  - **Parsing**: Exhaustive tests for DMS, DDM, and DD formats (lat/lon, N/S/E/W).
  - **Validation**: Bounds checking (-90..90, -180..180).
  - **Roundtrip**: Verifies that `parse -> format -> parse` preserves values.
  - **Normalization**: Tests `normalize_coord_input` with tuples, args, and strings.

### 4. Utils (`tests/test_utils.py`, `tests/test_utils_extra.py`)
- **Coverage**: High.
- **Scope**:
  - `ensure_date_column`: Verifies dataframe time standardization behavior and index fallbacks.

## Gaps & Observations
- The test suite provides robust validation for data retrieval and formatting, which is the core functionality of AidWeather. Further network integration tests could be used selectively to confirm API changes.
