# Assets

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

The `aidweather` package includes non-code assets located in `aidweather/assets/`. These files control configuration and model hyperparameters.

## 1. config.json

**Purpose**: The central configuration file for the library. It is loaded by `aidweather.config`.

**Key Sections**:
- **`base_urls`**: Defines the endpoints for NASA POWER API (daily/hourly, point/regional).
- **`params`**: Maps technical API parameter codes (e.g., "T2M") to human-readable names (e.g., "Temperature at 2 m").
  - `all`: A comprehensive list of supported parameters.
  - `default`: A subset of commonly used parameters.
- **`param_descriptions`**: detailed descriptions for each parameter, used for documentation or tooltips.
- **`color_map`**: Assigns specific hex colors to weather parameters for consistent visualization across downstream plots.
- **`cache_config`**: Controls local caching behavior (`enabled`, `path`).
- **`logging_config`**: Controls file logging (`filename`, `level`).
- **`api_limits`**: Settings like `max_parameters_per_point_request`.

## 2. model_config.json

**Purpose**: Defines the default search spaces for hyperparameter tuning. It is exposed via `aidweather.config.get_model_config()` for downstream modeling packages.

**Structure**:
- Keys correspond to model class names (e.g., "RandomForest", "ElasticNet").
- Values are dictionaries mapping parameter names (e.g., `model__n_estimators`) to values or distributions.
- **Distributions**:
  - `{"dist": "randint", "low": X, "high": Y}` maps to integer uniform distributions.
  - `{"dist": "uniform", "loc": L, "scale": S}` maps to continuous uniform distributions.
  - `{"dist": "loguniform", "a": A, "b": B}` maps to log-uniform distributions.

**Example**:
```json
"RandomForest": {
    "model__n_estimators": {"dist": "randint", "low": 50, "high": 500}
}
```
