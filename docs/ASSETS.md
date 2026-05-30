# Assets

The `aidweather` package includes two bundled JSON files in `src/aidweather/assets/`. They are loaded automatically at import time using `importlib.resources`, so the package works correctly regardless of how it's installed (wheel, editable, zipimport).

---

## config.json

The central configuration file. Loaded by `aidweather.config` and accessible via the `cfg` singleton.

**Sections:**

- **`base_urls`** — NASA POWER API endpoint URLs for daily and hourly, point and regional requests.
- **`params`** — Parameter code → short name mappings. Two groups: `"all"` (full catalogue) and `"default"` (a practical subset for common agricultural workflows).
- **`param_descriptions`** — Full agronomic descriptions for each parameter, including data source, resolution, and agricultural relevance.
- **`api_limits`** — Max parameters per request type (used for validation before requests are sent).
- **`cache_config`** — `enabled` flag. The cache path itself is resolved by Python using `platformdirs` and the `AIDWEATHER_CACHE_DIR` env var — it is not stored here.
- **`logging_config`** — File logging settings (`filename`, `level`).
- **`color_map`** — Hex color per parameter for consistent visualization across downstream packages (`aidviz`).

---

## model_config.json

Defines default hyperparameter search spaces for ML models. This file is not used by `aidweather` directly — it is exposed via `aidweather.config.get_model_config()` for downstream modeling packages such as `aidfarm`.

**Structure:** keys are model class names (e.g., `"RandomForest"`), values are parameter grids using distribution descriptors:

```json
{
  "RandomForest": {
    "model__n_estimators": {"dist": "randint", "low": 50, "high": 500}
  }
}
```

Supported distributions: `randint`, `uniform`, `loguniform`.
