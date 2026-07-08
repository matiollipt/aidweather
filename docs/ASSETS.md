# Assets

The `aidweather` package includes one bundled JSON file, `config.json`, in `src/aidweather/assets/`. It's loaded automatically at import time using `importlib.resources`, so the package works correctly regardless of how it's installed (wheel, editable, zipimport).

---

## config.json

The central configuration file. Loaded by `aidweather.config` and accessible via the `cfg` singleton.

**Sections:**

- **`base_urls`** — NASA POWER API endpoint URLs for daily and hourly, point and regional requests.
- **`params`** — Parameter code → short name mappings. Two groups: `"all"` (full catalogue) and `"default"` (a practical subset for common agricultural workflows).
- **`param_descriptions`** — Full agronomic descriptions for each parameter, including data source, resolution, and agricultural relevance.
- **`api_limits`** — Max parameters per request type (used for validation before requests are sent).
- **`cache_config`** — `enabled` flag and an optional `path` override (defaults to `null`, in which case the path is resolved via the `AIDWEATHER_CACHE_DIR` env var or `platformdirs`).
- **`logging_config`** — File logging settings (`filename`, `level`, and optional `path`).
