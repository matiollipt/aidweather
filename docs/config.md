# Configuration

`aidweather` loads its settings from a bundled `assets/config.json` file. Cache location, logging path, and rate limits can be overridden via environment variables or programmatically via the `cfg` singleton.

For a complete list of config objects and accessors, see the
[API Inventory](api_inventory.md#aidweatherconfig).

---

## Cache path

The cache database is stored at a platform-appropriate user directory by default:

| Platform | Default path |
|---|---|
| Linux | `~/.cache/aidweather/aidweather_cache.db` |
| macOS | `~/Library/Caches/aidweather/aidweather_cache.db` |
| Windows | `%LOCALAPPDATA%\aidweather\Cache\aidweather_cache.db` |

This is a **shared cache** — all your projects using `aidweather` on the same machine read from and write to the same file. That's intentional: NASA POWER data for a given location and date range is the same regardless of which project asked for it.

### Overriding the cache location

Set an environment variable before running your script or starting your server:

```bash
export AIDWEATHER_CACHE_DIR=/data/shared/aidweather_cache
```

Priority order when resolving the path:
1. `AIDWEATHER_CACHE_DIR` environment variable
2. A `path` key in `config.json` or programmatically set via `cfg.set("cache_config.path", "/custom/path")`
3. Platform default (see table above)


---

## Log file path

By default, log files are stored in a platform-appropriate directory resolved using Python's standard `platformdirs` library:

| Platform | Default path |
|---|---|
| Linux | `~/.local/state/aidweather/log/aidweather.log` |
| macOS | `~/Library/Logs/aidweather/aidweather.log` |
| Windows | `%USERPROFILE%\AppData\Local\aidweather\Logs\aidweather.log` |

### Overriding the log file location

Priority order when resolving the log path:
1. `AIDWEATHER_LOG_DIR` environment variable (if the filename is relative).
2. A `path` key under `logging_config` in `config.json` or programmatically set via `cfg.set("logging_config.path", "/custom/log/dir")` (if the filename is relative).
3. Platform default (see table above, if the filename is relative).

*Note: If `logging_config.filename` is set to an absolute path, it will be used directly as-is, ignoring the above directories.*


---

## Accessing config values in code

The singleton `cfg` object provides dot-notation access to all settings, and allows programmatically overriding values at runtime:

```python
from aidweather import cfg

# Get a nested value
daily_url = cfg.get_url("daily", "point")

# Set a config value programmatically (single source of truth for dynamic overrides)
cfg.set("cache_config.path", "/custom/path/to/cache")

# Get the resolved cache settings
cache = cfg.cache_config()
print(cache["path"])     # resolved cache directory (database file name aidweather_cache.db is inside it)
print(cache["enabled"])  # True by default

# Get parameter metadata
params = cfg.params(group="default")   # {code: name} dict
all_params = cfg.params(group="all")
descriptions = cfg.param_descriptions()

# Get logging config
log = cfg.logging_config()
```

---

## config.json sections

| Section | What it contains |
|---|---|
| `base_urls` | NASA POWER endpoint URLs (daily/hourly, point/regional) |
| `params` | Parameter code → short name mappings, grouped as `all` and `default` |
| `param_descriptions` | Full agronomic descriptions per parameter code |
| `api_limits` | API limits (max parameters, max workers, rate limit calls & period) |
| `cache_config` | Caching settings (`enabled` flag, optional `path`) |
| `logging_config` | File log settings (`filename`, `level`, and optional `path`) |

