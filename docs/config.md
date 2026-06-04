# Configuration

`aidweather` loads its settings from a bundled `assets/config.json` file on import. You don't need to touch this file for normal use — sensible defaults are in place and the package never fails to import even if the file is missing.

The main thing most users will want to configure is the **cache location**.

For a complete list of config objects and accessors, see the
[API Inventory](api_inventory.md#aidweatherconfig).

---

## Cache path

The cache database lives at a platform-appropriate user directory by default:

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

Or put it in a `.env` file in your project root — `PowerClient` loads `.env` automatically.

Priority order when resolving the path:
1. `AIDWEATHER_CACHE_DIR` environment variable
2. An absolute `path` key in `config.json` (project-level override)
3. Platform default (see table above)

---

## API key

Set your NASA POWER API key as an environment variable:

```bash
NASA_POWER_API_KEY=your_key_here
```

Or in a `.env` file in the project root. Without a key, the client uses IP-based limits (30,000 requests/day shared across all users on your IP).

---

## Accessing config values in code

The singleton `cfg` object provides dot-notation access to all settings:

```python
from aidweather import cfg

# Get a nested value
daily_url = cfg.get_url("daily", "point")

# Get the resolved cache settings
cache = cfg.cache_config()
print(cache["path"])     # resolved cache directory
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
| `api_limits` | Max parameters per request type |
| `cache_config` | `enabled` flag (path is resolved by Python, not set here) |
| `logging_config` | File log settings (`filename`, `level`) |

