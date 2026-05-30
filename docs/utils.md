# utils

> [!NOTE]
> **AidWeather Project Context**
> **Mission**: Weather data retrieval and validation for agricultural applications.
> **Key Features**: Modular architecture, production-ready caching, and end-to-end NASA POWER integration.
> **NASA POWER Compliance**: See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for data usage rights.

---

## `aidweather.utils` — DataFrame Utilities

### Purpose

Provides a single, focused primitive used by the client and downstream packages: robust date column standardisation. This module has no dependency on `numpy` beyond what `pandas` already carries.

### Public API

- `ensure_date_column(df, name="date", **kwargs) -> pd.DataFrame`:
  Guarantees a `datetime64[ns]` column exists with the specified `name`.
  Search order: column named `name` → `candidates` list → DataFrame `DatetimeIndex` (if `index_fallback=True`).

- `DateColumnOptions`: Dataclass holding the keyword configuration for `ensure_date_column`.

### Keyword arguments for `ensure_date_column`

| kwarg | type | default | description |
|---|---|---|---|
| `inplace` | `bool` | `False` | Modify the DataFrame in place |
| `candidates` | `list[str] \| None` | `None` | Alternative column names to search |
| `index_fallback` | `bool` | `True` | Fall back to DatetimeIndex |
| `normalize` | `bool` | `True` | Normalise to midnight |
| `strip_timezone` | `bool` | `True` | Remove timezone information |

### Data flow and dependencies

- **Internal imports**: stdlib only (`dataclasses`, `collections.abc`).
- **External dependencies**: `pandas`.

### Minimal usage example

```python
from aidweather.utils import ensure_date_column
import pandas as pd

df = pd.DataFrame({"obs_date": ["2023-01-01", "2023-06-15"]})
df = ensure_date_column(df, name="date", candidates=["obs_date"])
# df now has a datetime64[ns] column named "date"
```
