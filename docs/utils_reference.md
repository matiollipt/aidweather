# DataFrame Date Utilities Reference — `aidweather.utils`

This document details the public API and behavioral contracts of the `aidweather.utils` module.

For top-level Python API usage, see the [API Reference](api_reference.md).  
For details on how `ensure_date_column` operates within the client data processing pipeline, see the [Developer & Architecture Guide](developer_guide.md).

---

## 1. Purpose and Scope

`aidweather.utils` provides a focused utility function, `ensure_date_column`, alongside its configuration object `DateColumnOptions`. Their objective is clear and specific:

> **Given a pandas DataFrame that may store dates in various forms—such as a named column, an alternative column name, or a `DatetimeIndex`—produce a standardized `datetime64[ns]` date column under a canonical name, without modifying the original DataFrame unless explicitly requested.**

This simplifies tabular preprocessing when working with data from APIs, CSV files, and database queries where date column names and index structures vary.

---

## 2. `DateColumnOptions` — Configuration Dataclass

```python
@dataclass(frozen=True)
class DateColumnOptions:
    inplace:        bool                    = False
    candidates:     Iterable[str] | None    = None
    index_fallback: bool                    = True
    normalize:      bool                    = False
    strip_timezone: bool                    = True
```

`DateColumnOptions` is an **immutable frozen dataclass** that encapsulates configuration parameters for `ensure_date_column`. It allows you to define reusable date standardization settings across data pipelines.

### Fields

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inplace` | `bool` | `False` | Mutates *df* directly instead of creating a copy. |
| `candidates` | `Iterable[str] \| None` | `None` | Ordered fallback column names to check when primary *name* is missing. First match is used. |
| `index_fallback` | `bool` | `True` | Extracts dates from DataFrame `DatetimeIndex` if no matching column exists. |
| `normalize` | `bool` | `False` | Floors all timestamps to midnight (`00:00:00`) after parsing. Useful for daily aggregation. |
| `strip_timezone` | `bool` | `True` | Converts timezone-aware timestamps to UTC-equivalent timezone-naive values, preventing downstream `TypeError` issues. |

> [!NOTE]
> `DateColumnOptions` is immutable. To use a different configuration, create a new instance with updated field values.

---

## 3. `ensure_date_column` — Primary Function

```python
def ensure_date_column(
    df: pd.DataFrame,
    name: str = "date",
    *,
    inplace: bool = False,
    candidates: Iterable[str] | None = None,
    index_fallback: bool = True,
    normalize: bool = False,
    strip_timezone: bool = True,
) -> pd.DataFrame:
```

Ensures that *df* contains a valid `datetime64[ns]` column named *name*. Returns a modified copy by default.

### 3.1 Column Resolution Order

The function resolves date sources in strict priority order:

1. **Primary Column Name** (`name`): If a column named *name* already exists, it is parsed directly.
2. **Candidate Search** (`candidates`): If *name* is missing, candidate names are evaluated left-to-right; the first matching column is selected and renamed to *name*.
3. **DatetimeIndex Fallback** (`index_fallback=True`): If no candidate column matches and the DataFrame has a `DatetimeIndex`, index values are extracted into a new column named *name*.

If no valid date source is found, a `ValueError` is raised detailing all evaluated column names.

### 3.2 Standardization Steps

Once a date source is identified, two optional standardization transformations are applied:

1. **Timezone Stripping** (`strip_timezone=True`): Timezone-aware timestamps are converted to UTC-equivalent timezone-naive values (`dt.tz_convert(None)`). Timezone-naive inputs are left unchanged.
2. **Midnight Normalization** (`normalize=True`): Timestamps are floored to midnight (`00:00:00`) via `dt.normalize()`, retaining daily date components.

### 3.3 Copy vs. In-Place Behavior

| `inplace` | Behavior |
| :--- | :--- |
| `False` (default) | Creates and returns a copy of *df*, preserving the original DataFrame. |
| `True` | Mutates *df* directly and returns the modified instance. |

> [!CAUTION]
> Setting `inplace=True` modifies the input DataFrame in place. Use this only when optimizing memory usage in large data pipelines.

### 3.4 Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `df` | `pd.DataFrame` | *(required)* | Target DataFrame to process. |
| `name` | `str` | `"date"` | Desired canonical column name for parsed dates. |
| `inplace` | `bool` | `False` | When True, mutates *df* directly. |
| `candidates` | `Iterable[str] \| None` | `None` | Fallback column names to search when *name* is absent. |
| `index_fallback` | `bool` | `True` | Uses `DatetimeIndex` as date source when no column matches. |
| `normalize` | `bool` | `False` | Floors timestamps to midnight (`00:00:00`). |
| `strip_timezone` | `bool` | `True` | Removes timezone awareness from timestamps. |

### 3.5 Return Value

Returns the DataFrame (a copy by default, or the mutated original if `inplace=True`) containing a `datetime64[ns]` column under *name*. All non-date columns remain untouched.

### 3.6 Exceptions

| Exception | Condition |
| :--- | :--- |
| `ValueError` | Raised when no valid date source exists (*name* missing, no matching candidate columns, and index is not a `DatetimeIndex`). |

---

## 4. Usage Examples

### Basic: DataFrame with existing `"date"` column

```python
import pandas as pd
from aidweather.utils import ensure_date_column

df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "T2M": [22.1, 21.8]})
df = ensure_date_column(df)

print(df.dtypes)
# date    datetime64[ns]
# T2M     float64
```

### Alternative date column names

```python
df = pd.DataFrame({"timestamp": ["2023-01-01", "2023-01-02"], "T2M": [22.1, 21.8]})

# Search for "timestamp" as a fallback candidate:
df = ensure_date_column(df, candidates=["timestamp", "time", "dt"])

print(df.columns.tolist())  # ["date", "T2M"]
```

### DatetimeIndex fallback (`PowerClient` outputs)

```python
# PowerClient returns DataFrames with a DatetimeIndex named 'date':
df.index  # DatetimeIndex(['2023-01-01', '2023-01-02'], dtype='datetime64[ns]', name='date')

# Extract index into an explicit column:
df = ensure_date_column(df, index_fallback=True)

print(df["date"].dtype)  # datetime64[ns]
```

### Flooring sub-daily timestamps to midnight

```python
df = pd.DataFrame({
    "date": ["2023-01-01 06:00:00", "2023-01-02 18:30:00"],
    "T2M": [22.1, 21.8],
})
df = ensure_date_column(df, normalize=True)

print(df["date"])
# 0   2023-01-01
# 1   2023-01-02
```

### Handling timezone-aware timestamps

```python
df = pd.DataFrame({
    "date": pd.to_datetime(["2023-01-01", "2023-01-02"]).tz_localize("UTC"),
    "T2M": [22.1, 21.8],
})

# Default settings strip timezone info to produce naive UTC timestamps:
df = ensure_date_column(df)
print(df["date"].dt.tz)  # None
```

> [!TIP]
> To preserve timezone metadata for specialized downstream requirements, set `strip_timezone=False`. Note that mixing timezone-aware and timezone-naive series in pandas can trigger `TypeError` exceptions during joins.

### Using `DateColumnOptions` directly

```python
from aidweather.utils import ensure_date_column, DateColumnOptions

opts = DateColumnOptions(
    candidates=["timestamp", "time"],
    index_fallback=True,
    normalize=True,
    strip_timezone=True,
)

df = ensure_date_column(df, **{f.name: getattr(opts, f.name) for f in opts.__dataclass_fields__.values()})
```

---

## 5. Integration with `PowerClient` Results

`PowerClient` returns DataFrames indexed by a `DatetimeIndex` named `"date"`. To convert the index into a standard column (e.g. before saving to CSV or merging with tabular datasets), run `ensure_date_column`:

```python
from aidweather import PowerClient
from aidweather.utils import ensure_date_column

client = PowerClient()
df = client.get_point_data(lat=-23.55, lon=-46.63, start="2023-01-01", end="2023-01-31", params=["T2M"])

# Convert DatetimeIndex to explicit column:
df_flat = ensure_date_column(df)

print(df_flat.columns.tolist())  # ["date", "T2M", ...]
```

---

## 6. Design Rationale

### Why not rely solely on `df.reset_index()`?

Standard `df.reset_index()` converts indexes to columns, but `ensure_date_column` handles complex real-world edge cases in one step:

- Processes DataFrames where dates are already stored in columns (not indexes) under alternative names.
- Converts string date series to `datetime64[ns]` automatically.
- Strips timezone awareness and normalizes sub-daily timestamps in a single call.
- Evaluates candidate column lists cleanly without complex `if/else` checks in user code.

### Why use a frozen dataclass for configuration?

Making `DateColumnOptions` frozen (immutable) allows configuration instances to be defined as module constants, safely passed between functions, and reused without risk of unintended side effects or mutable argument bugs.

---

## 7. Public API Symbol Table

| Symbol | Type | Exported Modules |
| :--- | :--- | :--- |
| `ensure_date_column` | `function` | `aidweather.utils`, `aidweather` |
| `DateColumnOptions` | `dataclass` | `aidweather.utils`, `aidweather` |
