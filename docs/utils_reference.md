# DataFrame Date Utilities Reference — `aidweather.utils`

This document covers the public API and behavioural contracts of the `aidweather.utils` module.

For the top-level Python API surface, see the [API Reference](api_reference.md).
For how `ensure_date_column` is used inside the client response pipeline, see the [Developer & Architecture Guide](developer_guide.md).

---

## 1. Purpose and Scope

`aidweather.utils` provides a single high-level function, `ensure_date_column`, and its configuration companion `DateColumnOptions`. Their job is narrow and deliberate:

> **Given a DataFrame that may store dates in any of several forms — a named column, an alternative-name column, or a `DatetimeIndex` — produce a normalised `datetime64[ns]` column with a canonical name, without modifying the input unless explicitly requested.**

This is a common pre-processing concern when working with tabular output from APIs, CSV files, and database queries, where the date column name and type are not guaranteed to be consistent.

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

`DateColumnOptions` is a **frozen dataclass** used to bundle the keyword arguments of `ensure_date_column` into a single, immutable configuration object. It is also usable directly if you want to parameterise the utility in a pipeline or configuration-driven context.

### Fields

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `inplace` | `bool` | `False` | Mutate *df* directly instead of working on a copy. |
| `candidates` | `Iterable[str] \| None` | `None` | Ordered list of fallback column names to search when the primary *name* is absent. The first matching name wins. |
| `index_fallback` | `bool` | `True` | If `True`, extract dates from the DataFrame's `DatetimeIndex` when no matching column is found. |
| `normalize` | `bool` | `False` | Floor all timestamps to midnight (`00:00:00`) after parsing. Useful when daily granularity is required. |
| `strip_timezone` | `bool` | `True` | Convert tz-aware timestamps to tz-naive UTC-equivalent values. Suppresses downstream `TypeError` when mixing tz-aware and tz-naive data. |

> [!NOTE]
> `DateColumnOptions` is frozen (immutable after construction). It is designed as a value object — create a new instance whenever you need a different configuration.

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

Ensures *df* contains a `datetime64[ns]` column named *name*. Returns a copy of *df* by default.

### 3.1 Column Resolution Order

The function searches for a date source in the following priority order:

1. **Primary name** (`name`): if a column named *name* already exists, it is parsed in place.
2. **Candidates** (`candidates`): if *name* is absent, the candidate list is iterated left-to-right; the first matching column name is used as the source and renamed to *name*.
3. **DatetimeIndex fallback** (`index_fallback=True`): if no column is found and the DataFrame has a `DatetimeIndex`, the index values are extracted into a new column named *name*.

If none of these sources succeeds, a `ValueError` is raised with a diagnostic message listing the names that were tried.

### 3.2 Standardisation Steps

After resolving the date source, two optional standardisation steps are applied (in this order):

1. **Timezone stripping** (`strip_timezone=True`): if the resolved column is tz-aware, `dt.tz_convert(None)` converts it to a tz-naive representation equivalent to UTC. This step is skipped silently if the column is already tz-naive.

2. **Normalisation** (`normalize=True`): `dt.normalize()` floors all timestamps to `00:00:00`, retaining the date component only. Useful when the source data contains sub-daily timestamps but the intended output is daily granularity.

### 3.3 Copy vs. In-Place Behaviour

| `inplace` | Behaviour |
|:---|:---|
| `False` (default) | A copy of *df* is made before modification; the original is unchanged. |
| `True` | *df* is mutated directly; the returned object is the same object as the input. |

> [!CAUTION]
> With `inplace=True`, modifications to the returned DataFrame also affect the original. Use this only when memory usage is a concern and the caller intentionally discards the original.

### 3.4 Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `df` | `pd.DataFrame` | *(required)* | Input DataFrame to process. |
| `name` | `str` | `"date"` | Target column name for the resulting datetime column. |
| `inplace` | `bool` | `False` | Mutate *df* in place; otherwise return a copy. |
| `candidates` | `Iterable[str] \| None` | `None` | Fallback column names to try when *name* is absent. |
| `index_fallback` | `bool` | `True` | Use the `DatetimeIndex` as source when no column is found. |
| `normalize` | `bool` | `False` | Floor timestamps to midnight after parsing. |
| `strip_timezone` | `bool` | `True` | Strip tz info from tz-aware timestamps. |

### 3.5 Returns

The DataFrame (a copy, or the original if `inplace=True`) with a `datetime64[ns]` column named *name*. All other columns are preserved unchanged.

### 3.6 Raises

| Exception | Condition |
|:---|:---|
| `ValueError` | No suitable date source found: *name* is absent, no candidate column matches, and the index is not a `DatetimeIndex`. |

---

## 4. Usage Examples

### Basic: column already named `"date"`

```python
import pandas as pd
from aidweather.utils import ensure_date_column

df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "T2M": [22.1, 21.8]})
df = ensure_date_column(df)

print(df.dtypes)
# date    datetime64[ns]
# T2M     float64
```

### Column named differently from the target

```python
df = pd.DataFrame({"timestamp": ["2023-01-01", "2023-01-02"], "T2M": [22.1, 21.8]})

# Tell ensure_date_column to look for "timestamp" as a fallback:
df = ensure_date_column(df, candidates=["timestamp", "time", "dt"])

# The result has a "date" column; "timestamp" is dropped.
print(df.columns.tolist())  # ["date", "T2M"]
```

### DatetimeIndex fallback (typical for PowerClient output)

```python
# PowerClient returns DataFrames indexed by date:
df.index  # DatetimeIndex(['2023-01-01', '2023-01-02', ...], dtype='datetime64[ns]', name='date')

# ensure_date_column extracts the index into a column:
df = ensure_date_column(df, index_fallback=True)

print(df["date"].dtype)  # datetime64[ns]
```

### Normalise to midnight (daily granularity)

```python
# Source data has sub-daily timestamps:
df = pd.DataFrame({
    "date": ["2023-01-01 06:00:00", "2023-01-02 18:30:00"],
    "T2M": [22.1, 21.8],
})
df = ensure_date_column(df, normalize=True)

print(df["date"])
# 0   2023-01-01
# 1   2023-01-02
```

### Handling tz-aware timestamps

```python
# A DataFrame coming from a tz-aware data source:
df = pd.DataFrame({
    "date": pd.to_datetime(["2023-01-01", "2023-01-02"]).tz_localize("UTC"),
    "T2M": [22.1, 21.8],
})

# Default behaviour strips the timezone:
df = ensure_date_column(df)
print(df["date"].dt.tz)  # None
```

> [!TIP]
> If you need to retain timezone information for downstream processing, set `strip_timezone=False`. Be aware that mixing tz-naive and tz-aware columns will raise `TypeError` in many pandas operations.

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

> [!NOTE]
> `DateColumnOptions` is intentionally not passed directly to `ensure_date_column` — the function takes keyword arguments for ergonomic use at call sites. `DateColumnOptions` is useful when you want to represent a reusable configuration as a value object (e.g., stored in a config dict, passed through a pipeline stage, or used for equality checks in tests).

---

## 5. Interaction with `PowerClient` Output

`PowerClient` returns DataFrames with a `DatetimeIndex` named `"date"`. When you want to work with date as a plain column rather than an index (e.g., before writing to CSV or joining to another table), use `ensure_date_column` with the default settings:

```python
from aidweather import PowerClient
from aidweather.utils import ensure_date_column

client = PowerClient()
df = client.get_point_data(lat=-23.55, lon=-46.63, start="2023-01-01", end="2023-01-31", params=["T2M"])

# df has a DatetimeIndex — convert to a plain column:
df_flat = ensure_date_column(df)

print(df_flat.columns.tolist())  # ["date", "T2M", ...]
```

---

## 6. Design Notes

### Why not just `df.reset_index()`?

`reset_index()` is the standard pandas idiom for converting an index to a column. `ensure_date_column` goes further:

- It handles the case where the date is already a column (not an index), possibly with a different name.
- It parses string columns that were not yet converted to `datetime64`.
- It applies optional tz stripping and midnight normalisation in one call.
- It accepts a fallback candidate list, making it robust to heterogeneous DataFrame sources without requiring `if/elif` chains at call sites.

### Why a frozen dataclass for options?

`DateColumnOptions` being frozen (immutable) means a configured instance can be stored in a module-level constant, passed across function boundaries, and used as a default parameter without the risk of mutable default argument aliasing — a common Python gotcha with mutable defaults.

---

## 7. Public API Symbol Table

| Symbol | Type | Exported from |
|:---|:---|:---|
| `ensure_date_column` | `function` | `aidweather.utils`, `aidweather` |
| `DateColumnOptions` | `dataclass` | `aidweather.utils`, `aidweather` |
