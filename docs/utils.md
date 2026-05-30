# utils — DataFrame date utilities

The `utils` module exists because real-world data is messy. Farm records, sensor exports, and spreadsheets come in with date columns named anything from `"date"` to `"obs_date"` to `"timestamp"`, and sometimes the date isn't a column at all — it's the DataFrame index.

`ensure_date_column` handles all of that in one call and gives back a DataFrame with a clean, timezone-naive `datetime64[ns]` column you can rely on.

---

## ensure_date_column

```python
from aidweather import ensure_date_column
import pandas as pd

# Standard case — column is named "date"
df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"], "T2M": [22.1, 28.4]})
df = ensure_date_column(df)
print(df["date"].dtype)  # datetime64[ns]

# Column has a different name
df = pd.DataFrame({"obs_date": ["2023-01-01", "2023-06-15"], "T2M": [22.1, 28.4]})
df = ensure_date_column(df, name="date", candidates=["obs_date"])
# Result: "obs_date" is renamed to "date" and parsed as datetime

# Date is in the index (common with PowerClient output)
df = pd.DataFrame({"T2M": [22.1]}, index=pd.to_datetime(["2023-01-01"]))
df = ensure_date_column(df, name="date")
# Result: index is moved into a "date" column
```

---

## Options

| Argument | Default | What it does |
|---|---|---|
| `name` | `"date"` | Target column name in the output |
| `candidates` | `None` | List of alternative column names to search for |
| `index_fallback` | `True` | Use the DatetimeIndex if no column is found |
| `normalize` | `True` | Strip sub-day time components (normalize to midnight) |
| `strip_timezone` | `True` | Remove timezone info |
| `inplace` | `False` | Modify the DataFrame in place instead of returning a copy |

---

## When to use this

`ensure_date_column` is the recommended way to standardize the time axis before merging `PowerClient` output with your own data. `PowerClient` returns DataFrames indexed by date; calling `ensure_date_column` on your field data makes both sides ready to join.

```python
client = PowerClient(temporal_api="daily")
weather = client.get_point_data(...).reset_index()

# Standardize your field data
field = ensure_date_column(field_df, candidates=["sample_date", "collection_date"])

merged = weather.merge(field, on="date")
```
