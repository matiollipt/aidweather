# utils — DataFrame date utilities

`ensure_date_column` locates, parses, and standardises a datetime column in a pandas DataFrame. It searches by column name, a list of candidates, or the DataFrame's DatetimeIndex, and returns a timezone-naive `datetime64[ns]` column.

For a complete list of utility classes, public helpers, and internal helpers,
see the [API Inventory](api_inventory.md#aidweatherutils).

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
# Result: index values are copied into a new "date" column (the index itself is left in place)
```

---

## Options

| Argument | Default | What it does |
|---|---|---|
| `name` | `"date"` | Target column name in the output |
| `candidates` | `None` | List of alternative column names to search for |
| `index_fallback` | `True` | Use the DatetimeIndex if no column is found |
| `normalize` | `False` | Strip sub-day time components (normalize to midnight) |
| `strip_timezone` | `True` | Remove timezone info |
| `inplace` | `False` | Modify the DataFrame in place instead of returning a copy |

---

## Merging with PowerClient output

`PowerClient` returns DataFrames indexed by date. Use `ensure_date_column` on field data to align the time axis before joining.

```python
client = PowerClient(temporal_api="daily")
weather = client.get_point_data(...).reset_index()

# Standardize your field data
field = ensure_date_column(field_df, candidates=["sample_date", "collection_date"])

merged = weather.merge(field, on="date")
```
