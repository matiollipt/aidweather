# Developer & Architecture Guide — `aidweather`

This document details the internal design patterns, request lifecycles, and cache identity mechanisms for `aidweather` contributors.

---

## 1. Request Lifecycle Architecture

```text
User API Request / CLI Invocation
  │
  ├──► Input Normalization (normalize_coord_input -> GeoCoordinate)
  ├──► Parameter & Date Validation (_validate_inputs)
  ├──► Payload Construction (_build_point_payload / _build_regional_payload)
  ├──► Cache Key Generation (_make_cache_key -> v1_<sha256>)
  │      ├── Cache Hit Check (Read SQLite)
  │      └── Gap Resolution (_get_date_ranges_to_fetch)
  ├──► Network Execution (_fetch_and_parse_ranges via requests Session with retries)
  ├──► Response Serialization & NaN Normalization (-999 -> pd.NA)
  ├──► Merge & Cache Update (_merge_and_deduplicate -> SQLite Write)
  └──► Return DataFrame + Attach Spatial Provenance (df.attrs["spatial_provenance"])
```

---

## 2. Cache Key Identity & Schema Versioning

Cache keys in `_make_cache_key` use SHA-256 digests prefixed with a schema version tag (`v1_`):

```python
def _make_cache_key(payload: dict[str, Any], temporal_api: str = "daily") -> str:
    key_payload = payload.copy()
    key_payload.pop("start", None)
    key_payload.pop("end", None)
    key_payload["_temporal_api"] = temporal_api
    encoded = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    return "v1_" + hashlib.sha256(encoded).hexdigest()
```

By excluding `start` and `end` dates from `key_payload`, `aidweather` tracks cached date ranges per spatial payload and fetches only missing date segments.

---

## 3. Data Invariants

Contributors must maintain the following non-negotiable data invariants:
1. **Never Impute Missing Data**: Missing values from NASA POWER (fill code `-999`) must be converted strictly to explicit pandas missing values (`pd.NA` / `np.nan`). Never apply forward-filling (`ffill`), backward-filling (`bfill`), or linear interpolation.
2. **Preserve Original Units**: Do not apply silent unit conversions. Units returned by NASA POWER must be preserved and disclosed in parameter metadata.
3. **Preserve Coordinate Precision**: Coordinates must retain at least 4 decimal places precision.
4. **Preserve DataFrame Attributes**: Slicing or column selection helpers (`_filter_df_by_date`, `_ensure_all_params_in_df`) must preserve `df.attrs["spatial_provenance"]`.
