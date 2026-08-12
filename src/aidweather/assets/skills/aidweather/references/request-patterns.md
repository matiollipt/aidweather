# aidweather Request Patterns

Use these examples when translating messy user requests into deterministic
`aidweather` CLI calls. All examples use `--output *.json --no-preview` so
the result can be read back programmatically instead of parsed from console
tables.

## Single Decimal Point

User: "Get daily temperature and precipitation for -23.55, -46.63 in January 2023."

```bash
aidweather fetch \
  --lat -23.55 --lon -46.63 \
  --start 2023-01-01 --end 2023-01-31 \
  --params T2M,PRECTOTCORR \
  --output result.json --no-preview
```

## DMS Or DDM Coordinates

`--lat`/`--lon` on `fetch` take decimal degrees only. If the user supplies
DMS/DDM text (e.g. `23°33'0" S`), convert it to decimal degrees first with
`GeoCoordinate` in Python, then pass the resulting floats to the CLI:

```python
from aidweather import GeoCoordinate

coord = GeoCoordinate.from_strings('23°33\'0" S', '46°37\'48" W')
print(coord.lat, coord.lon)
```

```bash
aidweather fetch \
  --lat -23.55 --lon -46.63 \
  --start 2023-01-01 --end 2023-01-31 \
  --params T2M \
  --output result.json --no-preview
```

## Multiple Locations

There is no inline JSON/list flag for multiple points — write a CSV first,
then call `fetch-multi`:

`points.csv`:

```csv
lat,lon,name
-23.55,-46.63,sao-paulo
-22.90,-43.17,rio
```

```bash
aidweather fetch-multi \
  --points-file points.csv \
  --start 2023-01-01 --end 2023-01-07 \
  --params T2M,PRECTOTCORR \
  --workers 5 \
  --output result.json --no-preview
```

## CSV Points File With Elevation

`points.csv` may also carry `elevation`, `wind_elevation`, `wind_surface`
columns — they are forwarded to the API when present:

```csv
lat,lon,name,elevation
-23.55,-46.63,sao-paulo,760
-22.90,-43.17,rio,5
```

```bash
aidweather fetch-multi \
  --points-file points.csv \
  --start 2023-01-01 --end 2023-01-07 \
  --params T2M,PRECTOTCORR \
  --output weather.json --no-preview
```

## Invalid Place-Name-Only Request

User: "Get weather for São Paulo last January."

Do not guess coordinates. Ask:

```text
Please provide latitude and longitude for São Paulo, or a CSV with lat/lon columns.
```

## JSON Shape

`--output result.json` writes `DataFrame.to_json(orient="records",
date_format="iso")` — a flat array of row records, no metadata wrapper:

```json
[
  {"date": "2023-01-01T00:00:00.000Z", "T2M": 22.1, "PRECTOTCORR": 3.4, "lat": -23.55, "lon": -46.63},
  {"date": "2023-01-02T00:00:00.000Z", "T2M": 23.0, "PRECTOTCORR": 0.0, "lat": -23.55, "lon": -46.63}
]
```

`fetch-regional` output has a different, wider shape (one row per grid
cell/date with parameter columns) since the regional endpoint returns a
grid rather than a point series — inspect the first record before assuming
column names.
