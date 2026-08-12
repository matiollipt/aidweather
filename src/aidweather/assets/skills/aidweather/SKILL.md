---
name: aidweather
description: Use when an agent needs NASA POWER weather or solar data through aidweather, including CLI bootstrap, messy coordinate/date inputs, agricultural weather parameters, multi-location fetches, and JSON output for LLM-readable pipelines.
---

# aidweather

Use `aidweather` to fetch NASA POWER weather data for one or more known
latitude/longitude locations. For agent work, use the CLI and save JSON.

## Bootstrap

1. Check that the CLI is available:

   ```bash
   command -v aidweather
   ```

2. If available, verify it and continue with the Fetch Workflow below:

   ```bash
   aidweather --version
   ```

3. If missing, ask the user before installing anything. If they approve and
   network access is available, install the package with:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- -y
   ```

4. If installation is not approved or network access fails, return the exact
   command above and ask the user to run it locally.

## Fetch Workflow

1. Extract the user's date range, temporal resolution, parameter list, and
   latitude/longitude coordinates.
2. Do not infer coordinates from place names. If the user gives a name only,
   ask for latitude and longitude.
3. Use default parameters only when the user does not specify variables:
   `T2M,PRECTOTCORR`.
4. Keep `--workers` at or below `5` unless the user explicitly accepts higher
   concurrency (`fetch-multi` / `fetch-transect` only).
5. Save to a `.json` file and read it back rather than parsing the console
   preview table. Pass `--no-preview` to keep console output clean.
6. Attribute data to NASA POWER when summarizing results.

There is no dedicated "agent" subcommand — use the same commands a human
would, with `--output <path>.json --no-preview`.

### Single point

```bash
aidweather fetch \
  --lat -23.55 --lon -46.63 \
  --start 2023-01-01 --end 2023-01-31 \
  --params T2M,PRECTOTCORR \
  --output result.json --no-preview
```

### Multiple points

`fetch-multi` reads a CSV with `lat`/`lon` columns (optional `name`,
`elevation`, `wind_elevation`, `wind_surface`). There is no inline
JSON/list flag — build the CSV first if the user gives you a list of
points.

```bash
aidweather fetch-multi \
  --points-file points.csv \
  --start 2023-01-01 --end 2023-01-07 \
  --params T2M,PRECTOTCORR \
  --workers 5 \
  --output result.json --no-preview
```

### Transect (line between two points)

```bash
aidweather fetch-transect \
  --lat-start -23.55 --lon-start -46.63 \
  --lat-end -22.90 --lon-end -43.17 \
  --start 2023-01-01 --end 2023-01-07 \
  --params T2M \
  --num-points 10 \
  --output result.json --no-preview
```

### Regional bounding box

One parameter only, daily resolution only, box capped at 4.5° x 4.5°.

```bash
aidweather fetch-regional \
  --lat-min -24.0 --lat-max -23.0 \
  --lon-min -47.0 --lon-max -46.0 \
  --start 2023-01-01 --end 2023-01-07 \
  --params T2M \
  --output result.json --no-preview
```

## JSON Output Shape

`--output *.json` writes `pandas.DataFrame.to_json(orient="records",
date_format="iso")` — a flat JSON array of row records, not a wrapper object
with metadata. Each record has a `date` field plus one column per requested
parameter (and `lat`/`lon` for `fetch`/`fetch-multi`/`fetch-transect`
outputs). There is no separate `metadata`/`locations`/`failed_locations`
envelope — read row values directly.

For concrete examples, read `references/request-patterns.md`.

## Messy Requests

- Decimal degrees, DMS, and DDM coordinates are acceptable when lat/lon are
  present (`GeoCoordinate` in the Python API parses all three).
- If plain-language parameters are obvious, map them to POWER codes
  (`temperature` -> `T2M`, `rain` -> `PRECTOTCORR`). Run
  `aidweather params list --group all` if unsure which code applies.
- Use Python only if the user explicitly asks for Python. There is no
  `get_weather_json`/`get_weather_json_text` helper — use `PowerClient`
  directly, e.g.:

  ```python
  from aidweather import PowerClient

  client = PowerClient(temporal_api="daily")
  df = client.get_point_data(
      lat=-23.55, lon=-46.63, start="2023-01-01", end="2023-01-31",
      params=["T2M", "PRECTOTCORR"],
  )
  ```
