# NASA POWER — usage, limits, and how aidweather handles them

`aidweather` sits on top of [NASA POWER](https://power.larc.nasa.gov/), a public atmospheric data service. Using it responsibly keeps the service available for everyone.

---

## What aidweather uses from NASA POWER

The package is oriented around the **Temporal APIs**:

- **Daily API** — daily averages, min/max, and totals (e.g., temperature, precipitation, radiation)
- **Hourly API** — hourly resolved values
- **Monthly and Climatology APIs** — longer-term summaries (not yet in the default CLI, but accessible through the client)

For bulk or catalog-scale workflows, NASA also hosts the full dataset on AWS S3 as Zarr stores (Analysis Ready Data). The standard API is appropriate for farm-scale and site-level queries; AWS access is the right path for global or multi-year bulk extraction.

---

## API limits you need to know

### Parameters per request

| Service | Point | Regional |
|---|---|---:|
| Daily | 20 parameters | 1 parameter |
| Hourly | 15 parameters | — |
| Monthly / Climatology | 20 parameters | 1 parameter |

`PowerClient` validates these limits before sending any request and raises a `ValueError` if you exceed them.

### Concurrency

NASA explicitly discourages submitting more than **5 concurrent requests** from the same IP. The CLI and `get_multi_point_data` both default to `max_workers=5` for this reason.

### Repeated grid-cell requests

If your workflow repeatedly samples the same 0.5° × 0.5° grid cell, NASA may throttle or block the IP. Use caching — it's on by default.

### Time standard

All POWER temporal APIs default to **Local Solar Time (LST)**, not civil local time. LST is solar-time based and may differ from your local time zone. Keep this in mind when merging POWER data with timestamps from sensors or loggers.

---

## How aidweather keeps requests responsible

**Local SQLite cache** — every response is stored locally and reused on the next request for the same location and parameter set. This is the single most effective way to reduce API load.

**Retry with backoff** — transient failures (HTTP 429, 50x) are retried automatically with exponential backoff rather than hammering the server.

**Explicit User-Agent** — the client identifies itself (`aidweather/version`) in every request, which is good upstream citizenship and helps NASA track usage patterns.

**Parameter-count validation** — requests that would violate API limits are rejected before they hit the network.

**Point and regional endpoints are separate** — different validation rules apply to each, and keeping them separate prevents accidental misuse.

---

## Recommended practices

- Use **point requests** for normal single-farm or single-site workflows.
- Keep parameter lists within the documented cap for the temporal resolution you're using.
- Set `max_workers` to 5 or below when fetching multiple sites in parallel.
- If you're running the same large footprint repeatedly, **check the cache first** with `aidweather cache info`.
- For bulk historical extraction covering many grid cells or many years, consider NASA's **AWS ARD Zarr datastore** instead of the live API.

---

## Attribution

NASA POWER data is licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). If you publish work using POWER data, include this citation:

> *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program."*

See [NASA_POWER_Licence_Usage.md](NASA_POWER_Licence_Usage.md) for the full compliance summary.
