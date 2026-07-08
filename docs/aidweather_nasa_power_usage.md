# NASA POWER — license, usage, and how aidweather handles it

`aidweather` sits on top of [NASA POWER](https://power.larc.nasa.gov/), a public atmospheric data service. This page covers what the package pulls from POWER, the API limits it respects, and the licensing and attribution terms that apply to the data itself.

`aidweather` is an independent client for accessing NASA POWER data. It is not affiliated with, sponsored by, endorsed by, or approved by NASA.

---

## What aidweather uses from NASA POWER

The package is oriented around the **Temporal APIs**:

- **Daily API** — daily averages, min/max, and totals (e.g., temperature, precipitation, radiation)
- **Hourly API** — hourly resolved values

`aidweather` does not currently support NASA POWER's Monthly or Climatology APIs — `PowerClient` only accepts `"daily"` or `"hourly"` as its temporal resolution.

For bulk or catalog-scale workflows, NASA also hosts the full dataset on AWS S3 as Zarr stores (Analysis Ready Data). The standard API is appropriate for farm-scale and site-level queries; AWS access is the right path for global or multi-year bulk extraction.

---

## API limits you need to know

### Parameters per request

| Service | Point | Regional |
|---|---|---:|
| Daily | 20 parameters | 1 parameter |
| Hourly | 15 parameters | — |

### Regional bounding box

The regional endpoint accepts a geographic bounding box (south/north latitude, west/east longitude) and returns data on a 0.5° × 0.5° grid. The box must not exceed **4.5° on either axis**. Larger areas must be tiled into multiple requests.

`PowerClient` validates these limits before sending any request and raises a `ValueError` if you exceed them.

### Concurrency

NASA explicitly discourages submitting more than **5 concurrent requests** from the same IP. The CLI and `get_multi_point_data` both default to `max_workers=5` for this reason, and any higher requested values are automatically capped to enforce this limit.

### Rate limiting

NASA POWER doesn't publish a fixed daily request quota — in practice, exceeding its service limits gets you an HTTP 429 (Too Many Requests) response. `aidweather` guards against this on two fronts: a client-side sliding-window rate limiter (defaulting to 30 requests per minute, configurable in `config.json` or via `cfg.set()`), and automatic retry-with-backoff for any 429s that slip through.

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
- Use **regional requests** for grid-level analysis over an area up to 4.5° × 4.5°. For larger regions, tile the area into smaller bounding boxes.
- Keep parameter lists within the documented cap for the temporal resolution you're using.
- Set `max_workers` to 5 or below when fetching multiple sites in parallel.
- If you're running the same large footprint repeatedly, **check the cache first** with `aidweather cache info`.
- For bulk historical extraction covering many grid cells or many years, consider NASA's **AWS ARD Zarr datastore** instead of the live API.

---

## License and attribution

NASA POWER data is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- **Free to use** — you may download, use, remix, and build upon the data for any purpose.
- **Commercial use is explicitly allowed.**
- **Redistribution** of the data, raw or modified, is permitted.
- **No endorsement** — using the data doesn't imply NASA's endorsement of your project or use case.
- **Provided as-is** — without warranties of any kind; NASA is not liable for errors, omissions, or damages arising from its use.

If you publish work using POWER data (papers, software, reports, dashboards), include this citation:

> *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program."*

When possible, also link to the NASA POWER website: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/).

See the [NASA POWER API documentation](https://power.larc.nasa.gov/docs/services/api/), [NASA POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/), and [NASA Earthdata Data Use and Citation Guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) for current terms and responsible-use expectations.

### Cache security note

`aidweather`'s local SQLite cache (`aidweather_cache.db`) stores gzip-compressed responses but does **not** encrypt them at rest. If you're on a shared system, secure your cache directory the same way you would any other local data store.

| Aspect | Policy |
|---|---|
| Data license | CC BY 4.0 |
| Commercial use | Allowed |
| Attribution | Required |
| `aidweather` software license | Apache-2.0 |

---

## Choosing NASA POWER vs. other sources

NASA POWER is not the only option for historical weather data. For a structured comparison with Meteostat and other tools — covering data origins, spatial coverage, installation, known limitations, and a decision guide — see the [Data Source Comparison](data_source_comparison.md).
