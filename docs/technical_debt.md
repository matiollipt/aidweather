# Technical Debt & Release Limitations — `aidweather`

This document tracks known limitations, pending API additions, and future architectural enhancements for `aidweather`.

---

## 1. Remaining Limitations & Non-Breaking Boundaries

1. **Antimeridian Bounding Boxes**: NASA POWER's regional bounding box endpoint currently rejects boxes crossing 180° longitude. Multi-cell polygon requests crossing the antimeridian must be issued as multiple point requests or split regional boxes.
2. **Local Solar Time (LST) API Parameter**: NASA POWER's hourly endpoint defaults to LST timestamps when the `time-standard` query parameter is omitted (`aidweather` never sets it). Requesting UTC timestamps instead requires an explicit `time-standard=UTC` override, which is not currently exposed by `PowerClient` or the CLI.
3. **IMERG Precipitation Products**: High-resolution (~0.1° × 0.1°) IMERG parameters are available under specific NASA POWER communities. Full native grid metadata support for IMERG parameters will be expanded in a post-beta release.
4. **Cache coverage is inferred from data, not tracked separately ("look and understand" strategy)**: `_get_date_ranges_to_fetch` decides whether a cached entry already covers a requested date range by looking at the min/max timestamp actually present in the cached data — it does not separately record what date range was requested and confirmed. This relies on NASA POWER always returning a key for every day/hour in a response (using `-999` for missing values, never omitting a key). If that invariant is ever violated — most plausibly for very recent/provisional dates where NASA's response could in principle be genuinely truncated rather than fill-coded — the trailing dates will look "uncovered" and the cache will re-fetch that range from the API on every call, instead of caching it once. This is a deliberate simplicity trade-off: it means no request is ever permanently frozen at a stale/incomplete value (it keeps trying until NASA backfills it), at the cost of not giving a hard "fetched once, never called again" guarantee for that trailing edge. No code change is planned for this; it is called out here and in the `_get_date_ranges_to_fetch` docstring so it isn't mistaken for a bug during debugging.

---

## 2. Maintenance Procedures

- **Parameter Metadata Updates**: Synchronize `src/aidweather/assets/config.json` whenever NASA POWER releases updated parameter dictionaries or reanalysis source upgrades (e.g. MERRA-2 to GEOS-IT transition).
- **Cache Migration Strategy**: Cache keys are prefixed with `v1_`. If future cache serialization schemas change, increment the key version tag (`v2_`) to seamlessly bypass older local cached BLOB records without breaking client execution.
