# Technical Debt & Release Limitations — `aidweather`

This document tracks known limitations, pending API additions, and future architectural enhancements for `aidweather`.

---

## 1. Remaining Limitations & Non-Breaking Boundaries

1. **Antimeridian Bounding Boxes**: NASA POWER's regional bounding box endpoint currently rejects boxes crossing 180° longitude. Multi-cell polygon requests crossing the antimeridian must be issued as multiple point requests or split regional boxes.
2. **Local Solar Time (LST) API Parameter**: While LST availability is noted in parameter metadata, default API calls query UTC timestamps unless explicitly overridden in specialized payload dictionaries.
3. **IMERG Precipitation Products**: High-resolution (~0.1° × 0.1°) IMERG parameters are available under specific NASA POWER communities. Full native grid metadata support for IMERG parameters will be expanded in a post-beta release.

---

## 2. Maintenance Procedures

- **Parameter Metadata Updates**: Synchronize `src/aidweather/assets/config.json` whenever NASA POWER releases updated parameter dictionaries or reanalysis source upgrades (e.g. MERRA-2 to GEOS-IT transition).
- **Cache Migration Strategy**: Cache keys are prefixed with `v1_`. If future cache serialization schemas change, increment the key version tag (`v2_`) to seamlessly bypass older local cached BLOB records without breaking client execution.
