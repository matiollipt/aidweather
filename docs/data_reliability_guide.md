# Data reliability for agricultural use cases

This document covers reliability trade-offs between meteorological data sources used in agricultural analytics — specifically **NASA POWER** (used by `aidweather`) versus station-based and interpolated sources such as **Meteostat** and **Open-Meteo**.

## Interpolated station data: known limitations

Tools like Meteostat rely on observations from local weather stations combined with spatial interpolation. Ground-truth stations provide high-fidelity readings for their immediate location, but using interpolated records for long-term agricultural modeling introduces specific risks:

- **Equipment changes and failures:** sensors break, calibration drifts, and equipment is replaced. A localized anomaly in the time series may reflect a faulty instrument rather than an actual climate event.
- **Station relocation:** over a 20-year span, stations are frequently moved. A shift of a few kilometers, or a change in elevation, can break time-series continuity and silently shift the temperature or precipitation baseline.
- **Spatial interpolation error:** when a station goes offline, these APIs interpolate from the nearest available alternatives. In large agricultural regions — such as the Brazilian Cerrado — the next active station may be hundreds of kilometers away, in a different microclimate.

## NASA POWER: reanalysis characteristics

`aidweather` wraps the NASA POWER dataset, which is built on satellite observations assimilated with the MERRA-2 global climate model.

- **Temporal consistency:** the time series is derived from a stable planetary-scale model, not individual sensors. There are no station relocations, equipment changes, or calibration breaks.
- **Spatial coverage:** NASA POWER provides a uniform 0.5° × 0.5° global grid. Data quality is consistent even in areas where physical weather stations are sparse or absent.
- **Missing data handling:** POWER produces values for every grid cell at every time step. The `-999` fill values that occasionally appear in API responses are coerced to `NaN` by `aidweather`, so downstream pipelines do not silently ingest invalid sentinels.

## Trade-offs

Reanalysis data is modelled, not observed. It is smoothed over ~55 km and does not capture local terrain effects, urban heat islands, or convective precipitation peaks that a nearby rain gauge would record. For sites within 30–50 km of a well-maintained, continuously operating station, that station's records may be more accurate for the specific location.

For long-running pipelines where temporal consistency over decades matters more than spatial precision at the sub-grid scale, reanalysis sources like NASA POWER reduce the risk of silent discontinuities introduced by station evolution.

See the [Data Source Comparison](data_source_comparison.md) for a structured side-by-side and decision guide.
