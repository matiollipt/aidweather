# Data reliability guide for agricultural use cases

This document compares the reliability characteristics of meteorological data sources used in agricultural analytics — specifically, the **NASA POWER** dataset (which powers `aidweather`) against station-based, interpolated sources like **Meteostat** and **Open-Meteo**.

## The challenge: interpolation vs. stability

Agricultural data modeling requires decades of stable, consistent daily weather data to predict crop yields, evaluate drought risk, and train machine learning models. The choice of underlying meteorological data source can drastically impact the reliability of these models.

### The pitfalls of interpolated local stations

Tools like Meteostat or some Open-Meteo variants rely on local weather station observations (e.g., INMET in Brazil) combined with interpolation algorithms to estimate conditions at specific coordinates. Ground-truth stations provide high-fidelity readings, but relying purely on interpolated stations for long-term agricultural modeling introduces real risks:

- **Equipment changes & failures:** sensors break, calibration drifts, and equipment gets replaced. A localized anomaly might reflect a faulty thermometer or rain gauge rather than an actual climate shift.
- **Station relocation:** over a 20-year span, local stations are frequently moved. A shift of just a few kilometers, or a change in elevation, can break the continuity of the time series and quietly change the temperature or precipitation baseline.
- **Spatial interpolation errors:** when a local station goes offline, these APIs interpolate from the next nearest stations. In large agricultural regions (e.g., the Brazilian Cerrado or Amazon), the next available station might be hundreds of kilometers away, in a completely different microclimate.

### The NASA POWER advantage: stable grid data

`aidweather` is a client for the NASA POWER (Prediction of Worldwide Energy Resources) dataset, which is built on a stable, long-term grid of satellite observations and assimilation models (like MERRA-2).

- **Consistency over time:** because the data comes from planetary-scale satellite assimilation rather than individual, fragile ground sensors, the time series stays stable over decades. There's no such thing as a "station relocation" in a satellite-derived grid.
- **Uniform spatial coverage:** NASA POWER provides a continuous 0.5° × 0.5° global grid. Even in remote agricultural hubs where physical weather stations are sparse or nonexistent, data quality remains reliable without relying on extreme spatial interpolation.
- **Resilience to missing data:** NASA handles missing values at the model assimilation level. `aidweather` further standardizes this by coercing the few remaining `-999` fill values into clean numeric `NaN`s, so downstream pandas/numpy pipelines never silently ingest invalid sentinel values as real weather readings.

## Conclusion

For precision agriculture, consistency over decades is often more valuable than the marginal accuracy gain of station observations that only last a few years before a relocation or outage breaks the series. By leveraging NASA POWER's stable grid, `aidweather` avoids the hidden uncertainties of station evolution, equipment failures, and crude spatial interpolation — making it a solid foundational layer for long-term agricultural modeling.
