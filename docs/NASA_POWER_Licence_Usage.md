# NASA POWER License and Data Usage Guidelines

## Overview

AidWeather integrates data from the **NASA POWER** (Prediction Of Worldwide Energy Resources) service. This document outlines the legal requirements, usage rights, and attribution mandates associated with using NASA POWER data within the AidWeather ecosystem.

## 1. License Terms (CC BY 4.0)

The data provided by NASA POWER is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

### Usage Rights
- **Free to Use**: You may download, use, remix, and build upon the data for any purpose.
- **Commercial Use**: Commercial applications of the data are explicitly allowed.
- **Redistribution**: You are permitted to redistribute the data in its raw or modified form.

### Restrictions and Caveats
- **No Endorsement**: Usage of the data does not imply NASA's endorsement of your project, product, or specific use case.
- **"As-Is" Provision**: The data is provided without warranties of any kind. NASA is not liable for any errors, omissions, or damages resulting from its use.

## 2. Mandatory Citation Requirements

Users of AidWeather who utilize NASA POWER data in derived works (publications, software, reports) **must** provide appropriate credit. 

### Official Citation Format
NASA requests the following citation for any use of POWER data:
> *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science Directorate Applied Science Program."*

### Digital Attribution
When possible, provide a link to the NASA POWER website: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/).

## 3. Data Usage Guidelines for End-Users

### API Service Limits
AidWeather is designed to respect NASA's service limits. Users should be aware of the following tiers:
- **Unauthenticated (IP-based)**: Generally limited to 30,000 requests per day.
- **Personal API Key**: Recommended for production use to ensure higher throughput and reliability.

### Security and Caching
- **Local Caching**: AidWeather caches data locally in a SQLite database (`aidweather_cache.db`). Note that while data is compressed, it is **not encrypted at rest**. Users operating on shared systems should secure their cache directories.

## 4. Legal Compliance Summary

| Aspect | Policy |
| :--- | :--- |
| **Data License** | CC BY 4.0 |
| **Commercial Use** | Allowed |
| **Attribution** | Mandatory |
| **Software License** | Apache-2.0 (AidWeather) |

---
*This document is a self-contained guide for NASA POWER data usage compliance within AidWeather.*
