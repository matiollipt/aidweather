# NASA POWER API usage, limits, and AidWeather guardrails

## Why this note exists

AidWeather uses NASA POWER as an upstream public data service. That means AidWeather should make requests in a way that is efficient, respectful, and easy to audit.

This note is for:

- **Users** who want to know what NASA POWER allows.
- **Developers** who want to keep AidWeather aligned with NASA POWER guidance.

---

## What AidWeather uses from NASA POWER

AidWeather is primarily oriented around the **Temporal APIs**:

- **Hourly API** — analysis-ready hourly values.
- **Daily API** — daily average / min / max style values.
- **Monthly API** — monthly and annual summaries.
- **Climatology API** — climatological summaries over predefined or custom periods.

For larger-scale or bulk workflows, NASA POWER also provides:

- **AWS direct data access** to the POWER Analysis Ready Datastore (ARD), recommended by NASA for direct online access to the datastore.
- **ArcGIS services** for geospatial access patterns.

---

## The most important NASA POWER limits

### 1) Maximum parameters per request

| Service | Point request limit | Regional request limit |
|---|---:|---:|
| Hourly | **15 parameters** | not described in the uploaded hourly page because hourly currently focuses on point usage |
| Daily | **20 parameters** | **1 parameter** |
| Monthly | **20 parameters** | **1 parameter** |
| Climatology | **20 parameters** | **1 parameter** |

### 2) Repeated requests to the same grid cell can get you blocked

NASA POWER explicitly warns that if you are downloading the daily catalog on the **0.5° × 0.5° global grid**, you should only submit **one request per cell**. If an application keeps requesting the same relative location, it can be **blocked**.

### 3) Time standard matters

For the uploaded temporal docs:

- **Hourly API** supports **LST** and **UTC**, and defaults to **LST**.
- **Daily API** supports **LST** and **UTC**, and defaults to **LST**.
- **LST is not the same as the local government time zone**. It is solar-time based, so timestamps may not match civil time.

### 4) Wind elevation and custom surface inputs are constrained

For the relevant APIs:

- wind elevation is only available for the **Point** spatial option;
- wind elevation must be between **10 m and 300 m**;
- if a custom wind surface is used, a matching surface elevation should also be provided.

### 5) Site elevation is point-only

If site elevation is supplied, NASA POWER can return pressure adjusted to the provided elevation, but this is only supported for the **Point** spatial option.

### 6) API services have operational service limits

NASA POWER states that its API services have **rate, data transfer, and timeout limits**. For workflows that are too large for the live API, NASA recommends direct access to the **AWS ARD Zarr datastore**.

---

## What this means in practice for AidWeather users

### Safe usage rules

1. Prefer **point requests** for normal AidWeather workflows.
2. Keep parameter lists within the documented cap for the chosen temporal service.
3. Avoid repeatedly re-requesting the same location and same parameter bundle.
4. Use **caching** whenever possible.
5. Be careful with **parallel requests**. Faster is not always better when the upstream service is shared.
6. Use **AWS ARD access** for bulk historical extraction, full-grid workflows, or very large repeated jobs.
7. Be explicit about **time standard** when timestamp interpretation matters.

### Practical examples

- If you want daily weather for one farm with 6–10 variables, the standard API is appropriate.
- If you want hourly weather for one farm with 18 variables, split the request into chunks because hourly is capped at 15 parameters.
- If you want a large regional archive or repeated catalog-scale downloads, do **not** hammer the API; move to AWS-based access.

---

## How AidWeather helps users adhere to the rules today

AidWeather already contains several behaviors that help reduce unnecessary pressure on NASA POWER:

### 1) Local caching

The current `PowerClient` stores responses in a local **SQLite cache**, reducing repeated downloads for the same request footprint.

Why this helps:

- fewer duplicate requests;
- faster repeated analyses;
- less chance of triggering rate or blocking behavior.

### 2) Automatic .env loading for API Keys

The `PowerClient` automatically searches for a `.env` file in the current working directory. If found, it loads environment variables, including `NASA_POWER_API_KEY`.

Why this helps:

- simplifies configuration for users;
- encourages keeping secrets out of code and notebooks;
- enables easy switching between different API keys or "DEMO_KEY".

### 3) Retry handling for transient failures

The client uses an HTTP session configured with retries for statuses such as **429**, **500**, **502**, **503**, and **504**.

Why this helps:

- avoids brittle failures from temporary service issues;
- reduces the temptation to re-run the same job manually many times.

### 4) Explicit User-Agent

The client sends a recognizable `User-Agent` string.

Why this helps:

- improves service transparency;
- is a better upstream-citizen practice than anonymous scraping-like traffic.

### 5) Regional and point workflows are separated

AidWeather has separate methods for point and regional access patterns.

Why this helps:

- makes it easier to apply different validation rules for each endpoint;
- reduces accidental misuse of regional requests.

---

## Important caveat: AidWeather should enforce more of these limits explicitly

From the current code snapshot, AidWeather already helps through caching and retries, but it does **not yet appear to fully enforce all NASA POWER request-policy rules at the interface level**.

That means developers should still add or strengthen guardrails such as:

- parameter-count validation before request submission;
- hourly request chunking when `len(params) > 15`;
- daily/monthly/climatology chunking when `len(params) > 20`;
- hard validation that regional requests carry only **one** parameter;
- optional request throttling / backoff between concurrent jobs;
- warnings when users try repeated dense sampling of the same 0.5° cell;
- a clear switch to an **AWS-backed bulk mode** for large jobs.

---

## Recommended AidWeather policy

### For users

AidWeather should present NASA POWER as a **shared public service**, not as an unlimited backend.

### For developers

AidWeather should implement this policy:

- **Fail fast** on invalid request shapes.
- **Auto-split** oversized point requests into compliant chunks.
- **Reject** invalid regional multi-parameter requests.
- **Default to caching**.
- **Use retries conservatively**, not aggressively.
- **Document LST vs UTC clearly**.
- **Recommend AWS ARD** for bulk extraction.

---

## Suggested user-facing wording for docs

> AidWeather uses NASA POWER responsibly. The library caches repeated requests, uses resilient retry logic for transient failures, and should keep requests within NASA POWER’s documented limits. For bulk or catalog-scale access, prefer direct AWS ARD access instead of repeatedly calling the live API.

---

## Suggested developer checklist

- [ ] Validate parameter count by temporal API.
- [ ] Enforce one-parameter rule for regional daily/monthly/climatology requests.
- [ ] Add request chunking helpers.
- [ ] Add optional throttle controls for parallel jobs.
- [ ] Log whether data came from cache or live API.
- [ ] Expose time-standard choice clearly in the public API.
- [ ] Add an advanced bulk-data path or documentation for AWS ARD.

---

## Primary source pages reviewed

- NASA POWER Docs — Temporal APIs overview
- NASA POWER Docs — Hourly API
- NASA POWER Docs — Daily API
- NASA POWER Docs — Monthly API
- NASA POWER Docs — Climatology API
- NASA POWER Docs — AWS direct datastore access
- NASA POWER Docs — ArcGIS services
