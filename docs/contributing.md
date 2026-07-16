# Contributing & Testing Guide — `aidweather`

Thank you for contributing to `aidweather`. Please follow these setup and testing guidelines.

---

## 1. Development Setup

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
uv sync --extra test
```

---

## 2. Running Quality Verification Commands

Before submitting code, run the suite of automated checks:

### Automated Unit Tests
```bash
uv run --with-editable . --extra test pytest -q
```

### Type Checking with mypy
```bash
uv run mypy src/aidweather
```

---

## 3. Live Integration Tests

By default, unit tests use recorded and mocked JSON responses to avoid hitting live external servers. To run live integration tests against the live NASA POWER service endpoints:

```bash
AIDWEATHER_RUN_LIVE_TESTS=1 uv run --with-editable . --extra test pytest
```
