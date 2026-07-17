# Contributing & Testing Guide — `aidweather`

Thank you for contributing to `aidweather`. Please follow these setup and testing guidelines to maintain code quality and reproducibility.

---

## 1. Development Setup

Clone the repository and install development dependencies using `uv`:

```bash
git clone https://github.com/matiollipt/aidweather.git
cd aidweather
uv sync --extra test
```

---

## 2. Running Quality Verification Commands

Before submitting code changes, run the full suite of automated quality checks:

### Automated Unit Tests
```bash
uv run --with-editable . --extra test pytest -q
```

### Code Linting & Formatting with ruff
```bash
uv run ruff check .
```

### Static Type Checking with mypy
```bash
uv run mypy src/aidweather
```

---

## 3. Live Integration Tests

By default, unit tests use recorded or mocked JSON responses to avoid unnecessary traffic to live NASA POWER servers. To execute live integration tests against official NASA POWER API endpoints:

```bash
AIDWEATHER_RUN_LIVE_TESTS=1 uv run --with-editable . --extra test pytest
```
