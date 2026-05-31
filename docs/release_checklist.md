# Release Checklist

Use this short checklist before publishing a beta or patch release.

1. Confirm the version in `pyproject.toml`, `src/aidweather/__init__.py`, and `CHANGELOG.md`.
2. Run the unit suite:

   ```bash
   uv run --with-editable . --extra test pytest -q
   ```

3. Build and validate distributions:

   ```bash
   python -m build
   python -m twine check dist/*
   ```

4. Smoke test the built wheel in a clean environment:

   ```bash
   python -m venv /tmp/aidweather-wheel-smoke
   /tmp/aidweather-wheel-smoke/bin/python -m pip install dist/*.whl
   /tmp/aidweather-wheel-smoke/bin/python -c "import aidweather; print(aidweather.__version__)"
   /tmp/aidweather-wheel-smoke/bin/aidweather --help
   /tmp/aidweather-wheel-smoke/bin/aidweather params list
   ```

5. Review README beta status, NASA POWER attribution, and changelog notes.
