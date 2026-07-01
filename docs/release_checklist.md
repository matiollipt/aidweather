# Release Checklist

## Phase 1 — Pre-flight

1. Confirm the version in `pyproject.toml`, `src/aidweather/__init__.py`, and `CHANGELOG.md`.
2. Run the unit suite:

   ```bash
   uv run --with-editable . --extra test pytest -q
   ```

3. Run lint and type checks:

   ```bash
   uv run ruff check src/
   uv run mypy src/aidweather
   ```

4. Build and validate distributions:

   ```bash
   python -m build
   python -m twine check dist/*
   ```

5. Smoke test the built wheel in a clean environment:

   ```bash
   python -m venv /tmp/aidweather-wheel-smoke
   /tmp/aidweather-wheel-smoke/bin/python -m pip install dist/*.whl
   /tmp/aidweather-wheel-smoke/bin/python -c "import aidweather; print(aidweather.__version__)"
   /tmp/aidweather-wheel-smoke/bin/aidweather --help
   /tmp/aidweather-wheel-smoke/bin/aidweather params list
   ```

6. Review README beta status, NASA POWER attribution, and changelog notes.
7. Confirm no generated artifacts (e.g. `outputs/`) are tracked or bundled — check
   `git status` and inspect the sdist tarball contents.

## Phase 2 — TestPyPI dry run

One-time setup, before the first ever TestPyPI publish: register a pending trusted
publisher at https://test.pypi.org/manage/account/publishing/ with:

| Field | Value |
|---|---|
| PyPI Project Name | `aidweather` |
| Owner | `matiollipt` |
| Repository name | `aidweather` |
| Workflow name | `publish-testpypi.yml` |
| Environment name | `testpypi` |

Then, for every dry run:

1. Trigger `.github/workflows/publish-testpypi.yml` via `workflow_dispatch` in the
   Actions tab (no tag or GitHub Release needed).
2. Approve the `testpypi` environment when prompted.
3. Verify the install in a clean environment:

   ```bash
   python -m venv /tmp/aidweather-testpypi-smoke
   /tmp/aidweather-testpypi-smoke/bin/pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     aidweather==<version>
   /tmp/aidweather-testpypi-smoke/bin/aidweather --help
   ```

## Phase 3 — Real release (once ready to graduate off TestPyPI)

One-time setup, before the first ever PyPI publish: register a pending trusted publisher
at https://pypi.org/manage/account/publishing/ with:

| Field | Value |
|---|---|
| PyPI Project Name | `aidweather` |
| Owner | `matiollipt` |
| Repository name | `aidweather` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then, for the actual release:

1. Merge all release-prep changes to the default branch.
2. Create and push an annotated tag:

   ```bash
   git tag -a vX.Y.Z -m "aidweather X.Y.Z — <summary>"
   git push origin vX.Y.Z
   ```

3. Create the GitHub Release from that tag, using the matching `CHANGELOG.md` section as
   the release notes body (skip "Generate release notes" so the hand-written changelog
   stays primary).
4. Publish the release — this triggers `.github/workflows/publish.yml`. Approve the
   `pypi` environment when prompted.
5. Verify: check `https://pypi.org/project/aidweather/X.Y.Z/` is live, then in a clean
   environment run `pip install aidweather==X.Y.Z && aidweather --help`.
