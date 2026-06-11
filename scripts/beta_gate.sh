#!/usr/bin/env bash
# =============================================================================
#  AidWeather Beta Gate
#
#  Automated build, install, and validation check for 0.1.0b1 beta-readiness.
# =============================================================================

set -euo pipefail

# Print helper functions
info() { printf "\r\033[36mℹ\033[0m %s\n" "$1"; }
ok()   { printf "\r\033[32m✔\033[0m %s\n" "$1"; }
err()  { printf "\r\033[31m✘\033[0m %s\n" "$1" >&2; }

# Step 1: Clean build artifacts
info "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ .tox/ .pytest_cache/
ok "Clean completed."

# Step 2: Install build/dev tools to ensure environment has everything
info "Installing/updating build and validation tools..."
python -m pip install -U build twine pytest requests-mock ruff mypy --quiet
ok "Build & validation tools ready."

# Step 3: Build package
info "Building wheel and sdist..."
python -m build
ok "Package built successfully."

# Step 4: Verify config.json inclusion in built wheel
info "Verifying built wheel contents..."
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
if python -c "import zipfile; z = zipfile.ZipFile('$WHEEL_FILE'); assert any('config.json' in name for name in z.namelist())"; then
    ok "Verification success: config.json is present in the built wheel."
else
    err "Verification failed: config.json is MISSING from the built wheel."
    exit 1
fi

# Step 5: Install built wheel into a fresh temporary virtual environment
TEMP_VENV=$(mktemp -d -t aidweather-gate-XXXXXX)
info "Creating temporary virtual environment at $TEMP_VENV..."
python -m venv "$TEMP_VENV"
# Clean up temporary venv on exit
trap 'rm -rf "$TEMP_VENV"' EXIT

VENV_PYTHON="$TEMP_VENV/bin/python"
VENV_PIP="$TEMP_VENV/bin/pip"
VENV_CLI="$TEMP_VENV/bin/aidweather"

info "Installing built wheel into the fresh virtual environment..."
"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install "$WHEEL_FILE" --quiet
ok "Wheel installed."

# Step 6: Import smoke test
info "Running import smoke test..."
"$VENV_PYTHON" -c "
from aidweather import PowerClient
from aidweather.config import cfg
assert cfg.params('default')
print('Import smoke test successful!')
"
ok "Import smoke test passed."

# Step 7: CLI smoke test
info "Running CLI smoke test..."
"$VENV_CLI" --help > /dev/null
"$VENV_CLI" params list > /dev/null
ok "CLI smoke test passed."

# Step 8: Run pytest
info "Running pytest..."
python -m pytest -q
ok "Pytest passed."

# Step 9: Run Ruff linter
info "Running ruff check..."
ruff check .
ok "Ruff linter check passed."

# Step 10: Run Mypy type checker
info "Running mypy check..."
mypy src/aidweather
ok "Mypy type check passed."

ok "AidWeather beta gate completed successfully. Ready for 0.1.0b1 release!"
