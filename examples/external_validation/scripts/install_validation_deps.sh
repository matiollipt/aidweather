#!/usr/bin/env bash
# Install external dependencies needed for external_validation.py
# Run from the project root: bash examples/external_validation/scripts/install_validation_deps.sh

set -e

uv pip install \
    matplotlib \
    seaborn \
    scikit-learn \
    meteostat \
    openmeteo-requests \
    requests-cache \
    retry-requests

echo "Done. Run the script with: uv run examples/external_validation/scripts/external_validation.py"
