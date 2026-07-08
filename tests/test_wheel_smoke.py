# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.skipif(
    os.environ.get("AIDWEATHER_RUN_WHEEL_SMOKE") != "1",
    reason="AIDWEATHER_RUN_WHEEL_SMOKE=1 is not set"
)
def test_wheel_smoke():
    """Builds the package wheel and installs it in a temporary virtualenv, running CLI smoke tests."""
    # Ensure dist folder is clean or exists
    dist_dir = os.path.join(os.getcwd(), "dist")
    if os.path.exists(dist_dir):
        import shutil
        shutil.rmtree(dist_dir)

    # Build sdist and wheel using 'uv build'
    subprocess.check_call(["uv", "build"], cwd=os.getcwd())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create virtualenv
        venv_dir = os.path.join(tmpdir, "venv")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        
        # Determine python/pip executable paths in the venv
        if os.name == "nt":
            python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
            cli_bin = os.path.join(venv_dir, "Scripts", "aidweather.exe")
        else:
            python_bin = os.path.join(venv_dir, "bin", "python")
            cli_bin = os.path.join(venv_dir, "bin", "aidweather")
            
        # Find built wheel in dist/
        wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
        assert wheels, "No wheel built in dist/"
        latest_wheel = sorted(wheels)[-1]
        wheel_path = os.path.join(dist_dir, latest_wheel)
        
        # Install wheel
        subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([python_bin, "-m", "pip", "install", wheel_path])
        
        # Check version import
        output_version = subprocess.check_output([python_bin, "-c", "import aidweather; print(aidweather.__version__)"]).decode().strip()
        from aidweather import __version__
        assert output_version == __version__
        
        # Check CLI works
        subprocess.check_call([cli_bin, "--help"])
        subprocess.check_call([cli_bin, "params", "list"])
