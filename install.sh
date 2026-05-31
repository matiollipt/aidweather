#!/usr/bin/env bash
# =============================================================================
#  aidweather Installer
#
#  Usage:
#    ./install.sh [--dev] [--no-venv] [--venv-path DIR] [--clean] [-y]
#
#  Remote one-liner:
#    curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash
# =============================================================================

set -eu

info()  { printf 'info: %s\n' "$1"; }
ok()    { printf 'ok: %s\n' "$1"; }
warn()  { printf 'warning: %s\n' "$1"; }
die()   { printf 'error: %s\n' "$1" >&2; exit 1; }

# Defaults
DEV_MODE=false
USE_VENV=true
VENV_PATH=".venv"
CLEAN_VENV=false
AUTO_YES=false
USE_PIPX=false

# Help
show_help() {
    printf '\naidweather installer\n\n'
    printf 'Usage:\n'
    printf '  ./install.sh [options]\n\n'
    printf 'Options:\n'
    printf '  %-24s %s\n' "--pipx"         "Install globally in isolated env using pipx"
    printf '  %-24s %s\n' "--dev"           "Install developer tools (pytest, ruff, mypy, build)"
    printf '  %-24s %s\n' "--no-venv"       "Skip venv creation (use active/global Python)"
    printf '  %-24s %s\n' "--venv-path DIR" "Custom venv path (default: .venv)"
    printf '  %-24s %s\n' "--clean"         "Wipe and recreate venv before installing"
    printf '  %-24s %s\n' "-y, --yes"       "Skip confirmation prompts"
    printf '  %-24s %s\n' "-h, --help"      "Show this help and exit"
    printf '\nExamples:\n'
    printf '  ./install.sh                         # Default install in .venv\n'
    printf '  ./install.sh --pipx                  # Install globally via pipx\n'
    printf '  ./install.sh --dev --clean           # Fresh install with dev tools\n'
    printf '  curl -fsSL .../install.sh | bash -s -- --dev -y\n\n'
    exit 0
}

# Argument parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --pipx)         USE_PIPX=true;  shift ;;
        --dev|-d)       DEV_MODE=true;  shift ;;
        --no-venv)      USE_VENV=false; shift ;;
        --venv-path)    VENV_PATH="$2"; shift 2 ;;
        --clean)        CLEAN_VENV=true; shift ;;
        -y|--yes)       AUTO_YES=true;  shift ;;
        -h|--help)      show_help ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# Detect context (clone vs in-repo)
if [ -f "pyproject.toml" ] && [ -d "src/aidweather" ]; then
    REPO_ROOT="$PWD"
else
    REPO_ROOT="$PWD/aidweather"
fi

# Python detection
detect_python() {
    for cmd in python3 python; do
        if command -v "$cmd" > /dev/null 2>&1; then
            ver=$("$cmd" -c 'import sys; v=sys.version_info; print(v.major, v.minor)')
            maj=$(echo "$ver" | cut -d' ' -f1)
            min=$(echo "$ver" | cut -d' ' -f2)
            if [ "$maj" -ge 3 ] && [ "$min" -ge 9 ]; then
                echo "$cmd"; return 0
            fi
        fi
    done
    return 1
}

PYTHON_BIN=$(detect_python) || die "Python 3.9+ required but not found. Please install Python first."
PY_VER=$("$PYTHON_BIN" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}.{v.micro}")')
ok "Python $PY_VER  ($PYTHON_BIN)"

# Bootstrap: clone if not inside the repo
if [ "$REPO_ROOT" != "$PWD" ]; then
    info "Bootstrap mode: downloading aidweather"
    [ -d "$REPO_ROOT" ] && rm -rf "$REPO_ROOT"
    
    if command -v git > /dev/null 2>&1; then
        git clone https://github.com/matiollipt/aidweather.git "$REPO_ROOT" \
            || die "Clone failed. Check your internet connection."
        ok "Repository cloned to $REPO_ROOT"
    elif command -v curl > /dev/null 2>&1 && command -v unzip > /dev/null 2>&1; then
        info "git not found. Falling back to downloading ZIP via curl..."
        curl -fsSL -o "$PWD/aidweather-main.zip" https://github.com/matiollipt/aidweather/archive/refs/heads/main.zip \
            || die "Failed to download ZIP. Check your internet connection."
        unzip -q "$PWD/aidweather-main.zip" -d "$PWD" \
            || die "Failed to extract ZIP file."
        mv "$PWD/aidweather-main" "$REPO_ROOT"
        rm "$PWD/aidweather-main.zip"
        ok "Repository downloaded and extracted to $REPO_ROOT"
    elif command -v wget > /dev/null 2>&1 && command -v unzip > /dev/null 2>&1; then
        info "git not found. Falling back to downloading ZIP via wget..."
        wget -q -O "$PWD/aidweather-main.zip" https://github.com/matiollipt/aidweather/archive/refs/heads/main.zip \
            || die "Failed to download ZIP. Check your internet connection."
        unzip -q "$PWD/aidweather-main.zip" -d "$PWD" \
            || die "Failed to extract ZIP file."
        mv "$PWD/aidweather-main" "$REPO_ROOT"
        rm "$PWD/aidweather-main.zip"
        ok "Repository downloaded and extracted to $REPO_ROOT"
    else
        die "Neither git nor curl/wget + unzip were found. Please install git or curl and unzip to proceed."
    fi
    cd "$REPO_ROOT" || exit 1
fi

# Virtual environment
setup_venv() {
    case "$VENV_PATH" in
        /*) ABS_VENV="$VENV_PATH" ;;
        *)  ABS_VENV="$REPO_ROOT/$VENV_PATH" ;;
    esac

    if [ "$CLEAN_VENV" = true ] && [ -d "$ABS_VENV" ]; then
        info "Removing existing venv at $ABS_VENV..."
        rm -rf "$ABS_VENV"
    fi

    if [ ! -d "$ABS_VENV" ]; then
        info "Creating virtual environment at $ABS_VENV..."
        if command -v uv > /dev/null 2>&1; then
            uv venv "$ABS_VENV"
        else
            "$PYTHON_BIN" -m venv "$ABS_VENV"
        fi
    fi

    PYTHON_BIN="$ABS_VENV/bin/python"
    ok "Virtual environment ready: $ABS_VENV"
}

pip_install() {
    if command -v uv > /dev/null 2>&1; then
        if [ "$USE_VENV" = true ]; then
            VIRTUAL_ENV="$ABS_VENV" uv pip install "$@"
        else
            uv pip install --system "$@"
        fi
    else
        "$PYTHON_BIN" -m pip install --quiet "$@"
    fi
}

if [ "$USE_PIPX" = true ]; then
    if ! command -v pipx > /dev/null 2>&1; then
        die "pipx is not installed. Please install pipx first (e.g., 'brew install pipx' or 'sudo apt install pipx') or run without --pipx."
    fi
    USE_VENV=false
fi

if [ "$USE_VENV" = true ]; then
    setup_venv
else
    if [ "$USE_PIPX" = true ]; then
        info "Skipping standard venv; installing via pipx"
    else
        info "Skipping venv; using active/global Python"
    fi
    ABS_VENV=""
fi

venv_label=$( [ "$USE_VENV" = true ] && echo "$VENV_PATH" || echo "global/active Python" )
if [ "$USE_PIPX" = true ]; then
    venv_label="pipx (isolated user application)"
fi
info "Installing aidweather"
info "Environment: $venv_label"
info "Developer tools: $DEV_MODE"

if [ "$AUTO_YES" = false ]; then
    if [ -t 0 ] || [ -c /dev/tty ]; then
        printf 'Proceed with installation? [Y/n] '
        if [ -t 0 ]; then
            read -r answer
        else
            read -r answer </dev/tty 2>/dev/null || answer="y"
        fi
        case "$answer" in
            [Nn]*) info "Installation cancelled."; exit 0 ;;
        esac
    else
        warn "Non-interactive terminal detected. Proceeding with installation..."
    fi
fi

# Install core package
if [ "$USE_PIPX" = true ]; then
    info "Installing package with pipx"
    if pipx list | grep -q "package aidweather"; then
        info "Existing pipx install found; reinstalling"
        pipx install --force .
    else
        pipx install .
    fi
else
    info "Installing package"
    pip_install -e .
fi

# Developer tools
if [ "$DEV_MODE" = true ]; then
    if [ "$USE_PIPX" = true ]; then
        info "Installing developer tools into pipx environment"
        pipx inject aidweather \
            "pytest>=8.0.0" \
            "pytest-cov>=4.1.0" \
            "requests-mock>=1.11.0" \
            "ruff>=0.3.0" \
            "mypy>=1.9.0" \
            "build>=1.1.0" \
            "twine>=5.0.0"
    else
        info "Installing developer tools"
        pip_install \
            "pytest>=8.0.0" \
            "pytest-cov>=4.1.0" \
            "requests-mock>=1.11.0" \
            "ruff>=0.3.0" \
            "mypy>=1.9.0" \
            "build>=1.1.0" \
            "twine>=5.0.0"
    fi
fi

# Smoke test
info "Checking installation"
if [ "$USE_PIPX" = true ]; then
    # Try the standard pipx binary directory first to avoid other env pollution
    PIPX_LOCAL_BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}/aidweather"
    if [ -f "$PIPX_LOCAL_BIN" ] && "$PIPX_LOCAL_BIN" --help > /dev/null 2>&1; then
        # Warn if it's not in PATH or not the active one
        if ! command -v aidweather > /dev/null 2>&1 || [ "$(command -v aidweather)" != "$PIPX_LOCAL_BIN" ]; then
            warn "Note: $PIPX_LOCAL_BIN is not the active 'aidweather' in your PATH (found $(command -v aidweather || echo 'none') instead)."
            warn "You may need to run 'pipx ensurepath' and restart your terminal."
        fi
    elif command -v aidweather > /dev/null 2>&1 && aidweather --help > /dev/null 2>&1; then
        :
    else
        die "Smoke test failed: Could not locate a working aidweather executable."
    fi
else
    "$PYTHON_BIN" - <<'PYEOF'
import sys

try:
    __import__("aidweather")
except ImportError:
    sys.exit(1)

PYEOF
fi

# Done
printf 'aidweather installed successfully.\n'

if [ "$USE_PIPX" = true ]; then
    printf 'Run: aidweather --help\n'
    printf '  If the command is not found, ensure pipx binary path is in your PATH:\n'
    printf '    pipx ensurepath\n\n'
elif [ "$USE_VENV" = true ]; then
    printf 'Activate the environment with:\n'
    printf '  source %s/bin/activate\n' "$VENV_PATH"
fi
