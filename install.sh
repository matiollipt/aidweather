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
USE_UV_TOOL=false

# Help
show_help() {
    printf '\naidweather installer\n\n'
    printf 'Usage:\n'
    printf '  ./install.sh [options]\n\n'
    printf 'Options:\n'
    printf '  %-24s %s\n' "--uv-tool"       "Install globally in isolated env using uv tool"
    printf '  %-24s %s\n' "--dev"           "Install developer tools (pytest, ruff, mypy, build)"
    printf '  %-24s %s\n' "--no-venv"       "Skip venv creation (use active/global Python)"
    printf '  %-24s %s\n' "--venv-path DIR" "Custom venv path (default: .venv)"
    printf '  %-24s %s\n' "--clean"         "Wipe and recreate venv before installing"
    printf '  %-24s %s\n' "-y, --yes"       "Skip confirmation prompts"
    printf '  %-24s %s\n' "-h, --help"      "Show this help and exit"
    printf '\nExamples:\n'
    printf '  ./install.sh                         # Default install in .venv\n'
    printf '  ./install.sh --uv-tool               # Install globally via uv tool\n'
    printf '  ./install.sh --dev --clean           # Fresh install with dev tools\n'
    printf '  curl -fsSL .../install.sh | bash -s -- --dev -y\n\n'
    exit 0
}

# Argument parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --uv-tool)      USE_UV_TOOL=true; shift ;;
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

if [ "$USE_UV_TOOL" = true ]; then
    if ! command -v uv > /dev/null 2>&1; then
        die "uv is not installed. Please install uv first (see https://docs.astral.sh/uv/getting-started/installation/) or run without --uv-tool."
    fi
    USE_VENV=false
fi

if [ "$USE_VENV" = true ]; then
    setup_venv
else
    if [ "$USE_UV_TOOL" = true ]; then
        info "Skipping standard venv; installing via uv tool"
    else
        info "Skipping venv; using active/global Python"
    fi
    ABS_VENV=""
fi

venv_label=$( [ "$USE_VENV" = true ] && echo "$VENV_PATH" || echo "global/active Python" )
if [ "$USE_UV_TOOL" = true ]; then
    venv_label="uv tool (isolated user application)"
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
if [ "$USE_UV_TOOL" = true ]; then
    info "Installing package with uv tool"
    if uv tool list 2>/dev/null | grep -q "aidweather"; then
        info "Existing uv tool install found; reinstalling"
        uv tool install --reinstall .
    else
        uv tool install .
    fi
else
    info "Installing package"
    if [ "$DEV_MODE" = true ]; then
        pip_install -e .
    else
        pip_install .
    fi
fi

# Developer tools
if [ "$DEV_MODE" = true ]; then
    if [ "$USE_UV_TOOL" = true ]; then
        info "Installing developer tools into uv tool environment"
        uv tool install \
            --with "pytest>=8.0.0" \
            --with "pytest-cov>=4.1.0" \
            --with "requests-mock>=1.11.0" \
            --with "ruff>=0.3.0" \
            --with "mypy>=1.9.0" \
            --with "build>=1.1.0" \
            --with "twine>=5.0.0" \
            --reinstall .
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
if [ "$USE_UV_TOOL" = true ]; then
    # Try the standard uv tool binary directory first to avoid other env pollution
    UV_TOOL_LOCAL_BIN="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}/aidweather"
    if [ -f "$UV_TOOL_LOCAL_BIN" ] && "$UV_TOOL_LOCAL_BIN" --help > /dev/null 2>&1; then
        # Warn if it's not in PATH or not the active one
        if ! command -v aidweather > /dev/null 2>&1 || [ "$(command -v aidweather)" != "$UV_TOOL_LOCAL_BIN" ]; then
            warn "Note: $UV_TOOL_LOCAL_BIN is not the active 'aidweather' in your PATH (found $(command -v aidweather || echo 'none') instead)."
            warn "You may need to run 'uv tool update-shell' and restart your terminal."
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

if [ "$USE_UV_TOOL" = true ]; then
    printf 'Run: aidweather --help\n'
    printf '  If the command is not found, ensure uv tool binary path is in your PATH:\n'
    printf '    uv tool update-shell\n\n'
elif [ "$USE_VENV" = true ]; then
    printf 'Activate the environment with:\n'
    printf '  source %s/bin/activate\n' "$VENV_PATH"
fi
