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

# ── Colour helpers (stripped when stdout is not a terminal) ──────────────────
if [ -t 1 ]; then
    GREEN='\033[1;32m'       # aid
    ORANGE='\033[38;5;214m'  # weather
    PURPLE='\033[1;35m'      # accents / separators
    BLUE='\033[1;34m'
    YELLOW='\033[1;33m'
    RED='\033[1;31m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN=''; ORANGE=''; PURPLE=''; BLUE=''; YELLOW=''; RED=''; BOLD=''; NC=''
fi

info()  { printf '%b▸%b  %s\n'        "$BLUE"   "$NC" "$1"; }
ok()    { printf '%b✔%b  %s\n'        "$GREEN"  "$NC" "$1"; }
warn()  { printf '%b⚠%b  %s\n'        "$YELLOW" "$NC" "$1"; }
die()   { printf '%b✖  Error:%b %s\n' "$RED"    "$NC" "$1" >&2; exit 1; }
sep()   { printf '%b%s%b\n' "$PURPLE" "────────────────────────────────────────────────" "$NC"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
DEV_MODE=false
USE_VENV=true
VENV_PATH=".venv"
CLEAN_VENV=false
AUTO_YES=false

# ── Help ──────────────────────────────────────────────────────────────────────
show_help() {
    printf '\n%baidweather%b Installer\n\n' "$GREEN$BOLD" "$NC"
    printf 'Usage:\n'
    printf '  ./install.sh [options]\n\n'
    printf 'Options:\n'
    printf '  %-24s %s\n' "--dev"           "Install developer tools (pytest, ruff, mypy, build)"
    printf '  %-24s %s\n' "--no-venv"       "Skip venv creation (use active/global Python)"
    printf '  %-24s %s\n' "--venv-path DIR" "Custom venv path (default: .venv)"
    printf '  %-24s %s\n' "--clean"         "Wipe and recreate venv before installing"
    printf '  %-24s %s\n' "-y, --yes"       "Skip confirmation prompts"
    printf '  %-24s %s\n' "-h, --help"      "Show this help and exit"
    printf '\nExamples:\n'
    printf '  ./install.sh                         # Default install in .venv\n'
    printf '  ./install.sh --dev --clean           # Fresh install with dev tools\n'
    printf '  curl -fsSL .../install.sh | bash -s -- --dev -y\n\n'
    exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --dev|-d)       DEV_MODE=true;  shift ;;
        --no-venv)      USE_VENV=false; shift ;;
        --venv-path)    VENV_PATH="$2"; shift 2 ;;
        --clean)        CLEAN_VENV=true; shift ;;
        -y|--yes)       AUTO_YES=true;  shift ;;
        -h|--help)      show_help ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# ── Banner (aid=green, weather=orange, accents=purple) ────────────────────────
print_banner() {
    printf '\n'
    if command -v figlet > /dev/null 2>&1; then
        # Dynamic: use figlet and split at column 21 (width of "aid")
        figlet "aidweather" | while IFS= read -r line; do
            aid_col="${line:0:21}"
            wx_col="${line:21}"
            printf '%b%s%b%b%s%b\n' "$GREEN" "$aid_col" "$NC" "$ORANGE" "$wx_col" "$NC"
        done
    else
        # Pre-baked fallback (same figlet default font output)
        printf '%b                     %b%b                                                 %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b          "        # %b%b                        m    #                   %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b  mmm   mmm     mmm# %b%bm     m  mmm    mmm   mm#mm  # mm    mmm    m mm %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b "   #    #    #" "# %b%b"m m m" #"  #  "   #    #    #"  #  #"  #   #"  "%b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b m""#    #    #   # %b%b #m#m#  #""""  m"""#    #    #   #  #""""   #    %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b "mm"#  mm#mm  "#m## %b%b  # #   "#mm"  "mm"#    "mm  #   #  "#mm"   #    %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
        printf '%b                     %b%b                                                 %b\n' "$GREEN" "$NC" "$ORANGE" "$NC"
    fi
    printf '\n'
    printf '  %baid%b%bweather%b  —  Weather Data Toolkit for Agriculture\n' \
        "$GREEN$BOLD" "$NC" "$ORANGE$BOLD" "$NC"
    sep
    printf '\n'
}

print_banner

# ── Detect context (clone vs in-repo) ─────────────────────────────────────────
if [ -f "pyproject.toml" ] && [ -d "src/aidweather" ]; then
    REPO_ROOT="$PWD"
else
    REPO_ROOT="$PWD/aidweather"
fi

# ── Python detection ──────────────────────────────────────────────────────────
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

# ── Bootstrap: clone if not inside the repo ───────────────────────────────────
if [ "$REPO_ROOT" != "$PWD" ]; then
    info "Bootstrap mode — downloading/cloning aidweather..."
    [ -d "$REPO_ROOT" ] && rm -rf "$REPO_ROOT"
    
    if command -v git > /dev/null 2>&1; then
        git clone https://github.com/matiollipt/aidweather.git "$REPO_ROOT" \
            || die "Clone failed. Check your internet connection."
        ok "Repository cloned → $REPO_ROOT"
    elif command -v curl > /dev/null 2>&1 && command -v unzip > /dev/null 2>&1; then
        info "git not found. Falling back to downloading ZIP via curl..."
        curl -fsSL -o "$PWD/aidweather-main.zip" https://github.com/matiollipt/aidweather/archive/refs/heads/main.zip \
            || die "Failed to download ZIP. Check your internet connection."
        unzip -q "$PWD/aidweather-main.zip" -d "$PWD" \
            || die "Failed to extract ZIP file."
        mv "$PWD/aidweather-main" "$REPO_ROOT"
        rm "$PWD/aidweather-main.zip"
        ok "Repository downloaded and extracted → $REPO_ROOT"
    elif command -v wget > /dev/null 2>&1 && command -v unzip > /dev/null 2>&1; then
        info "git not found. Falling back to downloading ZIP via wget..."
        wget -q -O "$PWD/aidweather-main.zip" https://github.com/matiollipt/aidweather/archive/refs/heads/main.zip \
            || die "Failed to download ZIP. Check your internet connection."
        unzip -q "$PWD/aidweather-main.zip" -d "$PWD" \
            || die "Failed to extract ZIP file."
        mv "$PWD/aidweather-main" "$REPO_ROOT"
        rm "$PWD/aidweather-main.zip"
        ok "Repository downloaded and extracted → $REPO_ROOT"
    else
        die "Neither git nor curl/wget + unzip were found. Please install git or curl and unzip to proceed."
    fi
    cd "$REPO_ROOT" || exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
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

if [ "$USE_VENV" = true ]; then
    setup_venv
else
    info "Skipping venv — using active/global Python."
    ABS_VENV=""
fi

# ── Installation plan ─────────────────────────────────────────────────────────
printf '\n'
info "Installation plan"
sep
venv_label=$( [ "$USE_VENV" = true ] && echo "$VENV_PATH" || echo "global/active Python" )
printf '  %-18s %b%s%b\n' "Package:"     "$GREEN$BOLD" "aidweather" "$NC"
printf '  %-18s %s\n'     "Environment:" "$venv_label"
printf '  %-18s %s\n'     "Dev tools:"   "$DEV_MODE"
printf '  %-18s %s\n'     "Clean venv:"  "$CLEAN_VENV"
sep
printf '\n'

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

# ── Install core package ──────────────────────────────────────────────────────
info "Installing aidweather..."
pip_install -e .
ok "aidweather installed."

# ── Developer tools ───────────────────────────────────────────────────────────
if [ "$DEV_MODE" = true ]; then
    info "Installing developer tools..."
    pip_install \
        "pytest>=8.0.0" \
        "pytest-cov>=4.1.0" \
        "requests-mock>=1.11.0" \
        "ruff>=0.3.0" \
        "mypy>=1.9.0" \
        "build>=1.1.0" \
        "twine>=5.0.0"
    ok "Developer tools installed."
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
info "Running smoke test..."
"$PYTHON_BIN" - <<'PYEOF'
import sys

checks = [("aidweather", "aidweather")]
failures = []

for mod, label in checks:
    try:
        __import__(mod)
        print(f"  ✔  {label}")
    except ImportError as exc:
        print(f"  ✖  {label}  ({exc})")
        failures.append(label)

if failures:
    print(f"\n  {len(failures)} import(s) failed.")
    sys.exit(1)
else:
    print("\n  All imports OK.")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
printf '\n'
printf '%b══════════════════════════════════════════════════%b\n' "$GREEN" "$NC"
printf '%b  ✔  %baid%b%bweather%b%b setup complete%b\n' \
    "$GREEN$BOLD" "$NC" "$GREEN$BOLD" "$NC" "$ORANGE$BOLD" "$NC$BOLD" "$NC"
printf '%b══════════════════════════════════════════════════%b\n\n' "$GREEN" "$NC"

if [ "$USE_VENV" = true ]; then
    printf '  %bActivate your environment:%b\n' "$PURPLE$BOLD" "$NC"
    printf '    source %s/bin/activate\n\n' "$VENV_PATH"
fi
