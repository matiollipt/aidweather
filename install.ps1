<#
.SYNOPSIS
    aidweather Installer for Windows / PowerShell

.DESCRIPTION
    Installs the aidweather package, setting up a virtual environment or uv tool as requested.

.PARAMETER UvTool
    Install globally in an isolated environment using uv tool.

.PARAMETER Dev
    Install developer tools (pytest, ruff, mypy, build).

.PARAMETER NoVenv
    Skip virtual environment creation (use active/global Python).

.PARAMETER VenvPath
    Custom virtual environment path (default: .venv).

.PARAMETER Clean
    Wipe and recreate virtual environment before installing.

.PARAMETER Yes
    Skip confirmation prompts.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Dev -Clean
#>

[CmdletBinding()]
param (
    [switch]$UvTool,
    [switch]$Dev,
    [switch]$NoVenv,
    [string]$VenvPath = ".venv",
    [switch]$Clean,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

# Output helpers
function Info ($msg) { Write-Host "info: $msg" -ForegroundColor Cyan }
function Ok ($msg) { Write-Host "ok: $msg" -ForegroundColor Green }
function Warn ($msg) { Write-Warning $msg }
function Die ($msg) { Write-Error "error: $msg"; exit 1 }

# Python detection
function Detect-Python {
    foreach ($cmd in "py", "python", "python3") {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            try {
                $verStr = & $cmd -c "import sys; v=sys.version_info; print(f'{v.major} {v.minor}')"
                $parts = $verStr.Split(" ")
                $maj = [int]$parts[0]
                $min = [int]$parts[1]
                if ($maj -ge 3 -and $min -ge 9) {
                    return $cmd
                }
            } catch {
                # Ignore and try next
            }
        }
    }
    return $null
}

# 1. Detect Python
$PythonCmd = Detect-Python
if (-not $PythonCmd) {
    Die "Python 3.9+ required but not found. Please install Python first (or make sure it is in your PATH)."
}

$PyVer = & $PythonCmd -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"
Ok "Python $PyVer ($PythonCmd)"

# 2. Detect context (clone vs bootstrap)
if ((Test-Path "pyproject.toml") -and (Test-Path "src/aidweather")) {
    $RepoRoot = $PWD.Path
} else {
    $RepoRoot = Join-Path $PWD.Path "aidweather"
}

# Bootstrap if run outside the repository folder
if ($RepoRoot -ne $PWD.Path) {
    Info "Bootstrap mode: downloading aidweather"
    if (Test-Path $RepoRoot) {
        Remove-Item -Recurse -Force $RepoRoot
    }

    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone https://github.com/matiollipt/aidweather.git $RepoRoot
        if ($LASTEXITCODE -ne 0) {
            Die "Clone failed. Check your internet connection."
        }
        Ok "Repository cloned to $RepoRoot"
    } else {
        Info "git not found. Falling back to downloading ZIP..."
        $zipPath = Join-Path $PWD.Path "aidweather-main.zip"
        try {
            $webClient = New-Object System.Net.WebClient
            $webClient.DownloadFile("https://github.com/matiollipt/aidweather/archive/refs/heads/main.zip", $zipPath)
            Expand-Archive -Path $zipPath -DestinationPath $PWD.Path -Force
            Rename-Item -Path (Join-Path $PWD.Path "aidweather-main") -NewName "aidweather"
            Remove-Item $zipPath -Force
            Ok "Repository downloaded and extracted to $RepoRoot"
        } catch {
            Die "Failed to download or extract ZIP. Check your internet connection. (Error: $_)"
        }
    }
    Set-Location $RepoRoot
}

# 3. Setup install settings
$UseVenv = -not $NoVenv
if ($UvTool) {
    $UseVenv = $false
}

if ($UvTool -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Die "uv is not installed. Please install uv first or run without -UvTool."
}

# 4. Virtual environment creation
$AbsVenv = ""
$PythonBin = $PythonCmd

if ($UseVenv) {
    if ([System.IO.Path]::IsPathRooted($VenvPath)) {
        $AbsVenv = $VenvPath
    } else {
        $AbsVenv = Join-Path $RepoRoot $VenvPath
    }

    if ($Clean -and (Test-Path $AbsVenv)) {
        Info "Removing existing virtual environment at $AbsVenv..."
        Remove-Item -Recurse -Force $AbsVenv
    }

    if (-not (Test-Path $AbsVenv)) {
        Info "Creating virtual environment at $AbsVenv..."
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv venv $AbsVenv
        } else {
            & $PythonCmd -m venv $AbsVenv
        }
    }

    # Determine scripts location (Windows venv layout is Scripts\, Unix/macOS is bin\)
    $ScriptsFolder = Join-Path $AbsVenv "Scripts"
    if (-not (Test-Path $ScriptsFolder)) {
        $ScriptsFolder = Join-Path $AbsVenv "bin"
    }

    $PythonBin = Join-Path $ScriptsFolder "python.exe"
    if (-not (Test-Path $PythonBin)) {
        $PythonBin = Join-Path $ScriptsFolder "python"
    }
    Ok "Virtual environment ready: $AbsVenv"
}

# 5. Pip install helper
function Pip-Install {
    param([string[]]$Packages)
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        if ($UseVenv) {
            $env:VIRTUAL_ENV = $AbsVenv
            uv pip install $Packages
        } else {
            uv pip install --system $Packages
        }
    } else {
        & $PythonBin -m pip install --quiet $Packages
    }
}

# 6. Summary and Confirmation
$VenvLabel = if ($UseVenv) { $VenvPath } else { "global/active Python" }
if ($UvTool) { $VenvLabel = "uv tool (isolated user environment)" }

Info "Installing aidweather"
Info "Environment: $VenvLabel"
Info "Developer tools: $Dev"

if (-not $Yes) {
    # Check if host is interactive
    if ($Host.UI.RawUI) {
        $response = Read-Host "Proceed with installation? [Y/n]"
        if ($response -match "^[Nn]") {
            Info "Installation cancelled."
            exit 0
        }
    } else {
        Warn "Non-interactive shell detected. Proceeding with installation..."
    }
}

# 7. Execute Installation
if ($UvTool) {
    Info "Installing package with uv tool"
    if (uv tool list 2>$null | Select-String "aidweather") {
        Info "Existing uv tool install found; reinstalling"
        uv tool install --reinstall .
    } else {
        uv tool install .
    }
} else {
    Info "Installing package"
    Pip-Install @("-e", ".")
}

# 8. Install developer dependencies if requested
if ($Dev) {
    $devDeps = @(
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
        "requests-mock>=1.11.0",
        "ruff>=0.3.0",
        "mypy>=1.9.0",
        "build>=1.1.0",
        "twine>=5.0.0"
    )
    if ($UvTool) {
        Info "Installing developer tools into uv tool environment"
        uv tool install `
            --with "pytest>=8.0.0" `
            --with "pytest-cov>=4.1.0" `
            --with "requests-mock>=1.11.0" `
            --with "ruff>=0.3.0" `
            --with "mypy>=1.9.0" `
            --with "build>=1.1.0" `
            --with "twine>=5.0.0" `
            --reinstall .
    } else {
        Info "Installing developer tools"
        Pip-Install $devDeps
    }
}

# 9. Smoke check
Info "Checking installation"
if ($UvTool) {
    if (Get-Command aidweather -ErrorAction SilentlyContinue) {
        # Active in PATH
    } else {
        Warn "Could not find 'aidweather' in your current PATH."
        Warn "You may need to run 'uv tool update-shell' and restart your terminal."
    }
} else {
    & $PythonBin -c "import sys; import aidweather"
    if ($LASTEXITCODE -ne 0) {
        Die "Smoke test failed: Could not import aidweather."
    }
}

# 10. Done
Write-Host "aidweather installed successfully." -ForegroundColor Green

if ($UvTool) {
    Write-Host "Run: aidweather --help"
    Write-Host "  If the command is not found, ensure uv tool binary path is in your PATH:"
    Write-Host "    uv tool update-shell"
} elseif ($UseVenv) {
    $RelPath = Resolve-Path $AbsVenv -Relative
    Write-Host "Activate the environment with:"
    Write-Host "  For PowerShell: & $RelPath\Scripts\Activate.ps1"
    Write-Host "  For CMD:        $RelPath\Scripts\activate.bat"
}
