<#
.SYNOPSIS
    aidweather Installer for Windows / PowerShell

.DESCRIPTION
    Installs the aidweather package, setting up a virtual environment or uv tool as requested.

.PARAMETER UvTool
    Install globally in an isolated environment using uv tool.

.PARAMETER Dev
    Shortcut for -Extra test -Extra release (test tools + build/publish tools).

.PARAMETER Extra
    Install one or more pyproject.toml optional-dependency extras (e.g. "test").

.PARAMETER NoVenv
    Skip virtual environment creation (use active/global Python).

.PARAMETER VenvPath
    Custom virtual environment path (default: .venv).

.PARAMETER Clean
    Wipe and recreate virtual environment before installing.

.PARAMETER Yes
    Skip confirmation prompts.

.PARAMETER InstallSkill
    Copy the bundled Claude Code skill to $HOME\.claude\skills\aidweather.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Dev -Clean

.EXAMPLE
    .\install.ps1 -Extra test

.EXAMPLE
    .\install.ps1 -InstallSkill
#>

[CmdletBinding()]
param (
    [switch]$UvTool,
    [switch]$Dev,
    [string[]]$Extra = @(),
    [switch]$NoVenv,
    [string]$VenvPath = ".venv",
    [switch]$Clean,
    [switch]$Yes,
    [switch]$InstallSkill
)

# Collect requested pyproject.toml extras (deduplicated)
$ExtrasList = New-Object System.Collections.Generic.List[string]
foreach ($e in $Extra) {
    if (-not $ExtrasList.Contains($e)) { $ExtrasList.Add($e) }
}
if ($Dev) {
    foreach ($e in @("test", "release")) {
        if (-not $ExtrasList.Contains($e)) { $ExtrasList.Add($e) }
    }
}
$PkgSpec = if ($ExtrasList.Count -gt 0) { ".[$($ExtrasList -join ',')]" } else { "." }

$ErrorActionPreference = "Stop"

# Output helpers
function Info ($msg) { Write-Host "info: $msg" -ForegroundColor Cyan }
function Ok ($msg) { Write-Host "ok: $msg" -ForegroundColor Green }
function Warn ($msg) { Write-Warning $msg }
function Die ($msg) { Write-Error "error: $msg"; exit 1 }

# Ensure uv is installed
function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return $true
    }

    # Try to find uv in standard user installation locations
    $HomeDir = [System.Environment]::GetFolderPath("UserProfile")
    $UvPaths = @(
        (Join-Path $HomeDir ".local/bin"),
        (Join-Path $HomeDir ".cargo/bin"),
        (Join-Path $HomeDir "AppData/Roaming/local/bin")
    )
    foreach ($p in $UvPaths) {
        $uvExe = Join-Path $p "uv"
        $uvExeWin = Join-Path $p "uv.exe"
        if ((Test-Path $uvExe) -or (Test-Path $uvExeWin)) {
            $env:PATH = "$p$([System.IO.Path]::PathSeparator)$env:PATH"
            return $true
        }
    }

    # uv is not found anywhere. Prompt to install.
    if (-not $Yes) {
        # Check if host is interactive
        if ($Host.UI.RawUI) {
            $response = Read-Host "uv (modern Python package manager) was not found. Would you like to install it automatically to bootstrap the installation? [Y/n]"
            if ($response -match "^[Nn]") {
                return $false
            }
        } else {
            Warn "Non-interactive shell. Proceeding with uv installation..."
        }
    }

    Info "Installing uv..."
    $isWindows = $true
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        $isWindows = $IsWindows
    } elseif ($env:OS -match "Windows" -or $PSVersionTable.Platform -eq "Win32NT") {
        $isWindows = $true
    } else {
        $isWindows = $false
    }
    
    try {
        if ($isWindows) {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        } else {
            if (Get-Command curl -ErrorAction SilentlyContinue) {
                curl -LsSf https://astral.sh/uv/install.sh | sh
            } elseif (Get-Command wget -ErrorAction SilentlyContinue) {
                wget -qO- https://astral.sh/uv/install.sh | sh
            } else {
                Die "Neither curl nor wget found. Please install uv manually (https://astral.sh/uv)."
            }
        }
    } catch {
        Warn "Failed to install uv automatically: $_"
        return $false
    }

    # Reload PATH to find the newly installed uv
    foreach ($p in $UvPaths) {
        $uvExe = Join-Path $p "uv"
        $uvExeWin = Join-Path $p "uv.exe"
        if ((Test-Path $uvExe) -or (Test-Path $uvExeWin)) {
            $env:PATH = "$p$([System.IO.Path]::PathSeparator)$env:PATH"
            Ok "uv successfully installed and added to PATH."
            return $true
        }
    }

    return (Get-Command uv -ErrorAction SilentlyContinue) -ne $null
}

# Run Ensure-Uv to check or install uv
$HasUv = Ensure-Uv

# Python detection
function Detect-Python {
    # Try standard system python first
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

    # If system Python not found but uv is present, we can use uv
    if ($HasUv) {
        return "uv"
    }

    return $null
}

# 1. Detect Python
$PythonCmd = Detect-Python
if (-not $PythonCmd) {
    Die "Python 3.9+ required but not found. Please install Python first (or make sure it is in your PATH)."
}

if ($PythonCmd -eq "uv") {
    try {
        # uv run python will automatically download/use a managed Python version if none exists
        $PyVer = uv run python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"
        $PythonCmd = "uv run python"
    } catch {
        Die "uv was found but failed to run/download Python: $_"
    }
} else {
    $PyVer = & $PythonCmd -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"
}
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
Info "Extras: $(if ($ExtrasList.Count -gt 0) { $ExtrasList -join ',' } else { 'none' })"

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

# 7. Execute Installation (with requested extras, if any)
if ($UvTool) {
    Info "Installing package with uv tool"
    if (uv tool list 2>$null | Select-String "aidweather") {
        Info "Existing uv tool install found; reinstalling"
        uv tool install --reinstall $PkgSpec
    } else {
        uv tool install $PkgSpec
    }
} else {
    Info "Installing package"
    if ($Dev -or $ExtrasList.Count -gt 0) {
        Pip-Install @("-e", $PkgSpec)
    } else {
        Pip-Install @($PkgSpec)
    }
}

# 8. Smoke check
Info "Checking installation"
if ($UvTool) {
    if (Get-Command aidweather -ErrorAction SilentlyContinue) {
        # Active in PATH
    } else {
        Warn "Could not find 'aidweather' in your current PATH."
        Warn "You may need to run 'uv tool update-shell' and restart your terminal."
    }
} else {
    if ($PythonBin -eq "uv run python") {
        uv run python -c "import sys; import aidweather"
    } else {
        & $PythonBin -c "import sys; import aidweather"
    }
    if ($LASTEXITCODE -ne 0) {
        Die "Smoke test failed: Could not import aidweather."
    }
}

# 9. Optional: install the bundled Claude Code skill
if ($InstallSkill) {
    $SkillSrc = Join-Path $RepoRoot "src/aidweather/assets/skills/aidweather"
    if (Test-Path $SkillSrc) {
        $SkillDest = Join-Path $HOME ".claude/skills/aidweather"
        New-Item -ItemType Directory -Force -Path (Split-Path $SkillDest) | Out-Null
        if (Test-Path $SkillDest) {
            Remove-Item -Recurse -Force $SkillDest
        }
        Copy-Item -Recurse -Force $SkillSrc $SkillDest
        Ok "Claude Code skill installed to $SkillDest"
    } else {
        Warn "Skill source not found at $SkillSrc; skipping -InstallSkill."
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
