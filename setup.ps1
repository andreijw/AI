# ── RL CartPole Developer Setup Script (Windows Edition) ───────────────────
#
# This script automates environment verification and installation from scratch.
# It checks/installs Git and Python 3.12 (the version CI uses), creates the
# .venv virtual environment, installs the package with dev/video extras, and
# runs the same checks as CI (ruff, mypy, pytest).
#
# Usage:
#   .\setup.ps1 [-Update] [-NonInteractive] [-SkipChecks] [-InstallHooks]
#
# Parameters:
#   -Update          Upgrade Git/Python via winget and upgrade pip packages in .venv.
#   -NonInteractive  Run without prompting, automatically installing missing tools.
#   -SkipChecks      Skip the lint, type-check, and test run at the end.
#   -InstallHooks    Also run `pre-commit install` (hook versions in
#                    .pre-commit-config.yaml lag behind CI; see CLAUDE.md).
#

param (
    [switch]$Update,
    [switch]$NonInteractive,
    [switch]$SkipChecks,
    [switch]$InstallHooks
)

$ErrorActionPreference = "Stop"

$PythonVersion = "3.12"
$PythonWingetId = "Python.Python.3.12"
$RepoRoot = $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Ignore if not supported in the host environment
}

# ── Helper Functions ───────────────────────────────────────────────────────

function Show-Banner([string]$Title) {
    Write-Host ""
    Write-Host "=========================================================================" -ForegroundColor Cyan
    Write-Host ("  " + $Title) -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host "=========================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Confirm-Step([string]$Title, [string]$Question) {
    if ($NonInteractive) {
        return $true
    }
    $choices = [System.Management.Automation.Host.ChoiceDescription[]]@(
        [System.Management.Automation.Host.ChoiceDescription]::new("&Yes"),
        [System.Management.Automation.Host.ChoiceDescription]::new("&No")
    )
    return $Host.UI.PromptForChoice($Title, $Question, $choices, 0) -eq 0
}

function Get-WingetCommand {
    $cmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($cmd) {
        return "winget"
    }
    $appAlias = "$env:LocalAppData\Microsoft\WindowsApps\winget.exe"
    if (Test-Path $appAlias) {
        return $appAlias
    }
    return $null
}

function Update-EnvironmentPath {
    # Re-read PATH from the registry so freshly installed tools are visible
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $env:PATH = "$userPath;$machinePath"
}

function Install-WithWinget([string]$Name, [string]$Id, [string]$CustomArgs = "") {
    Show-Banner "INSTALLING $($Name.ToUpper())"
    $wingetCmd = Get-WingetCommand
    if (-not $wingetCmd) {
        Write-Host "[-] winget is not available; install $Name manually." -ForegroundColor Red
        return $false
    }
    Write-Host "[*] Executing winget to install $Name..." -ForegroundColor Cyan
    $wingetArgs = @("install", "--id", $Id, "--exact", "--scope", "user", "--silent",
        "--accept-source-agreements", "--accept-package-agreements")
    if ($CustomArgs) {
        $wingetArgs += @("--custom", $CustomArgs)
    }
    & $wingetCmd @wingetArgs | Out-Host
    Update-EnvironmentPath
    return $true
}

function Test-Git {
    Write-Host "[*] Checking for Git... " -NoNewline
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $gitVersion = & git --version
        Write-Host "FOUND ($($gitVersion.Trim()))" -ForegroundColor Green
        return $true
    }
    Write-Host "NOT FOUND" -ForegroundColor Red
    return $false
}

function Invoke-Probe([string]$Exe, [string[]]$ProbeArgs) {
    # Run a native probe command; return trimmed stdout, or $null on any failure.
    # (Windows PowerShell 5.1 throws on native stderr when ErrorActionPreference is Stop.)
    try {
        $output = & $Exe @ProbeArgs 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            return "$output".Trim()
        }
    } catch {
        # Treat as not found
    }
    return $null
}

function Find-Python {
    # 1. The py launcher knows every registered install
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $path = Invoke-Probe "py" @("-$PythonVersion", "-c", "import sys; print(sys.executable)")
        if ($path) {
            return $path
        }
    }

    # 2. Default winget/python.org install locations
    $tag = $PythonVersion.Replace(".", "")
    $candidates = @(
        "$env:LocalAppData\Programs\Python\Python$tag\python.exe",
        "$env:ProgramFiles\Python$tag\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    # 3. python on PATH, skipping the Microsoft Store stub in WindowsApps
    foreach ($cmd in Get-Command python, python3 -All -ErrorAction SilentlyContinue) {
        if ($cmd.Source -like "*\WindowsApps\*") {
            continue
        }
        $version = Invoke-Probe $cmd.Source @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")
        if ($version -eq $PythonVersion) {
            return $cmd.Source
        }
    }
    return $null
}

function Get-PythonInstall {
    Write-Host "[*] Checking for Python $PythonVersion... " -NoNewline
    $python = Find-Python
    if ($python) {
        Write-Host "FOUND ($python)" -ForegroundColor Green
        return $python
    }
    Write-Host "NOT FOUND" -ForegroundColor Red
    return $null
}

function Update-WingetPackage {
    $wingetCmd = Get-WingetCommand
    if (-not $wingetCmd) {
        Write-Host "[-] winget is not available. Cannot perform automatic updates." -ForegroundColor Red
        return
    }
    Show-Banner "UPDATING DEPENDENCIES VIA WINGET"
    foreach ($id in @("Git.Git", $PythonWingetId)) {
        Write-Host "[*] Checking updates for $id..." -ForegroundColor Cyan
        & $wingetCmd upgrade --id $id --exact --accept-source-agreements --accept-package-agreements | Out-Host
    }
    Update-EnvironmentPath
}

function Initialize-Venv([string]$Python) {
    Show-Banner "CREATING VIRTUAL ENVIRONMENT"

    if (Test-Path $VenvPython) {
        $venvVersion = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
        Write-Host "[+] Found existing .venv (Python $("$venvVersion".Trim()))." -ForegroundColor Green
        if ("$venvVersion".Trim() -ne $PythonVersion) {
            Write-Host "    -> WARNING: CI uses Python $PythonVersion. Delete .venv and re-run to match." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[*] Creating .venv with $Python..." -ForegroundColor Cyan
        & $Python -m venv $VenvDir | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[-] Failed to create .venv." -ForegroundColor Red
            return $false
        }
        Write-Host "[+] Created .venv." -ForegroundColor Green
    }

    Write-Host "[*] Upgrading pip..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip | Out-Host

    Write-Host "[*] Installing rl-cartpole in editable mode with [dev,video] extras..." -ForegroundColor Cyan
    $pipArgs = @("-m", "pip", "install", "-e", ".[dev,video]")
    if ($Update) {
        $pipArgs += "--upgrade"
    }
    & $VenvPython @pipArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[-] pip install failed. See errors above." -ForegroundColor Red
        return $false
    }
    Write-Host "[+] Dependencies installed." -ForegroundColor Green

    if ($InstallHooks) {
        Write-Host "[*] Installing pre-commit hooks..." -ForegroundColor Cyan
        & $VenvPython -m pre_commit install | Out-Host
    }
    return $true
}

function Invoke-CiCheck {
    Show-Banner "RUNNING CI CHECKS"
    $checks = @(
        @{ Name = "ruff check"; Args = @("-m", "ruff", "check", ".") },
        @{ Name = "ruff format --check"; Args = @("-m", "ruff", "format", "--check", ".") },
        @{ Name = "mypy src"; Args = @("-m", "mypy", "src") },
        @{ Name = "pytest"; Args = @("-m", "pytest", "-q") }
    )

    $failed = @()
    foreach ($check in $checks) {
        Write-Host "[*] Running $($check.Name)..." -ForegroundColor Cyan
        $checkArgs = $check.Args
        & $VenvPython @checkArgs | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[+] $($check.Name) passed." -ForegroundColor Green
        } else {
            Write-Host "[-] $($check.Name) failed." -ForegroundColor Red
            $failed += $check.Name
        }
    }
    return ,$failed
}

# ── Main Execution Flow ────────────────────────────────────────────────────

Push-Location $RepoRoot
try {
    Update-EnvironmentPath
    Show-Banner "RL CARTPOLE - WINDOWS DEVELOPER ENVIRONMENT SETUP"

    if ($Update) {
        Update-WingetPackage
    }

    if (-not (Test-Git)) {
        if (Confirm-Step "Install Missing Dependency" "Git was not found. Install it now via winget?") {
            Install-WithWinget "Git" "Git.Git" | Out-Null
        }
        if (-not (Test-Git)) {
            Write-Host "[-] Git is required to work on this repository. Exiting." -ForegroundColor Red
            exit 1
        }
    }

    $python = Get-PythonInstall
    if (-not $python) {
        if (Confirm-Step "Install Missing Dependency" "Python $PythonVersion was not found. Install it now via winget?") {
            # A per-user py launcher keeps the whole install free of UAC elevation prompts
            Install-WithWinget "Python $PythonVersion" $PythonWingetId "InstallLauncherAllUsers=0" | Out-Null
            $python = Get-PythonInstall
        }
        if (-not $python) {
            Write-Host "[-] Python $PythonVersion is required. Install it from https://www.python.org/downloads/" -ForegroundColor Red
            Write-Host "    -> Then restart your terminal and run setup again." -ForegroundColor Yellow
            exit 1
        }
    }

    if (-not (Initialize-Venv $python)) {
        exit 1
    }

    $failedChecks = @()
    if ($SkipChecks) {
        Write-Host "[*] Skipping CI checks as requested." -ForegroundColor Yellow
    } else {
        $failedChecks = Invoke-CiCheck
    }

    Show-Banner "SETUP SUMMARY"
    Write-Host "  [YES] Git is installed." -ForegroundColor Green
    Write-Host "  [YES] Python $PythonVersion found: $python" -ForegroundColor Green
    Write-Host "  [YES] .venv is ready: $VenvDir" -ForegroundColor Green
    if ($SkipChecks) {
        Write-Host "  [SKIP] CI checks were not run." -ForegroundColor Yellow
    } elseif ($failedChecks.Count -eq 0) {
        Write-Host "  [YES] All CI checks passed (ruff, mypy, pytest)." -ForegroundColor Green
    } else {
        Write-Host "  [NO]  Failed checks: $($failedChecks -join ', ')" -ForegroundColor Red
    }

    $context7Key = $env:CONTEXT7_API_KEY
    if (-not $context7Key) {
        $context7Key = [Environment]::GetEnvironmentVariable("CONTEXT7_API_KEY", "User")
    }
    if (-not $context7Key) {
        Write-Host "  [WARN] CONTEXT7_API_KEY is not set; the context7 MCP server will not work." -ForegroundColor Yellow
        Write-Host "         See README -> AI Assistant MCP Servers." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "[*] Activate the environment with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""

    if ($failedChecks.Count -gt 0) {
        exit 1
    }
} finally {
    Pop-Location
}
