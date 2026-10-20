<# : BIDS Manager installer for Windows. Double-click this file.
@echo off
setlocal
rem These lines only start the PowerShell script below them, in this same
rem file, without changing the execution policy and without administrator
rem rights. Everything the installer does is in the PowerShell part.
set "BIDSMGR_SELF=%~f0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$s = [IO.File]::ReadAllText($env:BIDSMGR_SELF); Invoke-Expression $s"
set "BIDSMGR_RC=%ERRORLEVEL%"
echo.
if not "%BIDSMGR_YES%"=="1" pause
exit /b %BIDSMGR_RC%
#>
# =======================================================================
# BIDS Manager installer for Windows 10 and 11 (x64), and Windows 11 on
# ARM through its x64 emulation
# Installer version 2.0.3. History: Installers/CHANGELOG.md
# Licence: MIT, ANCP Lab, University of Oldenburg
# https://github.com/ANCPLabOldenburg/BIDS-Manager
#
# Without administrator rights, and entirely inside %USERPROFILE%\BIDS-Manager,
# it installs:
#
#   uv\        a pinned copy of uv, checked against its published SHA-256.
#              A uv you already have is never used, nor its settings.
#   python\    the Python version below, downloaded by that uv
#   env\       an environment with bids-manager from PyPI
#
# and adds shortcuts to the Desktop and the Start menu. Running it again
# updates BIDS Manager to the newest release, or uninstalls it.
#
# Windows on ARM uses the x64 build: dcm2niix, which converts MRI, has no
# ARM64 build for Windows, and Windows 11 runs x64 programs on ARM.
#
# Optional settings, for testing and for administrators:
#   BIDSMGR_INSTALL_DIR     where to install (default %USERPROFILE%\BIDS-Manager)
#   BIDSMGR_PYTHON          Python version (default 3.10)
#   BIDSMGR_PACKAGE         what to install: "bids-manager==1.5.0", a wheel file...
#   BIDSMGR_YES=1           take the default answer to every question
#   BIDSMGR_NO_SHORTCUTS=1  no shortcuts on the Desktop or in the Start menu
#   BIDSMGR_UNINSTALL=1     remove BIDS Manager, without asking
# =======================================================================
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$InstallerVersion = '2.0.3'
$UvVersion = '0.13.1'
$UvTarget = 'x86_64-pc-windows-msvc'
$UvSha256 = '3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672'

function Setting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($value)) { return $Default }
    return $value
}

# 3.10 is what installer 1.x used and what every version pin was tested on.
# A newer Python brings newer libraries (brainchop gets tinygrad 0.12 or
# newer on 3.11 and later, untested here). The update button moves an
# installation to a newer Python when a release needs one.
$PythonVersion = Setting 'BIDSMGR_PYTHON' '3.10'
$Package = Setting 'BIDSMGR_PACKAGE' 'bids-manager'
$InstallDir = Setting 'BIDSMGR_INSTALL_DIR' (Join-Path $env:USERPROFILE 'BIDS-Manager')
$AssumeYes = (Setting 'BIDSMGR_YES' '0') -eq '1'
$NoShortcuts = (Setting 'BIDSMGR_NO_SHORTCUTS' '0') -eq '1'

# The folder is removed whole on reinstall and uninstall, so it must be one
# of its own: never relative, the user folder or a drive.
if (-not [IO.Path]::IsPathRooted($InstallDir)) {
    Write-Host 'BIDSMGR_INSTALL_DIR must be an absolute path to a folder of its own.'; exit 1
}
$InstallDir = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
if ($InstallDir -eq [IO.Path]::GetPathRoot($InstallDir).TrimEnd('\') -or
    $InstallDir -eq $env:USERPROFILE.TrimEnd('\')) {
    Write-Host 'BIDS Manager needs a folder of its own.'; exit 1
}

$UvDir = Join-Path $InstallDir 'uv'
$Uv = Join-Path $UvDir 'uv.exe'
$EnvDir = Join-Path $InstallDir 'env'
$EnvPy = Join-Path $EnvDir 'Scripts\python.exe'
$Manifest = Join-Path $InstallDir 'installation.json'
$LogFile = Join-Path $InstallDir 'install.log'
$IcoFile = Join-Path $InstallDir 'bids-manager.ico'
$Desktop = [Environment]::GetFolderPath('Desktop')
$StartMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'BIDS-Manager'
$MinDiskMB = 2500
$Utf8 = New-Object Text.UTF8Encoding $false

# -- Output ---------------------------------------------------------------
function Log([string]$Text) {
    if (Test-Path -LiteralPath $InstallDir) {
        [IO.File]::AppendAllText($LogFile, $Text + "`r`n", $Utf8)
    }
}
function Info([string]$Text) { Write-Host "[ok] $Text" -ForegroundColor Green; Log "[ok] $Text" }
function Step([string]$Text) { Write-Host "[..] $Text" -ForegroundColor Cyan; Log "[..] $Text" }
function Warn([string]$Text) { Write-Host "[!] $Text" -ForegroundColor Yellow; Log "[!] $Text" }
$Tmp = ''
function Die([string]$Text) {
    Write-Host "[x] $Text" -ForegroundColor Red
    Log "[x] $Text"
    Write-Host "    Log file: $LogFile"
    if ($Tmp) { Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue }
    exit 1
}
# Anything unexpected ends the installation with one readable line.
trap { Die "Unexpected error: $($_.Exception.Message)" }

# Ask QUESTION DEFAULT: the answer, or DEFAULT when nobody can answer.
function Ask([string]$Question, [string]$Default) {
    if ($AssumeYes -or [Console]::IsInputRedirected) { return $Default }
    $reply = Read-Host $Question
    if ([string]::IsNullOrWhiteSpace($reply)) { return $Default }
    return $reply.Trim()
}

# Run a program, show and log its output, return its exit code. Windows
# PowerShell turns a program's stderr into errors, and uv reports progress
# on stderr, so errors do not stop the script while it runs.
function Invoke-Program([string]$Exe, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Log $line
        }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

# Once as is, then once more with the system's certificates, which networks
# that inspect HTTPS (many universities) need.
function Invoke-Uv([string[]]$Arguments) {
    if ((Invoke-Program $Uv $Arguments) -eq 0) { return $true }
    Warn "Trying again with the system's certificates..."
    $env:UV_SYSTEM_CERTS = '1'
    try {
        return ((Invoke-Program $Uv $Arguments) -eq 0)
    } finally {
        Remove-Item Env:UV_SYSTEM_CERTS -ErrorAction SilentlyContinue
    }
}

# The output of a short Python command, or '' when it fails. Windows
# PowerShell 5.1 drops double quotes inside arguments it passes to a
# program, so the Python code uses single quotes only. Every call to the
# environment's Python uses -I: without it, Python looks in the current
# folder first, and run from inside a BIDS Manager checkout it would read
# that copy instead of the one just installed.
function Ask-Python([string]$Code) {
    if (-not (Test-Path -LiteralPath $EnvPy)) { return '' }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $EnvPy -I -c $Code 2>$null
        if ($LASTEXITCODE -ne 0) { return '' }
        return ("$out").Trim()
    } finally {
        $ErrorActionPreference = $previous
    }
}

# -- Isolation --------------------------------------------------------------
# Only this installer's uv, with only this installer's settings: variables
# set for another uv (an index, a cache, a Python preference) and any active
# environment must not reach it. These changes last only while it runs.
foreach ($variable in @(Get-ChildItem Env: | Where-Object { $_.Name -like 'UV_*' })) {
    Remove-Item -LiteralPath ('Env:' + $variable.Name)
}
foreach ($name in 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONHOME', 'PYTHONPATH') {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:UV_NO_CONFIG = '1'
$env:UV_CACHE_DIR = Join-Path $InstallDir 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'
$env:UV_MANAGED_PYTHON = '1'
$env:UV_PYTHON_DOWNLOADS = 'automatic'

# -- Platform ---------------------------------------------------------------
# The machine's architecture, not this PowerShell's: an x64 PowerShell on an
# ARM computer reports AMD64 in its own environment.
$arch = ''
try {
    $arch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment').PROCESSOR_ARCHITECTURE
} catch { }
if (-not $arch) {
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 } else { $arch = $env:PROCESSOR_ARCHITECTURE }
}
switch ($arch.ToUpper()) {
    'AMD64' { $Platform = 'Windows (x64)' }
    'ARM64' {
        if ([Environment]::OSVersion.Version.Build -lt 22000) {
            Write-Host 'BIDS Manager on an ARM computer needs Windows 11, which runs x64 programs.'
            Write-Host 'Windows 10 on ARM cannot.'
            exit 1
        }
        $Platform = 'Windows on ARM (x64 build, through Windows emulation)'
    }
    default {
        Write-Host "This Windows ($arch) is not supported: BIDS Manager needs 64-bit Windows."
        exit 1
    }
}
$PythonRequest = "cpython-$PythonVersion-windows-x86_64-none"

# -- Banner -----------------------------------------------------------------
try { [Console]::Clear() } catch { }
Write-Host ''
Write-Host '    +--------------------------------------------------+'
Write-Host '    |                                                  |'
Write-Host '    |          B I D S   M a n a g e r                 |'
Write-Host '    |                                                  |'
Write-Host '    |    Schema-driven BIDS converter and editor       |'
Write-Host '    |        ANCP Lab, University of Oldenburg         |'
Write-Host '    |                                                  |'
Write-Host '    +--------------------------------------------------+'
Write-Host ''
Write-Host "  Installer version:  $InstallerVersion"
Write-Host "  Platform:           $Platform"
Write-Host "  Python:             $PythonVersion"
Write-Host "  Install folder:     $InstallDir"
Write-Host ''

# -- Before anything is written ------------------------------------------------
try {
    $root = [IO.Path]::GetPathRoot($InstallDir)
    $free = (New-Object IO.DriveInfo $root).AvailableFreeSpace / 1MB
    if ($free -lt $MinDiskMB) {
        Warn ("Only {0:N0} MB free; about $MinDiskMB MB are needed." -f $free)
    }
} catch { }
try {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://pypi.org/simple/pip/' -Method Head -TimeoutSec 8 | Out-Null
} catch {
    Warn 'pypi.org cannot be reached. Check the internet connection.'
}

$Mode = 'install'
$UninstallNow = (Setting 'BIDSMGR_UNINSTALL' '0') -eq '1'
if ($UninstallNow) {
    $Mode = 'uninstall'
} elseif ((Test-Path -LiteralPath $Manifest) -or (Test-Path -LiteralPath $EnvDir)) {
    Write-Host '  BIDS Manager is already installed in:' -ForegroundColor Yellow
    Write-Host "    $InstallDir"
    Write-Host ''
    Write-Host '  [u] Update: install the newest BIDS Manager (default)'
    Write-Host '  [r] Reinstall: remove this installation and start again'
    Write-Host '  [x] Uninstall: remove BIDS Manager from this computer'
    Write-Host '  [a] Abort'
    Write-Host ''
    while ($true) {
        $choice = Ask '  Your choice [U/r/x/a]' 'u'
        if ($choice -match '^[uU]') { $Mode = 'update'; break }
        if ($choice -match '^[rR]') { $Mode = 'reinstall'; break }
        if ($choice -match '^[xX]') { $Mode = 'uninstall'; break }
        if ($choice -match '^[aA]') { Write-Host '  Nothing was changed.'; exit 0 }
        Write-Host '  Please type u, r, x or a.'
    }
} else {
    Write-Host '  The installer will:'
    Write-Host "    1. Download uv $UvVersion (about 20 MB) into the install folder"
    Write-Host "    2. Download Python $PythonVersion with it (about 30 MB)"
    Write-Host '    3. Install bids-manager and its dependencies (about 1.5 GB)'
    Write-Host '    4. Add shortcuts to the Desktop and the Start menu'
    Write-Host '    No administrator rights are needed.'
    Write-Host ''
    $confirm = Ask '  Continue? [Y/n]' 'y'
    if ($confirm -match '^[nN]') { Write-Host '  Nothing was changed.'; exit 0 }
}

# A running BIDS Manager holds its files open, and Windows cannot replace or
# remove a file that is open.
if ($Mode -ne 'install') {
    while ($true) {
        $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
            try { $_.Path -and $_.Path.StartsWith($InstallDir + '\', [StringComparison]::OrdinalIgnoreCase) } catch { $false }
        })
        if ($running.Count -eq 0) { break }
        if ($AssumeYes -or $UninstallNow -or [Console]::IsInputRedirected) {
            Write-Host '  BIDS Manager is running. Close it, then run the installer again.'; exit 1
        }
        Read-Host '  BIDS Manager is running. Close it, then press Enter' | Out-Null
    }
}

# Uninstall: everything this installer (or installer 1.x) put on the
# computer. Datasets and BIDS Manager's settings are not touched.
if ($Mode -eq 'uninstall') {
    Write-Host ''
    Write-Host '  This removes BIDS Manager completely:'
    Write-Host "    $InstallDir"
    Write-Host '    and its shortcuts on the Desktop and in the Start menu'
    Write-Host '  Your datasets and your settings are not touched.'
    if (-not $UninstallNow) {
        $confirm = Ask '  Remove it? [y/N]' 'n'
        if ($confirm -notmatch '^[yY]') { Write-Host '  Nothing was removed.'; exit 0 }
    }
    foreach ($link in @((Join-Path $Desktop 'BIDS-Manager.lnk'), (Join-Path $Desktop 'Uninstall BIDS-Manager.lnk'))) {
        Remove-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $StartMenu -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $InstallDir) {
        try { Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction Stop } catch { }
    }
    if (Test-Path -LiteralPath $InstallDir) {
        # Very deep paths defeat Remove-Item on Windows PowerShell 5.1; rd does not.
        # The path is passed as it is: PowerShell quotes it when it has spaces.
        & cmd.exe /c rd /s /q $InstallDir 2>$null
    }
    try { & ie4uinit.exe -show 2>$null } catch { }
    if (Test-Path -LiteralPath $InstallDir) {
        Write-Host "[x] Some files could not be removed from $InstallDir. Close any program using them and run the installer again." -ForegroundColor Red
        exit 1
    }
    Write-Host '[ok] BIDS Manager has been removed.' -ForegroundColor Green
    exit 0
}

if ($Mode -eq 'reinstall') {
    Step 'Removing the previous installation...'
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
[IO.File]::WriteAllText($LogFile, '', $Utf8)
Log "BIDS Manager installer $InstallerVersion ($Mode) started $(Get-Date -Format s)"
Log "Platform $Platform, Python $PythonRequest, package $Package"

$Tmp = Join-Path ([IO.Path]::GetTempPath()) ('bidsmgr-install-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

# =======================================================================
# Step 1/5: uv
# =======================================================================
$haveUv = ''
if (Test-Path -LiteralPath $Uv) {
    try { $haveUv = ((& $Uv --version) -split ' ')[1] } catch { }
}
if ($haveUv -eq $UvVersion) {
    Info "Step 1/5: uv $UvVersion is already here."
} else {
    Step "Step 1/5: Downloading uv $UvVersion..."
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-$UvTarget.zip"
    $zip = Join-Path $Tmp 'uv.zip'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
    } catch {
        Die "uv could not be downloaded from $url"
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLower() -ne $UvSha256) {
        Die 'The uv download does not match its published checksum, so it was not used.'
    }
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $Tmp 'uv') -Force
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $Tmp 'uv\uv.exe') -Destination $Uv -Force
    if ((Invoke-Program $Uv @('--version')) -ne 0) { Die 'uv does not run on this computer.' }
    Info "uv $UvVersion installed in $UvDir"
}

# =======================================================================
# Step 2/5: Python
# =======================================================================
Step "Step 2/5: Python $PythonVersion..."
# --no-registry: nothing is written to the registry, so the Python stays
# this installation's own and other programs never list it.
if (-not (Invoke-Uv @('python', 'install', $PythonRequest, '--no-bin', '--no-registry'))) {
    Die "Python $PythonVersion could not be installed."
}
Info 'Python ready.'

# =======================================================================
# Step 3/5: the environment
# =======================================================================
# Kept on update when it runs the requested Python for this build, or a
# newer one the update button moved it to because a release needed it (the
# record says so: moving it back would install an older BIDS Manager);
# built again otherwise (an installation from installer 1.x, another
# Python, a broken environment).
$envPython = Ask-Python 'import sys; print(''%d.%d'' % sys.version_info[:2])'
$sameBuild = $false
$moved = $false
if (Test-Path -LiteralPath $Manifest) {
    try {
        $priorRecord = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
        $sameBuild = ($priorRecord.installer.target -eq $UvTarget)
        $moved = ($priorRecord.python.moved_by_update -eq $true)
    } catch { }
}
$keep = $false
$keepMove = $false
if ($Mode -eq 'update' -and $sameBuild -and $envPython) {
    try {
        $wanted = [version]((($PythonVersion -split '\.')[0..1]) -join '.')
        $have = [version]$envPython
        $keep = ($have -eq $wanted) -or ($moved -and $have -gt $wanted)
        $keepMove = $keep -and ($have -ne $wanted)
    } catch { }
}
if ($keep) {
    Info "Step 3/5: The environment is kept (Python $envPython)."
} else {
    Step 'Step 3/5: Creating the environment...'
    # Installer 1.x kept an embeddable Python 3.10 here; nothing uses it any more.
    $old = Join-Path $InstallDir 'python310'
    if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
    # --seed adds pip, which the in-app updater of BIDS Manager 1.5 uses.
    if ((Invoke-Program $Uv @('venv', '--clear', '--seed', '--prompt', 'BIDS-Manager', '--python', $PythonRequest, $EnvDir)) -ne 0) {
        Die 'The environment could not be created.'
    }
    Info "Environment ready in $EnvDir"
}

# Pythons no environment uses any more (an earlier default, a move made by
# the update button), about 70 MB each.
$inUse = Ask-Python 'import sys; print(''%d.%d'' % sys.version_info[:2])'
if ($inUse) {
    foreach ($dir in @(Get-ChildItem -LiteralPath $env:UV_PYTHON_INSTALL_DIR -Directory -ErrorAction SilentlyContinue)) {
        if ($dir.Name -match '^cpython-(\d+\.\d+)\.\d+[^-]*-(.+)$' -and $Matches[1] -ne $inUse) {
            $minor = $Matches[1]
            if ((Invoke-Program $Uv @('python', 'uninstall', "cpython-$minor-$($Matches[2])")) -eq 0) {
                Info "Removed Python $minor, which nothing uses any more."
            }
        }
    }
}

# =======================================================================
# Step 4/5: BIDS Manager
# =======================================================================
Step "Step 4/5: Installing $Package (a few minutes the first time)..."
# --upgrade: an update installs the newest release, not the one already here.
# --compile-bytecode: what pip does, and what keeps the first start quick.
if (-not (Invoke-Uv @('pip', 'install', '--python', $EnvPy, '--upgrade', '--compile-bytecode', $Package))) {
    Die 'bids-manager could not be installed.'
}
if ((Invoke-Program $EnvPy @('-I', '-c', 'import bidsmgr, PyQt6.QtWidgets')) -ne 0) {
    Die 'bids-manager was installed but does not start.'
}
$BmVersion = Ask-Python 'from importlib.metadata import version; print(version(''bids-manager''))'
Invoke-Program $Uv @('cache', 'prune') | Out-Null
Info "BIDS Manager $BmVersion installed."

# The record of this installation. BIDS Manager reads it to find the uv and
# the Python it was installed with.
$env:BM_MANIFEST = $Manifest; $env:BM_INSTALLER = $InstallerVersion; $env:BM_TARGET = $UvTarget
$env:BM_UV = $Uv; $env:BM_UV_VERSION = $UvVersion; $env:BM_PYTHON = $PythonVersion
$env:BM_ENV = $EnvDir; $env:BM_PACKAGE = $Package; $env:BM_VERSION = $BmVersion
if ($keepMove) { $env:BM_KEEP_MOVE = '1' } else { $env:BM_KEEP_MOVE = '0' }
$writer = Join-Path $Tmp 'manifest.py'
[IO.File]::WriteAllText($writer, @'
import datetime, json, os, platform, sys
e = os.environ
path = e["BM_MANIFEST"]
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
try:
    with open(path, encoding="utf-8") as fh:
        previous = json.load(fh)
except (OSError, ValueError):
    previous = {}
first = previous.get("installed", now)
record = {
    "format": 1,
    "installed": first,
    "updated": now,
    "installer": {"version": e["BM_INSTALLER"], "os": "windows", "target": e["BM_TARGET"]},
    "uv": {"version": e["BM_UV_VERSION"], "executable": e["BM_UV"]},
    "uv_environment": {k: e[k] for k in (
        "UV_NO_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_MANAGED_PYTHON")},
    "python": {"requested": e["BM_PYTHON"], "version": platform.python_version(),
               "executable": sys.executable},
    "environment": e["BM_ENV"],
    "package": {"name": "bids-manager", "requested": e["BM_PACKAGE"],
                "version": e["BM_VERSION"]},
}
if e.get("BM_KEEP_MOVE") == "1":
    # A Python the update button moved to stays recorded as such.
    record["python"]["requested"] = previous["python"]["requested"]
    record["python"]["moved_by_update"] = True
with open(path, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2)
    fh.write("\n")
'@, $Utf8)
if ((Invoke-Program $EnvPy @('-I', $writer)) -ne 0) { Warn 'installation.json could not be written.' }

# EyeLink .edf files are read through a library built against the Visual
# C++ 2008 runtime, which Windows does not include. Installing it needs an
# administrator, so the installer says so and leaves it to the user.
if ((Invoke-Program $EnvPy @('-I', '-c', 'from eyelinkio.edf import _edf2py')) -ne 0) {
    Warn 'EyeLink .edf files cannot be read on this computer yet. They need the'
    Write-Host '    "Microsoft Visual C++ 2008 SP1 Redistributable (x64)", which needs an'
    Write-Host '    administrator to install. Or export the recordings as .asc.'
    Write-Host '    Everything else works.'
}

# =======================================================================
# Step 5/5: launcher, uninstaller and shortcuts
# =======================================================================
# Batch files read their text in the console's code page; chcp 65001 first
# makes a user folder with accented letters read correctly. The launcher
# puts the environment's Scripts folder first on PATH, as activating it
# would, so the programs installed with BIDS Manager are the ones found.
function Batch-Path([string]$Path) { return $Path.Replace('%', '%%') }
$RunBat = Join-Path $InstallDir 'run_BIDS_Manager.bat'
$UninstallBat = Join-Path $InstallDir 'uninstall_BIDS_Manager.bat'

[IO.File]::WriteAllText($RunBat, (@(
    '@echo off'
    'chcp 65001 >nul'
    'title BIDS Manager'
    ('set "PATH=' + (Batch-Path (Join-Path $EnvDir 'Scripts')) + ';%PATH%"')
    ('"' + (Batch-Path (Join-Path $EnvDir 'Scripts\bidsmgr.exe')) + '" %*')
    'if errorlevel 1 pause'
) -join "`r`n") + "`r`n", $Utf8)

[IO.File]::WriteAllText($UninstallBat, (@(
    '@echo off'
    'chcp 65001 >nul'
    'echo.'
    'echo   BIDS Manager uninstaller'
    'echo.'
    'echo   This removes BIDS Manager completely:'
    ('echo     "' + (Batch-Path $InstallDir) + '"')
    'echo   Your datasets and your settings are not touched.'
    'echo.'
    'set /p "CONFIRM=  Remove it? [y/N] "'
    'if /i not "%CONFIRM%"=="y" goto :keep'
    ('del /q "' + (Batch-Path (Join-Path $Desktop 'BIDS-Manager.lnk')) + '" 2>nul')
    ('del /q "' + (Batch-Path (Join-Path $Desktop 'Uninstall BIDS-Manager.lnk')) + '" 2>nul')
    ('rd /s /q "' + (Batch-Path $StartMenu) + '" 2>nul')
    'cd /d "%TEMP%"'
    'echo   BIDS Manager has been removed.'
    'pause'
    ('(goto) 2>nul & rd /s /q "' + (Batch-Path $InstallDir) + '"')
    ':keep'
    'echo   Nothing was removed.'
    'pause'
) -join "`r`n") + "`r`n", $Utf8)

if ($NoShortcuts) {
    Info 'Step 5/5: No shortcuts (BIDSMGR_NO_SHORTCUTS=1).'
} else {
    Step 'Step 5/5: Adding shortcuts to the Desktop and the Start menu...'
    $pkgDir = Ask-Python 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(''bidsmgr'').origin))'
    $icon = ''
    if ($pkgDir -and (Test-Path -LiteralPath (Join-Path $pkgDir 'gui\assets\windows\AppIcon.ico'))) {
        Copy-Item -LiteralPath (Join-Path $pkgDir 'gui\assets\windows\AppIcon.ico') -Destination $IcoFile -Force
        $icon = $IcoFile
    } else {
        Warn 'The app icon was not found; the shortcuts show a default icon.'
    }
    try {
        $shell = New-Object -ComObject WScript.Shell
        New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null
        $links = @(
            @((Join-Path $Desktop 'BIDS-Manager.lnk'), $RunBat, 'Start BIDS Manager', $icon),
            @((Join-Path $Desktop 'Uninstall BIDS-Manager.lnk'), $UninstallBat, 'Remove BIDS Manager', ''),
            @((Join-Path $StartMenu 'BIDS-Manager.lnk'), $RunBat, 'Start BIDS Manager', $icon),
            @((Join-Path $StartMenu 'Uninstall BIDS-Manager.lnk'), $UninstallBat, 'Remove BIDS Manager', '')
        )
        foreach ($link in $links) {
            $lnk = $shell.CreateShortcut($link[0])
            $lnk.TargetPath = $link[1]
            $lnk.WorkingDirectory = $InstallDir
            $lnk.Description = $link[2]
            if ($link[3]) { $lnk.IconLocation = $link[3] + ',0' }
            $lnk.Save()
        }
        try { & ie4uinit.exe -show 2>$null } catch { }
        Info 'Shortcuts are on the Desktop and in the Start menu.'
    } catch {
        Warn "The shortcuts could not be created ($($_.Exception.Message))."
        Write-Host "    Start BIDS Manager with: $RunBat"
    }
}

Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue

# =======================================================================
# Done
# =======================================================================
Write-Host ''
Write-Host '======================================================='
Write-Host "  BIDS Manager $BmVersion is installed." -ForegroundColor Green
Write-Host '======================================================='
Write-Host ''
Write-Host '  Start it:   BIDS-Manager on the Desktop or in the Start menu.'
Write-Host '              A console window shows what it is doing.'
Write-Host '  Update:     run this installer again'
Write-Host '  Uninstall:  Uninstall BIDS-Manager on the Desktop'
Write-Host "  Log:        $LogFile"
Write-Host ''
exit 0
