[CmdletBinding()]
param(
    [int]$Port = 8787,
    [int]$HealthTimeoutSeconds = 2,
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Port -ne 8787) { throw "This launcher is fixed to the loopback Generation Backend port 8787." }
$BackendUrl = "http://127.0.0.1:$Port"
$RuntimeDirectory = Join-Path $ProjectRoot ".generation_backend"
$LogDirectory = Join-Path $RuntimeDirectory "logs"
$PidFile = Join-Path $RuntimeDirectory "backend.pid.json"
$LifecycleFile = Join-Path $RuntimeDirectory "lifecycle.json"
$MutexName = "Local\CodexGenerationBackendLauncher"

function Write-Lifecycle {
    param(
        [hashtable]$Values
    )

    New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
    $Values | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LifecycleFile -Encoding UTF8
}

function Get-ListeningPids {
    try {
        $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
        return @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    } catch {
        $lines = @(netstat -ano -p tcp 2>$null | Select-String (":$Port\s+.*LISTENING\s+(\d+)$"))
        $pids = foreach ($line in $lines) {
            if ($line.Matches.Count -gt 0) { [int]$line.Matches[0].Groups[1].Value }
        }
        return @($pids | Select-Object -Unique)
    }
}

function Get-BackendHealth {
    try {
        return Invoke-RestMethod -UseBasicParsing -Uri "$BackendUrl/api/generation/health" -TimeoutSec $HealthTimeoutSeconds
    } catch {
        return $null
    }
}

function Test-BackendReady {
    $health = Get-BackendHealth
    if ($null -eq $health) { return $false }
    return (
        $health.service_ready -eq $true -and
        $health.lm_studio_configured -eq $true -and
        $health.comfyui_url_configured -eq $true
    )
}

function Find-BackendPython {
    $candidates = New-Object System.Collections.Generic.List[string]
    $configured = [Environment]::GetEnvironmentVariable("GENERATION_BACKEND_PYTHON", "Process")
    if (-not [string]::IsNullOrWhiteSpace($configured)) { [void]$candidates.Add($configured) }

    $localVenv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localVenv) { [void]$candidates.Add($localVenv) }

    # Discover the existing StabilityMatrix Python runtime without embedding
    # the user's localized Desktop subdirectory name in this script.
    $desktop = Join-Path $env:USERPROFILE "Desktop"
    $desktopDirectories = @(Get-ChildItem -LiteralPath $desktop -Directory -ErrorAction SilentlyContinue)
    foreach ($desktopDirectory in $desktopDirectories) {
        $stabilityPython = Join-Path $desktopDirectory.FullName "StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $stabilityPython) { [void]$candidates.Add($stabilityPython) }
    }

    $pathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pathPython) { [void]$candidates.Add($pathPython.Source) }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            & $candidate -c "import aiohttp, dotenv" 1>$null 2>$null
            if ($LASTEXITCODE -eq 0) { return (Resolve-Path -LiteralPath $candidate).Path }
        } catch {
            continue
        }
    }
    throw "A Python runtime with aiohttp and python-dotenv was not found. Set GENERATION_BACKEND_PYTHON or install the project runtime."
}

function Rotate-BackendLogs {
    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    $oldLogs = @(Get-ChildItem -LiteralPath $LogDirectory -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -Skip 14)
    foreach ($oldLog in $oldLogs) {
        Remove-Item -LiteralPath $oldLog.FullName -Force -ErrorAction SilentlyContinue
    }
}

function Wait-BackendReady {
    param([int]$TimeoutSeconds)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-BackendReady) { return $true }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    return $false
}

$mutex = [System.Threading.Mutex]::new($false, $MutexName)
$mutexAcquired = $false
$startedAt = [DateTime]::UtcNow
$stdoutLog = $null
$stderrLog = $null

try {
    try {
        $mutexAcquired = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $mutexAcquired = $true
    }

    if (-not $mutexAcquired) {
        if (Wait-BackendReady -TimeoutSeconds $StartupTimeoutSeconds) { exit 0 }
        throw "Another Backend launcher is active, but the Backend did not become healthy. No duplicate process was started."
    }

    if (Test-BackendReady) {
        $existingPids = @(Get-ListeningPids)
        $previousRecord = $null
        if (Test-Path -LiteralPath $PidFile) {
            try { $previousRecord = Get-Content -LiteralPath $PidFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $previousRecord = $null }
        }
        Write-Lifecycle @{
            event = "already_healthy"
            status = "healthy"
            launch_method = "health_preflight"
            backend_pid = if ($existingPids.Count -gt 0) { $existingPids[0] } else { $null }
            launcher_pid = $PID
            started_at_utc = $startedAt.ToString("o")
            last_health_at_utc = [DateTime]::UtcNow.ToString("o")
            exit_code = $null
            host = "127.0.0.1"
            port = $Port
            working_directory = $ProjectRoot
            process_path = if ($null -ne $previousRecord) { $previousRecord.process_path } else { $null }
            stdout_log = if ($null -ne $previousRecord) { $previousRecord.stdout_log } else { $null }
            stderr_log = if ($null -ne $previousRecord) { $previousRecord.stderr_log } else { $null }
        }
        Remove-Item -LiteralPath (Join-Path $RuntimeDirectory "last_launch_error.txt") -Force -ErrorAction SilentlyContinue
        exit 0
    }

    $occupiedPids = @(Get-ListeningPids)
    if ($occupiedPids.Count -gt 0) {
        Write-Lifecycle @{
            event = "start_rejected_port_occupied"
            status = "error"
            launch_method = "launch_generation_backend.ps1"
            backend_pid = $occupiedPids[0]
            launcher_pid = $PID
            started_at_utc = $startedAt.ToString("o")
            last_health_at_utc = $null
            exit_code = $null
            host = "127.0.0.1"
            port = $Port
            working_directory = $ProjectRoot
        }
        throw "Port $Port is already occupied, but the Generation Backend health check failed. No duplicate process was started."
    }

    $python = Find-BackendPython
    Rotate-BackendLogs
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutLog = Join-Path $LogDirectory "backend-$stamp.stdout.log"
    $stderrLog = Join-Path $LogDirectory "backend-$stamp.stderr.log"
    # generation_backend_server.py owns dotenv loading and defaults to the
    # loopback 8787 contract. Use cmd's native START command so the backend
    # host is independent of this launcher and its parent terminal.
    $helperScript = Join-Path $ProjectRoot "run_generation_backend.ps1"
    $escapedHelper = $helperScript.Replace('"', '""')
    $startCommand = 'start "" /b powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $escapedHelper + '" -PythonPath "' + $python.Replace('"', '""') + '" -StdoutLog "' + $stdoutLog.Replace('"', '""') + '" -StderrLog "' + $stderrLog.Replace('"', '""') + '"'
    & cmd.exe /d /c $startCommand
    if ($LASTEXITCODE -ne 0) { throw "Detached Backend host could not be started." }

    $initialRecord = @{
        event = "started"
        status = "starting"
        launch_method = "launch_generation_backend.ps1"
        launcher_pid = $PID
        launch_process_pid = $null
        backend_pid = $null
        started_at_utc = $startedAt.ToString("o")
        last_health_at_utc = $null
        exit_code = $null
        process_path = $python
        working_directory = $ProjectRoot
        stdout_log = $stdoutLog
        stderr_log = $stderrLog
        host = "127.0.0.1"
        port = $Port
    }
    Write-Lifecycle $initialRecord
    $initialRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PidFile -Encoding UTF8

    if (Wait-BackendReady -TimeoutSeconds $StartupTimeoutSeconds) {
        $backendPids = @(Get-ListeningPids)
        $initialRecord.status = "healthy"
        $initialRecord.event = "started_healthy"
        $initialRecord.backend_pid = if ($backendPids.Count -gt 0) { $backendPids[0] } else { $null }
        $initialRecord.last_health_at_utc = [DateTime]::UtcNow.ToString("o")
        Write-Lifecycle $initialRecord
        $initialRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PidFile -Encoding UTF8
        Remove-Item -LiteralPath (Join-Path $RuntimeDirectory "last_launch_error.txt") -Force -ErrorAction SilentlyContinue
        exit 0
    }

    $initialRecord.status = "startup_timeout"
    $initialRecord.event = "startup_failed"
    $initialRecord.exit_code = $null
    Write-Lifecycle $initialRecord
    throw "Generation Backend did not become healthy within $StartupTimeoutSeconds seconds. See stdout_log and stderr_log in lifecycle.json."
} catch {
    if ($_.Exception.Message -notlike "*No duplicate process was started*") {
        New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
        Set-Content -LiteralPath (Join-Path $RuntimeDirectory "last_launch_error.txt") -Value $_.Exception.Message -Encoding UTF8
    }
    exit 1
} finally {
    if ($mutexAcquired) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
