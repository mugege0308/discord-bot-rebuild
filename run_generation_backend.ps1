[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,
    [Parameter(Mandatory = $true)]
    [string]$StdoutLog,
    [Parameter(Mandatory = $true)]
    [string]$StderrLog
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$serverScript = Join-Path $ProjectRoot "generation_backend_server.py"

try {
    Set-Location -LiteralPath $ProjectRoot
    & $PythonPath -u $serverScript 1>> $StdoutLog 2>> $StderrLog
    exit $LASTEXITCODE
} catch {
    $_ | Out-File -LiteralPath $StderrLog -Append -Encoding UTF8
    exit 1
}
