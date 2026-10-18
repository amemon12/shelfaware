param([switch]$Setup, [switch]$Lan)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Test-Path -LiteralPath $bundledPython) { $basePython = $bundledPython }
    else { $basePython = (Get-Command python -ErrorAction Stop).Source }
    & $basePython -m venv (Join-Path $repoRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed. Install Python 3.11+.' }
    $Setup = $true
}
if ($Setup) {
    & $venvPython -m pip install -r (Join-Path $repoRoot 'backend/requirements.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependencies failed.' }
    Push-Location (Join-Path $repoRoot 'web')
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Web dependencies failed.' }
    } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'web/node_modules/vite/bin/vite.js'))) {
    throw 'Run this script with -Setup first.'
}
foreach ($port in @(8000, 5173)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Reuse that server or stop it explicitly before starting another."
    }
}
$runtimeDir = Join-Path $repoRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$bindAddress = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
$api = Start-Process -FilePath $venvPython -ArgumentList @('-m','uvicorn','app.main:app','--host',$bindAddress,'--port','8000') -WorkingDirectory (Join-Path $repoRoot 'backend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'backend.out.log') -RedirectStandardError (Join-Path $runtimeDir 'backend.err.log')
$nodeExe = (Get-Command node -ErrorAction Stop).Source
$web = Start-Process -FilePath $nodeExe -ArgumentList @('node_modules/vite/bin/vite.js','--host',$bindAddress,'--port','5173','--strictPort') -WorkingDirectory (Join-Path $repoRoot 'web') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'web.out.log') -RedirectStandardError (Join-Path $runtimeDir 'web.err.log')
@{backend=$api.Id;web=$web.Id;started_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir 'processes.json')
Write-Host 'Shelf Aware: http://localhost:5173'
Write-Host "Processes: API $($api.Id), web $($web.Id). Logs: $runtimeDir"
Write-Host 'For a phone use the computer LAN IP; keep this on a trusted network. Native iPhone camera works over LAN; browser live camera needs a secure origin.'
Write-Host "Stop only these processes when finished: Stop-Process -Id $($api.Id),$($web.Id)"

