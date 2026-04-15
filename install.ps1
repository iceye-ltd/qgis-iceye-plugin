param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot").Path,
    [string]$QgisProfile = "default",
    [switch]$UseJunction
)

$ErrorActionPreference = "Stop"

$pluginName = "ICEYE_toolbox"
$sourcePath = Join-Path $RepoRoot $pluginName

if (-not (Test-Path $sourcePath)) {
    throw "Plugin source folder not found: $sourcePath"
}

if (-not $env:APPDATA) {
    throw "APPDATA is not set. Run this in a native Windows PowerShell session."
}

$pluginsDir = Join-Path $env:APPDATA "QGIS\QGIS3\profiles\$QgisProfile\python\plugins"
$targetPath = Join-Path $pluginsDir $pluginName

Write-Host "Source : $sourcePath"
Write-Host "Target : $targetPath"

New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null

if (Test-Path $targetPath) {
    Write-Host "Removing existing target: $targetPath"
    Remove-Item -Path $targetPath -Recurse -Force
}

if ($UseJunction) {
    Write-Host "Creating directory junction..."
    cmd /c "mklink /J `"$targetPath`" `"$sourcePath`"" | Out-Host
}
else {
    try {
        Write-Host "Creating symbolic link..."
        New-Item -ItemType SymbolicLink -Path $targetPath -Target $sourcePath | Out-Null
    }
    catch {
        Write-Warning "Symbolic link failed ($($_.Exception.Message)). Falling back to junction..."
        cmd /c "mklink /J `"$targetPath`" `"$sourcePath`"" | Out-Host
    }
}

Write-Host ""
Write-Host "Install complete."
Write-Host "QGIS will load plugin from your repo (live-edit workflow)."
Write-Host "If QGIS is open, reload the plugin (or restart QGIS)."
