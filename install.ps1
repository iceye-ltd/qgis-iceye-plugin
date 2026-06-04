
param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot").Path,
    [string]$QgisProfile = "default",
    [switch]$UseJunction
)
 
$ErrorActionPreference = "Stop"
 
$pluginName = "iceye_toolbox"
 
# In this repo the plugin package IS the repo root (metadata.txt / __init__.py
# live at top level), so we link the repo root itself as the plugin folder.
$sourcePath = $RepoRoot
 
if (-not (Test-Path (Join-Path $sourcePath "metadata.txt"))) {
    throw "metadata.txt not found in $sourcePath - is this the plugin repo root?"
}
 
if (-not $env:APPDATA) {
    throw "APPDATA is not set. Run this in a native Windows PowerShell session."
}
 
$base = Join-Path $env:APPDATA "QGIS"
 
# Pick the profile folder that actually exists (QGIS4 or QGIS3); prefer the
# newest. Fall back to QGIS3 if QGIS hasn't been launched yet.
$qgisDir = "QGIS3"
foreach ($v in @("QGIS4", "QGIS3")) {
    if (Test-Path (Join-Path $base "$v\profiles\$QgisProfile\python")) {
        $qgisDir = $v
        break
    }
}
 
$pluginsDir = Join-Path $base "$qgisDir\profiles\$QgisProfile\python\plugins"
$targetPath = Join-Path $pluginsDir $pluginName
 
Write-Host "Profile dir : $qgisDir"
Write-Host "Source      : $sourcePath"
Write-Host "Target      : $targetPath"
 
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
Write-Host "Install complete ($qgisDir)."
Write-Host "QGIS will load plugin from your repo (live-edit workflow)."
Write-Host "If QGIS is open, reload the plugin (or restart QGIS)."