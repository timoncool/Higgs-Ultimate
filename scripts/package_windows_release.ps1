[CmdletBinding()]
param(
    [string]$Version,
    [switch]$SkipBuild,
    [switch]$SkipUpx,
    [string]$UpxPath
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$desktopRoot = Join-Path $repoRoot "desktop"
$tauriRoot = Join-Path $desktopRoot "src-tauri"
$releaseRoot = Join-Path $tauriRoot "target\release"

if ([string]::IsNullOrWhiteSpace($Version)) {
    $package = Get-Content (Join-Path $desktopRoot "package.json") -Raw | ConvertFrom-Json
    $Version = [string]$package.version
}
$Version = $Version.TrimStart("v")

if (-not $SkipBuild) {
    Push-Location $desktopRoot
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Tauri build failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

$sourceExe = Join-Path $releaseRoot "higgs-audio-studio.exe"
$sourceResources = Join-Path $releaseRoot "resources"
if (-not (Test-Path -LiteralPath $sourceExe)) { throw "Release executable not found: $sourceExe" }
if (-not (Test-Path -LiteralPath $sourceResources)) { throw "Release resources not found: $sourceResources" }

$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $releaseRoot "artifacts\v$Version"))
$portableDir = [System.IO.Path]::GetFullPath((Join-Path $artifactRoot "Higgs Audio v3 Studio $Version Portable"))
if (-not $portableDir.StartsWith($artifactRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Portable output escaped the release artifact directory"
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
if (Test-Path -LiteralPath $portableDir) {
    Remove-Item -LiteralPath $portableDir -Recurse -Force
}
New-Item -ItemType Directory -Path $portableDir -Force | Out-Null
Copy-Item -LiteralPath $sourceExe -Destination (Join-Path $portableDir "Higgs Audio v3 Studio.exe")
Copy-Item -LiteralPath $sourceResources -Destination (Join-Path $portableDir "resources") -Recurse
$portableEngineDir = Join-Path $portableDir "resources\engine"
if (Test-Path -LiteralPath $portableEngineDir) {
    Remove-Item -LiteralPath $portableEngineDir -Recurse -Force
}
New-Item -ItemType Directory -Path $portableEngineDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $portableDir "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $portableDir "data\speakers") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $portableDir "data\temp") -Force | Out-Null
Set-Content -LiteralPath (Join-Path $portableDir "portable.flag") -Value "Higgs Audio v3 Studio portable mode" -Encoding Ascii

if (-not $SkipUpx) {
    $upx = $null
    if ($UpxPath) {
        $upx = Get-Item -LiteralPath $UpxPath -ErrorAction Stop
    } else {
        $upxCommand = Get-Command upx.exe -ErrorAction SilentlyContinue
        if ($upxCommand) { $upx = Get-Item -LiteralPath $upxCommand.Source }
    }
    if ($upx) {
        & $upx.FullName --best --lzma (Join-Path $portableDir "Higgs Audio v3 Studio.exe")
        if ($LASTEXITCODE -ne 0) { throw "UPX failed with exit code $LASTEXITCODE" }
    } else {
        Write-Warning "UPX was not found. The portable executable was left uncompressed."
    }
}

$msi = Get-ChildItem (Join-Path $releaseRoot "bundle\msi") -Filter "*$Version*.msi" -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$nsis = Get-ChildItem (Join-Path $releaseRoot "bundle\nsis") -Filter "*$Version*-setup.exe" -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($msi) { Copy-Item -LiteralPath $msi.FullName -Destination $artifactRoot -Force }
if ($nsis) { Copy-Item -LiteralPath $nsis.FullName -Destination $artifactRoot -Force }

Write-Host "Portable: $portableDir"
if ($nsis) { Write-Host "NSIS:     $(Join-Path $artifactRoot $nsis.Name)" }
if ($msi) { Write-Host "MSI:      $(Join-Path $artifactRoot $msi.Name)" }
