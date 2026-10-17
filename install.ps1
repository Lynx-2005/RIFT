# Install rift on Windows: download the release zip and put rift.exe on PATH.
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

$repo = "Lynx-2005/RIFT"
$target = "x86_64-pc-windows-msvc"
$dir = Join-Path $env:LOCALAPPDATA "rift\bin"
$url = "https://github.com/$repo/releases/latest/download/rift-$target.zip"

New-Item -ItemType Directory -Force -Path $dir | Out-Null
$zip = Join-Path $env:TEMP "rift-$target.zip"

Write-Host "rift: downloading $target"
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $dir -Force
Remove-Item $zip

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$dir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$dir", "User")
    Write-Host "rift: added $dir to your user PATH; open a new terminal"
}
Write-Host "rift: installed $dir\rift.exe"
