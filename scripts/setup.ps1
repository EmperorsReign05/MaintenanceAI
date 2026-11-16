[CmdletBinding()]
param(
    [string]$Python = "py",
    [switch]$IncludeLstm
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Invoke-Expression "$Python $($Arguments -join ' ')"
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Python $($Arguments -join ' ')" }
}

Write-Host "Creating virtual environment in .venv ..."
Invoke-Python -m venv .venv
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
if ($IncludeLstm) { & $VenvPython -m pip install tensorflow }

Write-Host ""
Write-Host "Setup complete. Follow the ordered commands in PROJECT_GUIDE.md."
