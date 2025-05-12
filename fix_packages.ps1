# PowerShell script to fix corrupted Python packages

# First run the Python script to remove corrupted packages
Write-Host "Running Python script to remove corrupted packages..." -ForegroundColor Cyan
python fix_packages.py

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Reinstall potentially corrupted packages
Write-Host "`nReinstalling potentially corrupted packages..." -ForegroundColor Cyan

# Define packages to reinstall
$packages = @(
    "whisperx",
    "openai-whisper",
    "sympy",
    "pywin32"
)

# Reinstall each package
foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Yellow
    python -m pip install --force-reinstall $package
}

# Check for any remaining invalid distributions
Write-Host "`nChecking for remaining issues..." -ForegroundColor Cyan
python -m pip check

Write-Host "`nDone! Your Python environment should now be fixed." -ForegroundColor Green
Write-Host "If you still see warnings, you may need to manually remove the corrupted packages." -ForegroundColor Yellow 