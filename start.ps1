# SpriteMind Local Launcher (Windows PowerShell)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             SPRITEMIND: LOCAL AI PLATFORM                " -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Initializing deterministic renderer and neural SVAN model..." -ForegroundColor Yellow

# Set Python path to backend
$env:PYTHONPATH = "$PSScriptRoot\backend"
$env:HOST = "127.0.0.1"
$env:PORT = "7860"

Write-Host "Launching server at http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate the local server.`n" -ForegroundColor DarkGray

# Launch Python backend server (which serves API and mounted React frontend)
python "$PSScriptRoot\backend\run_server.py"
