# Starts the API (port 8000) and the web app (port 5173) in two new windows.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File .\dev.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$node = "C:\Program Files\nodejs"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; .\.venv\Scripts\python -m uvicorn app.main:app --reload --reload-dir app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:Path = '$node;' + `$env:Path; Set-Location '$root\frontend'; npm run dev"

Write-Host "API:  http://localhost:8000/docs"
Write-Host "App:  http://localhost:5173"
