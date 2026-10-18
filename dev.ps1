# Starts the API (port 8000), the web app (port 5173) and, if its trained model is present, the
# M2 photo-matching service (port 8002) in new windows.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File .\dev.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$node = "C:\Program Files\nodejs"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; .\.venv\Scripts\python -m uvicorn app.main:app --reload --reload-dir app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:Path = '$node;' + `$env:Path; Set-Location '$root\frontend'; npm run dev"

# M2 photo matching: needs ml\photo_matching\.venv and ml\photo_matching\checkpoints\m2.pt (see ml\photo_matching\README.md)
$m2 = "$root\ml\photo_matching"
if (Test-Path "$m2\checkpoints\m2.pt") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$m2'; `$env:CHECKPOINT = 'checkpoints/m2.pt'; .\.venv\Scripts\python -m uvicorn pawmatch.api:app --port 8002"
    Write-Host "M2:   http://localhost:8002/health  (set AI_PROVIDER=app.ai.models:PawModels in backend\.env)"
} else {
    Write-Host "M2:   skipped (no ml\photo_matching\checkpoints\m2.pt)"
}

Write-Host "API:  http://localhost:8000/docs"
Write-Host "App:  http://localhost:5173"
