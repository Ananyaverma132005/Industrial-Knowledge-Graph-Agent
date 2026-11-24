# Starts the API server + web UI on http://localhost:8000
Set-Location $PSScriptRoot
# Open the browser once the server has had time to load the embedding model.
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", "Start-Sleep 12; Start-Process http://localhost:8000"
& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
