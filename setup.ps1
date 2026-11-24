# One-time setup: virtual env, dependencies, sample data, ML model and knowledge graph.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python 3.11 virtual environment..."
    py -3.11 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m spacy download en_core_web_sm

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

& $py make_sample_manuals.py   # sample SOP PDFs (replace with real manuals any time)
& $py train_model.py           # RandomForest failure model
& $py ingest.py                # asset registry + tickets -> knowledge graph (spaCy NER)
& $py ingest_pdf.py            # PDF manuals -> SOP vectors

Write-Host "`nSetup complete. Start the app with:  powershell -File run.ps1"
