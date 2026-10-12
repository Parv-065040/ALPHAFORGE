# ALPHAFORGE one-time setup (Windows PowerShell). Run from the project root.
$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.10+ not found. Install from python.org and re-run." }
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - open it and paste YOUR Alpaca PAPER keys." }
Write-Host "Optional (FinBERT): .\.venv\Scripts\python.exe -m pip install transformers torch   (CUDA torch: see pytorch.org)"
Write-Host "Optional (LLM agents): install Ollama, then: ollama pull qwen3:8b ; ollama pull qwen3:4b ; set USE_LLM=true in .env"
Write-Host "Next: .\scripts\run_preflight.ps1"
