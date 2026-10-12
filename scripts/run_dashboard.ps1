param([switch]$Demo)
if ($Demo) { .\.venv\Scripts\python.exe -m app.main dashboard --demo } else { .\.venv\Scripts\python.exe -m app.main dashboard }
