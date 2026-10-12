# Blocks ALL new entries immediately (existing broker-side bracket stops stay active). Resume with -Resume.
param([switch]$Resume)
if ($Resume) { .\.venv\Scripts\python.exe -m app.main halt --resume } else { .\.venv\Scripts\python.exe -m app.main halt --reason "operator emergency halt" }
