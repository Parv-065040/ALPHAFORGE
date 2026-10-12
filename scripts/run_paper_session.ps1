# Dry-run by default (ENABLE_PAPER_TRADING=false in .env). To allow PAPER orders set ENABLE_PAPER_TRADING=true
# in .env AND pass -ConfirmPaper. Preflight must pass first; the session refuses to start otherwise.
param([switch]$ConfirmPaper, [switch]$Once)
$a = @("-m","app.main","paper-session")
if ($ConfirmPaper) { $a += "--confirm-paper" }
if ($Once) { $a += "--once" }
.\.venv\Scripts\python.exe @a
