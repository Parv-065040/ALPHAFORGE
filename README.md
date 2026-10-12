# ALPHAFORGE

Multi-agent, **paper-only** trading system for liquid US stocks/ETFs on an Alpaca paper account, built for a course
trading competition. Python 3.10+, Pydantic, SQLite, Streamlit + Plotly dashboard, optional FinBERT / Ollama / CrewAI.

> **There is no live-trading path.** The broker adapter hard-codes `https://paper-api.alpaca.markets`, requires
> `ALPACA_PAPER=true`, and fails closed otherwise. Orders are disabled until you set `ENABLE_PAPER_TRADING=true`
> **and** pass `--confirm-paper`. This software does not guarantee profit and nothing here is financial advice.

## What was verified and what was not
| Verified in the build sandbox | NOT verified (needs you) |
|---|---|
| 162 automated tests pass (unit, mocked-broker lifecycle, end-to-end pipeline on fixtures, replay, dashboard render) | Anything against the live Alpaca paper API (payload shapes follow docs; never called) |
| Dashboard renders (headless) for demo and empty databases | FinBERT, Ollama and CrewAI were **not run** (not installed in the sandbox) |
| Risk limits, idempotency, reconciliation, timeout handling via a fake broker | Real strategy performance - replay on synthetic data proves mechanics only |
| Paper-only guard, secret redaction | Whether your course permits short selling (kept **off**) |

## Quick start (Windows PowerShell, from this folder)
```powershell
.\scripts\setup_windows.ps1          # venv + dependencies + creates .env
notepad .env                          # paste YOUR Alpaca PAPER key id / secret. Keep ALPACA_PAPER=true
.\scripts\run_tests.ps1               # expect all green
.\scripts\run_dashboard.ps1 -Demo     # synthetic demo data, clearly bannered
.\scripts\run_preflight.ps1           # checks keys, paper mode, account, optional models
.\.venv\Scripts\python.exe -m app.main scan            # ONE dry-run cycle, never submits orders
.\scripts\run_paper_session.ps1 -Once                  # session in dry-run (orders disabled by default)
```
To allow paper orders: set `ENABLE_PAPER_TRADING=true` in `.env`, run preflight, then
`.\scripts\run_paper_session.ps1 -ConfirmPaper`. Stop trading instantly with `.\scripts\emergency_halt.ps1`.
Linux/macOS: replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Commands (`python -m app.main ...`)
`preflight` · `scan` (dry-run) · `paper-session [--confirm-paper] [--once]` · `replay --demo | --data-dir data/replay` ·
`dashboard [--demo]` · `demo-seed` · `report` · `review` · `halt [--resume]` · `smoke-order --i-understand`

## Layout
`app/` config, schemas, data, sentiment, strategies, risk, agents, execution, storage, flow, replay, analytics,
dashboard · `config/default.yaml` · `tests/` · `scripts/` · `docs/` (architecture, risk policy, strategies,
data & sentiment, operations runbook, testing, demo script, model selection, limitations)

## Optional components
- **FinBERT**: `pip install transformers torch` (CUDA build of torch for the RTX GPU). Without it a transparent
  rule-based scorer is used and labelled `rule_based` everywhere.
- **LLM agents**: install Ollama, `ollama pull qwen3:8b` (fallback `qwen3:4b`), set `USE_LLM=true`. Without it,
  agents use deterministic fallbacks and the proposal source is labelled `quant_only`.
- **CrewAI Flow**: `pip install crewai`, then `scan --crewai` (wrapper untested; the plain flow is the tested path).

See `docs/LIMITATIONS.md` before relying on anything.
