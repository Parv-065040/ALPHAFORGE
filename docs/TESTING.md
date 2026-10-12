# Testing
`.\scripts\run_tests.ps1` (≈1 minute). Layout: `tests/unit` (config guards, schemas, agents with fake LLMs, strategies,
risk boundaries, sentiment, time/DST/holidays, metrics, storage, dashboard states, preflight),
`tests/integration` (mocked-broker lifecycle, end-to-end pipeline on fixtures, Alpaca REST adapters on a mocked
session, headless dashboard render), `tests/replay` (determinism, no look-ahead, next-open fills, regression).
Key properties tested: no live path; no orders unless explicitly enabled; execution needs a stored approved decision;
idempotent submission; timeout reconciliation; risk-per-trade never exceeded; exposure respected within a cycle;
malformed LLM output → no-trade; secrets never in logs/config views/dashboard; empty/unavailable dashboard states
never show fabricated numbers; features/replay have no look-ahead.
Not covered: live Alpaca behaviour, FinBERT/Ollama/CrewAI execution, concurrency across multiple processes.
