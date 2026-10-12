# 5-minute demo script
1. `run_tests.ps1` - show green suite (30 s).
2. `run_dashboard.ps1 -Demo` - point out the DEMO banner; tour Portfolio, Strategy monitor (signals, score components,
   no-trade reasons), Agent activity, Sentiment Intelligence, Operations, Config (no secrets).
3. Terminal: `python -m app.main replay --demo` then explain it is synthetic and mechanics-only.
4. `run_preflight.ps1` - show fail-closed checks (paper flag, keys present/missing, orders disabled).
5. `python -m app.main scan` - dry-run cycle with real data; show signals or an honest "no qualifying signals".
6. Explain the authority split: LLM can skip/shrink only; risk engine and execution guard are deterministic.
7. If enabled: `paper-session --confirm-paper --once` during market hours, then show the order and decision trail in
   the dashboard. Close with `emergency_halt.ps1`.
Be upfront: results are paper trading, strategies are hypotheses, past/synthetic results do not predict returns.
