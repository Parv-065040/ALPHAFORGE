# Architecture

```
Alpaca data/news ─► providers ─► sentiment pipeline ─┐
                                 features/regime ────┼─► scanner ─► Signals
                                                     │              │
          Agents (bounded, typed, optional LLM) ◄────┘              ▼
   Catalyst Analyst · Regime Analyst · Portfolio Manager ─► TradeProposal (Python builds it; PM may only skip/scale DOWN)
                                                              │
                                              RiskEngine (deterministic authority)
                                                              │ stored RiskDecision
                                         ExecutionService (idempotent, reconciling) ─► Alpaca PAPER
                                                              │
                                            SQLite ─► analytics ─► Streamlit dashboard
```
**Authority split.** Numbers (features, scoring, sizing, limits, order construction) are ordinary Python. LLMs only
classify/summarise and can only skip or shrink a trade; malformed or ambiguous output becomes a safe no-trade.
**Single path to orders.** `ExecutionService.execute` requires: trading enabled + paper confirmed, a *stored*
approved `RiskDecision` matching the proposal, fresh approval and proposal, and a recent broker reconciliation.
**Orchestration.** `app/flow/pipeline.py` (plain Python, tested) with steps `data → scan → propose+risk → execute`.
`app/flow/crewai_flow.py` wraps the same steps in a CrewAI Flow (unverified; optional).
**Time.** All timestamps are timezone-aware UTC internally; the XNYS calendar (`exchange_calendars`) handles DST,
holidays and early closes. Daily bars used for signals are completed sessions only (today's partial bar is dropped).
**Persistence.** SQLite with versioned migrations; every scan, candidate, signal, proposal, risk decision, order
state change, fill, account/position snapshot, agent run, provider health and alert is stored. Demo data lives in a
separate DB flagged `origin=demo`.
