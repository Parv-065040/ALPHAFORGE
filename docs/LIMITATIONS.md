# Known limitations and unverified items
- **Never run against live Alpaca.** REST shapes (account, orders incl. bracket + nested legs, bars, quotes, news)
  follow Alpaca's documented API but are unproven; expect small fixes on first contact. Run `preflight`, then `scan`,
  then dry-run sessions before enabling orders.
- FinBERT, Ollama/Qwen and CrewAI were not executed here. The CrewAI wrapper is untested.
- Free IEX data: partial volume, quote timeliness varies; catalyst volume confirmation uses last daily bar.
- Bracket-order behaviours (GTC parent with child legs, cancel semantics) must be checked in the Alpaca paper UI.
- Realized P&L is derived from recorded fills via FIFO (not broker-confirmed); equity/exposure come from broker snapshots.
- Strategy parameters are hypotheses; replay uses synthetic or user-supplied CSV data only; no walk-forward or
  statistical validation is included. Past or paper results do not predict future returns.
- Single-process design (SQLite); do not run two sessions on the same DB.
- Dividends, splits mid-position, halts, and pre/post-market trading are not specially handled.
- Short selling is disabled and untested against a real account; confirm course rules and account permissions first.
- Not financial advice.
