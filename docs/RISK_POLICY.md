# Risk policy (all values in `config/default.yaml`; provisional hypotheses, not validated optima)

| Control | Default | Enforced in |
|---|---|---|
| Risk per trade | 0.5% of equity (0.75% needs `higher_risk_approved: true`; config refuses to load otherwise) | `size_position` |
| Max position | 15% of equity | sizing |
| Max gross exposure | 70% of equity (includes working orders and proposals approved earlier in the same cycle) | sizing |
| Envelopes ($100k): intraday 40k, swing 25k, reserve 25k, uncommitted 10k | caps notional per horizon | sizing |
| Correlated groups (e.g. mega-cap tech) | 30% of equity | sizing |
| Liquidity | avg $ volume ≥ $20M; ≤1% of 20-day avg shares | check + sizing |
| Max positions / new entries per day | 8 / 4 | checks |
| Min reward/risk | 1.5 | check |
| Daily loss circuit breaker | −2% vs broker `last_equity` → block new entries | check, alert |
| Drawdown review | 5% from peak equity → block new entries until operator review | check, alert |
| Existing positions on breaker | `hold_with_stops` (broker-side bracket stops remain) or `flatten_intraday` | flow |
| Short selling | **off** (`ALLOW_SHORTING=false`); also needs asset `shortable` | check |
| Session | no entries in first 5 / last 15 minutes; intraday positions flattened 10 minutes before close | check, execution |
| Proposal / approval TTL | 30 min / 5 min | check, execution |
Entries are **blocked** (not "best-effort") on: stale quotes/bars, unreconciled broker, expired/duplicate proposal,
operator halt, market closed, any breaker. Exits (stops, targets, flatten) are never blocked by entry rules.
Sizing is `min()` of all caps and floors to whole shares; `qty × stop distance ≤ risk budget` is tested.
