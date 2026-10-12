# Operations runbook (Windows PowerShell)

**Daily start:** `run_preflight.ps1` → fix any FAIL → `run_dashboard.ps1` in one window → `run_paper_session.ps1`
(dry-run) for a few sessions → only then enable paper orders (`ENABLE_PAPER_TRADING=true`, `-ConfirmPaper`).
**Startup recovery:** the session reconciles with Alpaca first (account, positions, orders). Unknown orders or an
unreachable broker keep entries blocked and raise alerts.
**Emergency stop:** `emergency_halt.ps1` blocks all new entries. Existing positions keep their broker-side bracket
stops. To flatten manually use the Alpaca paper dashboard. Resume with `-Resume`.
**Broker timeout during submit:** the system never assumes failure; it looks up the order by `client_order_id`,
retries once only if the order is confirmed absent, otherwise marks it `unknown` and blocks entries until reconcile.
**Alerts to watch:** `CIRCUIT_BREAKER`, `RECONCILE_FAILED`, `ORDER_UNKNOWN`, `ORDER_REJECTED`, `DATA_UNAVAILABLE`.
**Smoke test (explicit only):** `python -m app.main smoke-order --i-understand` submits then cancels a 1-share limit
order far from market (needs `ENABLE_PAPER_TRADING=true`). Never runs automatically.
**Logs:** `logs/alphaforge.log` (JSON, secrets redacted). **DB:** `data/alphaforge.db`; back it up by copying the file
while the session is stopped. **Schedule:** one session per trading day; the loop idles outside market hours.
**Never** paste keys into chat, issues or screenshots; rotate keys in Alpaca if exposed.
