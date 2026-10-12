from __future__ import annotations

from app.analytics.metrics import (equity_metrics, exposure_metrics, pair_trades, slippage_bps, trade_metrics,
                                   turnover)
from app.storage.db import Storage


def performance_summary(db: Storage, starting_equity: float | None = None) -> dict:
    hist = db.account_history()
    fills = db.fills_with_orders()
    eqm = equity_metrics(hist, starting_equity)
    tm = trade_metrics(pair_trades(fills))
    pos = db.latest_positions()
    ex = exposure_metrics(pos, eqm.get("ending_equity") if eqm.get("available") else None)
    avg_eq = sum(h.equity for h in hist) / len(hist) if hist else None
    unreal = ex.get("unrealized_pnl") if ex.get("available") else None
    return {"equity": eqm, "trades": tm, "exposure": ex, "turnover": turnover(fills, avg_eq),
            "slippage": slippage_bps(fills), "realized_pnl": tm.get("realized_pnl"), "unrealized_pnl": unreal,
            "provenance": "equity/exposure/unrealized = broker snapshots; trade stats & realized P&L = derived from recorded fills"}
