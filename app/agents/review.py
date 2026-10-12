"""Performance / strategy review. RECOMMENDS only: writes recommendation rows, never touches configuration."""
from __future__ import annotations

from app.analytics.metrics import pair_trades, trade_metrics
from app.storage.db import Storage


def review_strategies(db: Storage, min_trades: int = 20) -> list[str]:
    trades = pair_trades(db.fills_with_orders())
    by: dict[str, list] = {}
    for t in trades:
        by.setdefault(t.strategy or "unattributed", []).append(t)
    recs: list[str] = []
    if not trades:
        recs.append("No closed trades yet; no strategy changes can be recommended.")
    for strat, ts in by.items():
        m = trade_metrics(ts)
        if m["n_trades"] < min_trades:
            recs.append(f"{strat}: only {m['n_trades']} closed trades (<{min_trades}); sample too small to judge.")
            continue
        pf = m["profit_factor"]
        if pf is not None and pf < 1.0:
            recs.append(f"{strat}: profit factor {pf:.2f} < 1 over {m['n_trades']} trades; consider reviewing or "
                        "disabling (operator decision; config is NOT changed automatically).")
        else:
            recs.append(f"{strat}: profit factor {pf} over {m['n_trades']} trades; no change recommended.")
    for r in recs:
        db.add_recommendation("*", r, {"n_trades": len(trades)})
    return recs
