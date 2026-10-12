"""Performance metrics. Each value carries its provenance: 'broker' (account snapshots) or 'derived'
(FIFO pairing of recorded fills; estimates, not broker-confirmed realized P&L)."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from app.schemas import AccountSnapshot, Position


@dataclass
class Trade:
    symbol: str
    strategy: str | None
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    direction: str


def equity_metrics(history: list[AccountSnapshot], starting_equity: float | None = None) -> dict:
    if not history:
        return {"available": False}
    eq = np.array([h.equity for h in history], dtype=float)
    start = starting_equity or float(eq[0])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    last = history[-1]
    daily_pnl = last.equity - last.last_equity if last.last_equity else None
    return {"available": True, "source": history[-1].source, "start_equity": start, "ending_equity": float(eq[-1]),
            "total_return_pct": float((eq[-1] / start - 1) * 100) if start > 0 else None,
            "max_drawdown_pct": float(dd.max() * 100), "peak_equity": float(peak[-1]),
            "daily_pnl": daily_pnl, "buying_power": last.buying_power, "cash": last.cash,
            "n_snapshots": len(history)}


def pair_trades(fills: list[dict]) -> list[Trade]:
    """FIFO pairing per symbol; supports long and short round trips. Input rows ordered by time."""
    books: dict[str, deque] = defaultdict(deque)     # symbol -> deque[[signed_qty, price, strategy]]
    trades: list[Trade] = []
    for f in fills:
        sym, q, px = f["symbol"], float(f["qty"]), float(f["price"])
        signed = q if f["side"] == "buy" else -q
        strat = f.get("strategy")
        book = books[sym]
        while signed != 0 and book and (book[0][0] > 0) != (signed > 0):
            lot = book[0]
            m = min(abs(lot[0]), abs(signed))
            long = lot[0] > 0
            pnl = (px - lot[1]) * m if long else (lot[1] - px) * m
            trades.append(Trade(sym, lot[2] or strat, m, lot[1], px, pnl, "long" if long else "short"))
            lot[0] += -m if long else m
            signed += m if signed < 0 else -m
            if abs(lot[0]) < 1e-9:
                book.popleft()
        if abs(signed) > 1e-9:
            book.append([signed, px, strat])
    return trades


def trade_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"available": False, "source": "derived", "n_trades": 0}
    pnl = np.array([t.pnl for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = wins.sum(), -losses.sum()
    by: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by[t.strategy or "unattributed"].append(t.pnl)
    return {"available": True, "source": "derived", "n_trades": len(trades), "realized_pnl": float(pnl.sum()),
            "win_rate": float(len(wins) / len(pnl)), "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(gp / gl) if gl > 0 else (None if gp == 0 else float("inf")),
            "attribution": {k: {"pnl": float(sum(v)), "n": len(v)} for k, v in by.items()}}


def exposure_metrics(positions: list[Position], equity: float | None) -> dict:
    if not positions or not equity or equity <= 0:
        return {"available": bool(equity), "gross_pct": 0.0, "net_pct": 0.0, "unrealized_pnl": 0.0, "n_positions": 0,
                "source": "broker"}
    gross = sum(abs(p.market_value) for p in positions)
    net = sum(p.market_value for p in positions)
    return {"available": True, "source": "broker", "gross_pct": gross / equity * 100, "net_pct": net / equity * 100,
            "unrealized_pnl": float(sum(p.unrealized_pl for p in positions)), "n_positions": len(positions)}


def turnover(fills: list[dict], avg_equity: float | None) -> float | None:
    if not fills or not avg_equity:
        return None
    return float(sum(abs(float(f["qty"]) * float(f["price"])) for f in fills) / avg_equity)


def slippage_bps(fills: list[dict]) -> dict:
    """Entry slippage vs proposal reference price, measurable only for fills with a recorded ref_price."""
    vals = []
    for f in fills:
        ref = f.get("ref_price")
        if ref and f.get("proposal_id"):
            sign = 1 if f["side"] == "buy" else -1
            vals.append(sign * (float(f["price"]) / float(ref) - 1) * 1e4)
    if not vals:
        return {"available": False}
    return {"available": True, "avg_bps": float(np.mean(vals)), "n": len(vals), "source": "derived"}
