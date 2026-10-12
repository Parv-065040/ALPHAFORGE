"""Configurable score (hypothesis weights, NOT proven optimal). Components with no data are excluded
and the remaining weights renormalised, so absent news neither rewards nor penalises."""
from __future__ import annotations

import numpy as np

from app.schemas import Direction, SymbolSentiment


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def components(f: dict, direction: Direction, price: float, rs_rank: float | None, regime_score: float,
               sent: SymbolSentiment | None, level: float) -> dict[str, float | None]:
    sign = 1.0 if direction == Direction.LONG else -1.0
    trend = (0.4 * _clip(sign * (f["close"] / f["sma_slow"] - 1) / 0.10)
             + 0.3 * _clip(sign * f["sma_slow_slope"] / 0.03)
             + 0.3 * _clip(sign * f["roc_63"] / 0.20))
    # relative strength: cross-sectional percentile when available, else scaled spread vs benchmark
    rs = _clip(rs_rank) if rs_rank is not None else _clip(0.5 + sign * f["rs"] / 0.20 * 0.5)
    if rs_rank is not None and direction == Direction.SHORT:
        rs = 1.0 - rs
    brk = sign * (price - level) / f["atr"]
    vol = 0.5 * _clip((f["vol_ratio"] - 1.0) / 1.5) + 0.5 * _clip(brk / 1.5)
    cat = None
    if sent is not None and sent.n_items > 0 and not sent.stale:
        cat = _clip(sign * sent.score * (0.5 + 0.5 * sent.max_materiality))
    reg = _clip(regime_score if direction == Direction.LONG else 1 - regime_score)
    return {"trend_momentum": trend, "relative_strength": rs, "volume_breakout": vol, "catalyst": cat, "regime": reg}


def total_score(comps: dict[str, float | None], weights: dict[str, float]) -> float:
    num = den = 0.0
    for k, w in weights.items():
        v = comps.get(k)
        if v is None:
            continue
        num += w * v
        den += w
    return round(num / den, 4) if den > 0 else 0.0
