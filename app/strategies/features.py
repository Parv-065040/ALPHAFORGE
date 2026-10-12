"""Feature computation on COMPLETED daily bars only. Every feature at index i uses bars[:i+1] (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import MomentumCfg


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame, cfg: MomentumCfg, bench: pd.DataFrame | None = None,
                     rs_lookback: int = 63) -> dict[str, float] | None:
    """Return the feature dict for the LAST bar of df, or None when history is insufficient/invalid."""
    need = max(cfg.sma_trend, rs_lookback, cfg.breakout_lookback, 50) + 12
    if df is None or len(df) < need:
        return None
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    a = atr(df, cfg.atr_period)
    sma_f, sma_s, sma_t = c.rolling(cfg.sma_fast).mean(), c.rolling(cfg.sma_slow).mean(), c.rolling(cfg.sma_trend).mean()
    last = len(df) - 1
    n = cfg.breakout_lookback
    f = {
        "close": float(c.iloc[last]),
        "prev_close": float(c.iloc[last - 1]),
        "sma_fast": float(sma_f.iloc[last]),
        "sma_slow": float(sma_s.iloc[last]),
        "sma_trend": float(sma_t.iloc[last]),
        "sma_slow_slope": float(sma_s.iloc[last] / sma_s.iloc[last - 10] - 1),
        "atr": float(a.iloc[last]),
        "high_n": float(h.iloc[last - n:last].max()),     # N bars BEFORE the last bar
        "low_n": float(l.iloc[last - n:last].min()),
        "high_50": float(h.iloc[last - 49:last + 1].max()),
        "vol_ratio": float(v.iloc[last] / max(v.iloc[last - 20:last].mean(), 1.0)),
        "avg_volume": float(v.iloc[last - 19:last + 1].mean()),
        "avg_dollar_volume": float((c * v).iloc[last - 19:last + 1].mean()),
        "roc_63": float(c.iloc[last] / c.iloc[last - 63] - 1),
        "ret_1d": float(c.iloc[last] / c.iloc[last - 1] - 1),
    }
    f["atr_pct"] = f["atr"] / f["close"]
    rs_stock = float(c.iloc[last] / c.iloc[last - rs_lookback] - 1)
    rs_bench = 0.0
    if bench is not None and len(bench) > rs_lookback:
        b = bench["close"].reindex(df.index, method="ffill")
        if b.notna().iloc[last] and b.notna().iloc[last - rs_lookback]:
            rs_bench = float(b.iloc[last] / b.iloc[last - rs_lookback] - 1)
    f["rs"] = rs_stock - rs_bench
    if not all(np.isfinite(x) for x in f.values()) or f["close"] <= 0 or f["atr"] <= 0:
        return None
    return f
