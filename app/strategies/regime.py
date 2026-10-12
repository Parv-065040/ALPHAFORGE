"""Deterministic market regime from the benchmark. LLM commentary can never override this."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import RegimeCfg


@dataclass
class RegimeState:
    regime: str                 # risk_on | neutral | risk_off | unclear
    score: float                # 0 (bearish) .. 1 (bullish); 0.5 when unclear
    reasons: list[str] = field(default_factory=list)


def compute_regime(bench: pd.DataFrame | None, cfg: RegimeCfg) -> RegimeState:
    if bench is None or len(bench) < 70:
        return RegimeState("unclear", 0.5, ["benchmark history missing/insufficient"])
    c = bench["close"]
    s20, s50 = c.rolling(20).mean(), c.rolling(50).mean()
    pc = c.shift(1)
    tr = pd.concat([bench["high"] - bench["low"], (bench["high"] - pc).abs(), (bench["low"] - pc).abs()], axis=1).max(axis=1)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / c.iloc[-1])
    score, reasons = 0.0, []
    if c.iloc[-1] > s50.iloc[-1]:
        score += 0.35; reasons.append("benchmark above 50d SMA")
    else:
        reasons.append("benchmark below 50d SMA")
    if s20.iloc[-1] > s50.iloc[-1]:
        score += 0.25; reasons.append("20d SMA above 50d SMA")
    if s50.iloc[-1] > s50.iloc[-11]:
        score += 0.20; reasons.append("50d SMA rising")
    if atr_pct <= cfg.max_benchmark_atr_pct:
        score += 0.20; reasons.append(f"volatility contained (ATR {atr_pct:.2%})")
    else:
        reasons.append(f"volatility elevated (ATR {atr_pct:.2%})")
    regime = "risk_on" if score >= 0.7 else "neutral" if score >= 0.4 else "risk_off"
    return RegimeState(regime, round(score, 3), reasons)
