"""Bar-by-bar replay of the SAME strategy code. No look-ahead: signals use bars[:i+1] (completed day i);
entries fill at the NEXT bar's open; stops/targets are evaluated on later bars (stop wins if both touch).
Catalyst strategy is not replayable (no historical news store) and is excluded. Results are not predictions."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings
from app.data.providers import bars_from_records, synthetic_daily_bars
from app.schemas import Direction
from app.strategies.features import compute_features
from app.strategies.regime import compute_regime
from app.strategies.setups import momentum_breakout, relative_strength
from app.strategies.scoring import components, total_score


@dataclass
class ReplayTrade:
    symbol: str
    strategy: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    qty: int
    pnl: float
    reason: str


@dataclass
class ReplayResult:
    origin: str
    start_equity: float
    end_equity: float
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    trades: list[ReplayTrade] = field(default_factory=list)
    benchmark_return_pct: float | None = None
    metrics: dict = field(default_factory=dict)


def load_csv_dir(path: str) -> dict[str, pd.DataFrame]:
    """CSV per symbol: columns date,open,high,low,close,volume (e.g. SPY.csv)."""
    out = {}
    for f in Path(path).glob("*.csv"):
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["ts"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("ts").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
        out[f.stem.upper()] = df
    return out


def demo_frames(symbols: list[str], n: int = 500, seed: int = 7) -> dict[str, pd.DataFrame]:
    return {s: synthetic_daily_bars(s, n, seed, drift=0.0005 + (i % 5) * 0.0001, vol=0.012 + (i % 4) * 0.003)
            for i, s in enumerate(symbols)}


def run_replay(settings: Settings, frames: dict[str, pd.DataFrame], origin: str, start_equity: float = 100_000,
               slippage_bps: float = 5.0, max_hold_days: int = 20, use_rs: bool = False) -> ReplayResult:
    st = settings.strategies
    bench_sym = settings.data.benchmark_symbol
    syms = [s for s in settings.universe.symbols if s in frames]
    if bench_sym not in frames or not syms:
        raise ValueError(f"replay needs benchmark {bench_sym} and at least one universe symbol")
    dates = frames[bench_sym].index
    warm = max(st.momentum.sma_trend, 63, 50) + 15
    cash, equity = start_equity, start_equity
    open_pos: dict[str, dict] = {}
    res = ReplayResult(origin, start_equity, start_equity)
    slip = slippage_bps / 1e4
    rcfg = settings.risk
    pending: list[dict] = []
    for i in range(warm, len(dates) - 1):
        day, nxt = dates[i], dates[i + 1]
        # 1) fill pending entries at today's open (decided yesterday)
        for o in pending:
            df = frames[o["symbol"]]
            if day not in df.index:
                continue
            px = float(df.loc[day, "open"]) * (1 + slip)
            risk_ps = px - o["stop"]
            if risk_ps <= 0:
                continue
            gross = sum(p["qty"] * p["last"] for p in open_pos.values())
            eq_now = cash + gross
            qty = min(math.floor(eq_now * rcfg.risk_per_trade_pct / risk_ps),
                      math.floor(eq_now * rcfg.max_position_pct / px),
                      math.floor(max(eq_now * rcfg.max_gross_exposure_pct - gross, 0) / px), math.floor(cash / px))
            if qty >= 1 and len(open_pos) < rcfg.max_positions and o["symbol"] not in open_pos:
                cash -= qty * px
                open_pos[o["symbol"]] = {"qty": qty, "entry": px, "stop": o["stop"], "target": o["target"],
                                         "entry_date": str(day.date()), "strategy": o["strategy"], "days": 0, "last": px}
        pending = []
        # 2) manage open positions on today's bar (stop first if both touched)
        for sym in list(open_pos):
            df = frames[sym]
            if day not in df.index:
                continue
            b, p = df.loc[day], open_pos[sym]
            p["days"] += 1
            exit_px, why = None, ""
            if b["open"] <= p["stop"]:
                exit_px, why = float(b["open"]), "gap through stop"
            elif b["low"] <= p["stop"]:
                exit_px, why = p["stop"], "stop"
            elif b["high"] >= p["target"]:
                exit_px, why = p["target"], "target"
            elif p["days"] >= max_hold_days:
                exit_px, why = float(b["close"]), "time stop"
            p["last"] = float(b["close"])
            if exit_px is not None:
                exit_px *= (1 - slip)
                cash += p["qty"] * exit_px
                res.trades.append(ReplayTrade(sym, p["strategy"], p["entry_date"], str(day.date()), p["entry"],
                                              exit_px, p["qty"], (exit_px - p["entry"]) * p["qty"], why))
                del open_pos[sym]
        equity = cash + sum(p["qty"] * p["last"] for p in open_pos.values())
        res.equity_curve.append((str(day.date()), equity))
        # 3) generate signals from completed bar `day` (history up to and including day only)
        bench = frames[bench_sym].iloc[max(0, i + 1 - 220): i + 1]      # windows used are <=100 bars
        regime = compute_regime(bench, settings.regime)
        if regime.regime in ("unclear", "risk_off"):
            continue
        feats = {}
        for sym in syms:
            df = frames[sym]
            if day not in df.index:
                continue
            f = compute_features(df.loc[:day].iloc[-220:], st.momentum, bench, st.relative_strength.rs_lookback)
            if f:
                feats[sym] = f
        if len(feats) >= 8:
            rank = pd.Series({s: f["rs"] for s, f in feats.items()}).rank(pct=True)
        else:
            rank = None
        for sym, f in feats.items():
            if sym in open_pos or f["avg_dollar_volume"] < rcfg.min_avg_dollar_volume:
                continue
            setups = []
            if st.momentum.enabled:
                setups.append(momentum_breakout(sym, f, f["close"], st.momentum))
            if use_rs and st.relative_strength.enabled:
                setups += [s for s in relative_strength(sym, f, f["close"], st.relative_strength, False)
                           if s.direction == Direction.LONG]
            for su in setups:
                if not su.ok:
                    continue
                comps = components(f, su.direction, f["close"], None if rank is None else float(rank[sym]),
                                   regime.score, None, f["high_n"])
                if total_score(comps, settings.scoring.weights) >= settings.scoring.min_score:
                    rr = abs(su.target - su.entry) / abs(su.entry - su.stop)
                    if rr >= rcfg.min_reward_risk:
                        pending.append({"symbol": sym, "stop": su.stop, "target": su.target, "strategy": su.strategy})
                    break
    for sym, p in open_pos.items():       # mark-to-market, left open
        pass
    res.end_equity = equity
    bf = frames[bench_sym]
    sub = bf.iloc[warm:len(dates) - 1]
    res.benchmark_return_pct = float((sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100) if len(sub) > 1 else None
    eq = np.array([e for _, e in res.equity_curve]) if res.equity_curve else np.array([start_equity])
    peak = np.maximum.accumulate(eq)
    pnl = np.array([t.pnl for t in res.trades]) if res.trades else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    res.metrics = {"total_return_pct": (equity / start_equity - 1) * 100, "max_drawdown_pct": float(((peak - eq) / peak).max() * 100),
                   "n_trades": len(res.trades), "win_rate": float(len(wins) / len(pnl)) if len(pnl) else None,
                   "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
                   "open_positions_at_end": len(open_pos)}
    return res
