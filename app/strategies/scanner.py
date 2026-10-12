"""Numeric-first universe scanner: features -> setups -> scores -> signals (+ explicit no-trade reasons)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from app.config import Settings
from app.data.providers import check_bar_freshness, check_quote_freshness
from app.schemas import Candidate, Direction, Horizon, Signal, SymbolSentiment, new_id
from app.strategies.features import compute_features
from app.strategies.regime import RegimeState, compute_regime
from app.strategies.scoring import components, total_score
from app.strategies.setups import Setup, momentum_breakout, news_catalyst, relative_strength
from app.timeutil import last_completed_session


@dataclass
class ScanResult:
    scan_id: str
    ts: datetime
    regime: RegimeState
    candidates: list[Candidate] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    no_trade: list[dict] = field(default_factory=list)       # {symbol, strategy, reasons}
    features: dict[str, dict] = field(default_factory=dict)
    blocked_reason: str | None = None
    universe_size: int = 0


def _level_for(setup: Setup, f: dict) -> float:
    return f["low_n"] if setup.direction == Direction.SHORT else f["high_n"]


def scan(settings: Settings, bars: dict[str, pd.DataFrame], quotes: dict[str, dict],
         sentiments: dict[str, SymbolSentiment], now: datetime, require_fresh_quotes: bool = True) -> ScanResult:
    st, uni = settings.strategies, settings.universe.symbols
    bench_sym = settings.data.benchmark_symbol
    bench = bars.get(bench_sym)
    regime = compute_regime(bench, settings.regime)
    res = ScanResult(new_id("scan"), now, regime, universe_size=len(uni))
    if regime.regime == "unclear":
        res.blocked_reason = "benchmark data unavailable: new entries blocked"
        return res
    lcs = last_completed_session(now)

    feats: dict[str, dict] = {}
    for sym in uni:
        df = bars.get(sym)
        ok, why = check_bar_freshness(df, lcs, settings.data.max_bar_age_sessions)
        if not ok:
            res.no_trade.append({"symbol": sym, "strategy": "*", "reasons": [f"bars: {why}"]})
            continue
        f = compute_features(df, st.momentum, bench, st.relative_strength.rs_lookback)
        if f is None:
            res.no_trade.append({"symbol": sym, "strategy": "*", "reasons": ["insufficient/invalid history"]})
            continue
        feats[sym] = f
    res.features = feats
    if not feats:
        res.blocked_reason = "no symbols with usable data"
        return res

    rs_series = pd.Series({s: f["rs"] for s, f in feats.items()})
    rs_rank = rs_series.rank(pct=True) if len(rs_series) >= 8 else None

    best: dict[str, tuple[float, Setup, dict]] = {}
    for sym, f in feats.items():
        q = quotes.get(sym)
        qok, qwhy = check_quote_freshness(q, now, settings.data.max_quote_age_seconds)
        if qok:
            price = float(q["mid"])
        elif require_fresh_quotes:
            res.no_trade.append({"symbol": sym, "strategy": "*", "reasons": [f"quote: {qwhy}"]})
            continue
        else:
            price = f["close"]
        if abs(price / f["close"] - 1) > 0.25:
            res.no_trade.append({"symbol": sym, "strategy": "*", "reasons": ["price vs last close implausible (>25%)"]})
            continue
        setups: list[Setup] = []
        sent = sentiments.get(sym)
        if st.momentum.enabled:
            setups.append(momentum_breakout(sym, f, price, st.momentum))
        if st.catalyst.enabled:
            setups.append(news_catalyst(sym, f, price, sent, st.catalyst))
        if st.relative_strength.enabled:
            setups += relative_strength(sym, f, price, st.relative_strength, settings.allow_shorting)
        for su in setups:
            if not su.ok:
                res.no_trade.append({"symbol": sym, "strategy": su.strategy, "reasons": su.blockers})
                continue
            comps = components(f, su.direction, price, None if rs_rank is None else float(rs_rank[sym]),
                               regime.score, sent, _level_for(su, f))
            score = total_score(comps, settings.scoring.weights)
            cand = Candidate(symbol=sym, strategy=su.strategy, direction=su.direction, score=score,
                             components={k: round(v, 4) for k, v in comps.items() if v is not None},
                             features={k: round(float(v), 6) for k, v in f.items()}, reasons=su.reasons)
            res.candidates.append(cand)
            if score < settings.scoring.min_score:
                res.no_trade.append({"symbol": sym, "strategy": su.strategy,
                                     "reasons": [f"score {score:.2f} < {settings.scoring.min_score}"]})
                continue
            if su.direction == Direction.LONG and regime.regime == "risk_off" and settings.regime.block_longs_in_risk_off:
                res.no_trade.append({"symbol": sym, "strategy": su.strategy, "reasons": ["risk_off regime blocks longs"]})
                continue
            if su.direction == Direction.SHORT and not settings.allow_shorting:
                res.no_trade.append({"symbol": sym, "strategy": su.strategy, "reasons": ["shorting disabled"]})
                continue
            if sym not in best or score > best[sym][0]:
                best[sym] = (score, su, comps)

    ttl = timedelta(minutes=settings.risk.proposal_ttl_minutes)
    for sym, (score, su, comps) in best.items():
        f = feats[sym]
        res.signals.append(Signal(
            symbol=sym, strategy=su.strategy, direction=su.direction, horizon=Horizon(su.horizon),
            entry_price=round(su.entry, 4), stop_price=round(su.stop, 4), target_price=round(su.target, 4),
            score=score, components={k: round(v, 4) for k, v in comps.items() if v is not None},
            created_at=now, expires_at=now + ttl, reasons=su.reasons, catalyst_news_id=su.catalyst_news_id,
            avg_dollar_volume=f["avg_dollar_volume"], avg_volume=f["avg_volume"]))
    res.signals.sort(key=lambda s: -s.score)
    res.candidates.sort(key=lambda c: -c.score)
    if not res.signals:
        res.blocked_reason = "no qualifying signals (no-trade is a valid outcome)"
    return res
