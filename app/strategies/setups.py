"""Three strategies. Each returns a Setup (entry/stop/target) or a list of no-trade reasons."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import CatalystCfg, MomentumCfg, RelStrengthCfg
from app.schemas import Direction, SymbolSentiment


@dataclass
class Setup:
    symbol: str
    strategy: str
    direction: Direction
    horizon: str
    entry: float
    stop: float
    target: float
    reasons: list[str] = field(default_factory=list)       # why it qualified
    blockers: list[str] = field(default_factory=list)      # why it did not
    catalyst_news_id: str | None = None

    @property
    def ok(self) -> bool:
        return not self.blockers


def _levels(price: float, atr: float, stop_mult: float, r_mult: float, direction: Direction):
    d = stop_mult * atr
    if direction == Direction.LONG:
        return price - d, price + r_mult * d
    return price + d, price - r_mult * d


def momentum_breakout(sym: str, f: dict, price: float, cfg: MomentumCfg) -> Setup:
    """Trend-aligned breakout above the prior N-bar high with volume expansion; no chasing extended moves."""
    s = Setup(sym, "momentum", Direction.LONG, cfg.horizon, price, 0, 0)
    if not (f["close"] > f["sma_slow"] > f["sma_trend"]):
        s.blockers.append("trend filter failed (close>SMA50>SMA100)")
    if price <= f["sma_fast"]:
        s.blockers.append("price below fast SMA")
    broke = max(f["close"], price) > f["high_n"]
    if not broke:
        s.blockers.append("no breakout above prior N-bar high")
    ext = (price - max(f["high_n"], f["close"])) / f["atr"]
    if broke and ext > cfg.max_chase_atr:
        s.blockers.append(f"extended {ext:.2f} ATR beyond breakout (chase limit {cfg.max_chase_atr})")
    if f["vol_ratio"] < cfg.min_volume_ratio:
        s.blockers.append(f"volume ratio {f['vol_ratio']:.2f} < {cfg.min_volume_ratio}")
    if not (cfg.min_atr_pct <= f["atr_pct"] <= cfg.max_atr_pct):
        s.blockers.append(f"ATR% {f['atr_pct']:.2%} outside [{cfg.min_atr_pct:.0%},{cfg.max_atr_pct:.0%}]")
    s.stop, s.target = _levels(price, f["atr"], cfg.stop_atr_multiple, cfg.target_r_multiple, Direction.LONG)
    if not s.blockers:
        s.reasons += [f"breakout above {f['high_n']:.2f}", f"volume x{f['vol_ratio']:.2f}", "trend aligned"]
    return s


def news_catalyst(sym: str, f: dict, price: float, sent: SymbolSentiment | None, cfg: CatalystCfg) -> Setup:
    """Positive, material, novel, fresh news CONFIRMED by price move and volume. News alone never qualifies."""
    s = Setup(sym, "catalyst", Direction.LONG, cfg.horizon, price, 0, 0)
    if sent is None or sent.n_items == 0:
        s.blockers.append("no sentiment data for symbol")
    else:
        s.catalyst_news_id = sent.top_news_id
        if sent.stale:
            s.blockers.append("news stale")
        if sent.conflicting:
            s.blockers.append("conflicting news")
        if sent.score < cfg.min_decayed_sentiment:
            s.blockers.append(f"decayed sentiment {sent.score:.2f} < {cfg.min_decayed_sentiment}")
        if sent.max_materiality < cfg.min_materiality:
            s.blockers.append("materiality below threshold")
        if sent.min_novelty_of_top < cfg.min_novelty:
            s.blockers.append("story not novel")
    move = price / f["close"] - 1
    if move < cfg.min_price_move_pct:
        s.blockers.append(f"price confirmation missing (move {move:.2%} < {cfg.min_price_move_pct:.0%})")
    if f["vol_ratio"] < cfg.min_volume_ratio:
        s.blockers.append(f"volume confirmation missing (ratio {f['vol_ratio']:.2f} < {cfg.min_volume_ratio})")
    s.stop, s.target = _levels(price, f["atr"], cfg.stop_atr_multiple, cfg.target_r_multiple, Direction.LONG)
    if not s.blockers and sent:
        s.reasons += [f"sentiment {sent.score:.2f} confirmed by price {move:.2%} and volume x{f['vol_ratio']:.2f}"]
    return s


def relative_strength(sym: str, f: dict, price: float, cfg: RelStrengthCfg, allow_short: bool) -> list[Setup]:
    """Optional: long RS continuation near highs; short breakdown continuation (short signals off by default)."""
    out: list[Setup] = []
    lg = Setup(sym, "relative_strength", Direction.LONG, cfg.horizon, price, 0, 0)
    if f["rs"] < cfg.min_rs_long:
        lg.blockers.append(f"RS {f['rs']:.2%} < {cfg.min_rs_long:.0%}")
    if not (price > f["sma_fast"] > f["sma_slow"]):
        lg.blockers.append("not in uptrend (price>SMA20>SMA50)")
    if (f["high_50"] - price) / f["high_50"] > 0.03:
        lg.blockers.append("more than 3% below 50-bar high")
    lg.stop, lg.target = _levels(price, f["atr"], cfg.stop_atr_multiple, cfg.target_r_multiple, Direction.LONG)
    if lg.ok:
        lg.reasons.append(f"RS {f['rs']:.2%} vs benchmark near highs")
    out.append(lg)
    if cfg.allow_short_signals:
        sh = Setup(sym, "relative_strength", Direction.SHORT, cfg.horizon, price, 0, 0)
        if not allow_short:
            sh.blockers.append("shorting disabled (ALLOW_SHORTING=false)")
        if f["rs"] > -cfg.min_rs_long:
            sh.blockers.append("relative weakness insufficient")
        if min(f["close"], price) >= f["low_n"]:
            sh.blockers.append("no breakdown below prior N-bar low")
        if not (price < f["sma_slow"]):
            sh.blockers.append("price above SMA50")
        if f["vol_ratio"] < cfg.min_volume_ratio:
            sh.blockers.append("volume ratio too low")
        sh.stop, sh.target = _levels(price, f["atr"], cfg.stop_atr_multiple, cfg.target_r_multiple, Direction.SHORT)
        if sh.ok:
            sh.reasons.append("breakdown continuation with relative weakness")
        out.append(sh)
    return out
