"""Deterministic risk engine. Ordinary Python; no LLM can change or bypass it at runtime.
Stale data, unreconciled broker state, breakers, expiry, duplicates and short restrictions block ENTRIES."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.config import RiskCfg, SessionCfg, Settings
from app.schemas import (AccountSnapshot, Direction, Horizon, OrderRecord, Position, RiskDecision, SYMBOL_RE,
                         TradeProposal)
from app.timeutil import SessionStatus


@dataclass
class RiskContext:
    now: datetime
    account: AccountSnapshot
    positions: list[Position]
    open_orders: list[OrderRecord]
    session: SessionStatus
    quote_fresh: bool
    bars_fresh: bool
    broker_reconciled: bool
    peak_equity: float
    entries_today: int = 0
    shortable: bool | None = None
    operator_halt: bool = False
    already_decided: bool = False
    position_horizons: dict[str, str] = field(default_factory=dict)   # symbol -> horizon of open positions


def breaker_state(cfg: RiskCfg, account: AccountSnapshot, peak_equity: float) -> dict:
    """Daily-loss circuit breaker uses broker 'last_equity' (prior close) as start-of-day equity."""
    sod = account.last_equity if account.last_equity > 0 else account.equity
    daily = (account.equity - sod) / sod
    dd = (peak_equity - account.equity) / peak_equity if peak_equity > 0 else 0.0
    return {"daily_pnl_pct": daily, "drawdown_pct": dd,
            "daily_loss_tripped": daily <= -cfg.daily_loss_limit_pct,
            "drawdown_review": dd >= cfg.drawdown_review_pct}


def size_position(cfg: RiskCfg, equity: float, entry: float, stop: float, direction: Direction,
                  buying_power: float, gross_exposure: float, avg_volume: float = 0.0,
                  horizon_notional: float = 0.0, horizon: Horizon = Horizon.SWING,
                  group_headroom: float | None = None) -> tuple[int, dict[str, int]]:
    """Return (qty, per-constraint caps). qty = min of all caps, floored; 0 means not tradable."""
    dist = abs(entry - stop)
    if equity <= 0 or entry <= 0 or dist <= 0:
        return 0, {"invalid_inputs": 0}
    caps = {
        "risk_budget": math.floor(equity * cfg.risk_per_trade_pct / dist),
        "position_cap": math.floor(equity * cfg.max_position_pct / entry),
        "gross_exposure": math.floor(max(equity * cfg.max_gross_exposure_pct - gross_exposure, 0) / entry),
        "buying_power": math.floor(max(buying_power * (1 - cfg.buying_power_buffer_pct), 0) / entry),
        "envelope": math.floor(max(getattr(cfg.envelopes, horizon.value) - horizon_notional, 0) / entry),
    }
    if avg_volume > 0:
        caps["liquidity"] = math.floor(avg_volume * cfg.max_adv_participation)
    if group_headroom is not None:
        caps["correlated_group"] = math.floor(max(group_headroom, 0) / entry)
    return max(min(caps.values()), 0), caps


class RiskEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cfg: RiskCfg = settings.risk
        self.sess: SessionCfg = settings.session

    def _group_headroom(self, symbol: str, positions: list[Position], equity: float) -> float | None:
        for members in self.s.universe.correlation_groups.values():
            if symbol in members:
                used = sum(abs(p.market_value) for p in positions if p.symbol in members)
                return equity * self.cfg.max_group_exposure_pct - used
        return None

    def size_for(self, direction: Direction, entry: float, stop: float, horizon: Horizon, symbol: str,
                 avg_volume: float, account: AccountSnapshot, positions: list[Position],
                 position_horizons: dict[str, str]) -> tuple[int, dict[str, int]]:
        gross = sum(abs(x.market_value) for x in positions)
        hz = sum(abs(x.market_value) for x in positions if position_horizons.get(x.symbol, "swing") == horizon.value)
        return size_position(self.cfg, account.equity, entry, stop, direction, account.buying_power, gross,
                             avg_volume, hz, horizon, self._group_headroom(symbol, positions, account.equity))

    def evaluate(self, p: TradeProposal, ctx: RiskContext) -> RiskDecision:
        cfg, checks, reasons = self.cfg, {}, []

        def chk(name: str, ok: bool, why: str) -> None:
            checks[name] = bool(ok)
            if not ok:
                reasons.append(f"{name}: {why}")

        eq = ctx.account.equity
        # structural validity
        valid = (bool(SYMBOL_RE.match(p.symbol)) and p.qty > 0 and p.entry_price > 0 and p.stop_price > 0
                 and p.target_price > 0)
        if valid and p.direction == Direction.LONG:
            valid = p.stop_price < p.entry_price < p.target_price
        elif valid:
            valid = p.target_price < p.entry_price < p.stop_price
        chk("proposal_valid", valid, "symbol/qty/price/stop/target invalid or on wrong side")
        chk("proposal_not_expired", p.expires_at > ctx.now and p.created_at <= ctx.now + timedelta(seconds=5),
            "proposal expired or timestamp invalid")
        chk("not_already_decided", not ctx.already_decided, "proposal already processed (duplicate)")
        held = {x.symbol for x in ctx.positions}
        working = {o.symbol for o in ctx.open_orders}
        chk("no_duplicate_or_conflict", p.symbol not in held and p.symbol not in working,
            "position or working order already exists for symbol")
        chk("data_fresh", ctx.quote_fresh and ctx.bars_fresh, "critical market data stale/missing")
        chk("broker_reconciled", ctx.broker_reconciled, "broker state not reconciled")
        chk("account_healthy", eq > 0 and ctx.account.buying_power > 0, "equity/buying power unavailable")
        chk("operator_not_halted", not ctx.operator_halt, "operator halt active")
        bs = breaker_state(cfg, ctx.account, ctx.peak_equity)
        chk("daily_loss_breaker", not bs["daily_loss_tripped"],
            f"daily P&L {bs['daily_pnl_pct']:.2%} breached -{cfg.daily_loss_limit_pct:.0%}")
        chk("drawdown_review", not bs["drawdown_review"],
            f"drawdown {bs['drawdown_pct']:.2%} >= review trigger {cfg.drawdown_review_pct:.0%} (operator review)")
        st = ctx.session
        in_session = st.is_open
        chk("market_open", in_session, "market closed")
        if in_session and st.minutes_since_open is not None and st.minutes_to_close is not None:
            chk("session_window", st.minutes_since_open >= self.sess.entry_blackout_open_minutes
                and st.minutes_to_close > self.sess.entry_blackout_close_minutes,
                "inside open/close entry blackout")
            if p.horizon == Horizon.INTRADAY:
                chk("intraday_time_left", st.minutes_to_close > self.sess.intraday_flatten_minutes_before_close + 15,
                    "insufficient time left for intraday trade")
        if p.direction == Direction.SHORT:
            chk("short_permitted", self.s.allow_shorting and ctx.shortable is True,
                "shorting disabled by config or asset not confirmed shortable")
        chk("max_positions", len(ctx.positions) < cfg.max_positions, "max open positions reached")
        chk("max_entries_today", ctx.entries_today < cfg.max_new_entries_per_day, "daily entry limit reached")
        chk("liquidity", p.avg_dollar_volume >= cfg.min_avg_dollar_volume,
            f"avg dollar volume {p.avg_dollar_volume:,.0f} below minimum")
        risk_per_share = abs(p.entry_price - p.stop_price)
        reward = abs(p.target_price - p.entry_price)
        chk("reward_risk", risk_per_share > 0 and reward / risk_per_share >= cfg.min_reward_risk,
            f"reward/risk below {cfg.min_reward_risk}")

        gross = sum(abs(x.market_value) for x in ctx.positions)
        hz_notional = sum(abs(x.market_value) for x in ctx.positions
                          if ctx.position_horizons.get(x.symbol, "swing") == p.horizon.value)
        qty_cap, caps = size_position(
            cfg, eq, p.entry_price, p.stop_price, p.direction, ctx.account.buying_power, gross, p.avg_volume,
            hz_notional, p.horizon, self._group_headroom(p.symbol, ctx.positions, eq)) if valid else (0, {})
        approved_qty = min(p.qty, qty_cap)
        chk("size_nonzero", approved_qty >= 1, f"sizing caps leave no tradable quantity {caps}")
        approved = all(checks.values())
        return RiskDecision(
            proposal_id=p.id, approved=approved, approved_qty=approved_qty if approved else 0,
            reasons=reasons if not approved else (["approved"] + (
                [f"resized {p.qty}->{approved_qty} by caps {caps}"] if approved_qty < p.qty else [])),
            checks=checks, decided_at=ctx.now,
            limits_snapshot={"risk_per_trade_pct": cfg.risk_per_trade_pct, "max_position_pct": cfg.max_position_pct,
                             "max_gross_exposure_pct": cfg.max_gross_exposure_pct,
                             "daily_pnl_pct": bs["daily_pnl_pct"], "drawdown_pct": bs["drawdown_pct"]})
