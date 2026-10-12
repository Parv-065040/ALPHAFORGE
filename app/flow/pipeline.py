"""Predictable orchestration: data -> sentiment -> scan -> agents -> proposal -> RISK -> execution.
Plain-Python flow (authoritative). app/flow/crewai_flow.py wraps the same steps in a CrewAI Flow if installed."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.agents.agents import CatalystAnalyst, PortfolioManager, RegimeAnalyst
from app.agents.llm import LLMClient
from app.config import Settings
from app.data.http import ProviderError
from app.data.providers import MarketDataProvider, NewsProvider, check_quote_freshness
from app.execution.service import ExecutionResult, ExecutionService
from app.risk.engine import RiskContext, RiskEngine, breaker_state
from app.schemas import (CatalystAssessment, OrderStatus, Position, RegimeAssessment, RiskDecision, SentimentResult, Signal, SymbolSentiment,
                         TradeProposal, new_id)
from app.sentiment.pipeline import Scorer, aggregate, analyze
from app.storage.db import Storage
from app.strategies.scanner import ScanResult, scan
from app.timeutil import ET, UTC, last_completed_session, session_status, utcnow
from datetime import datetime as _dt

log = logging.getLogger("alphaforge.flow")
SHORTLIST = 5


@dataclass
class CycleState:
    now: datetime
    dry_run: bool
    bars: dict = field(default_factory=dict)
    quotes: dict = field(default_factory=dict)
    sentiments: dict[str, SymbolSentiment] = field(default_factory=dict)
    sent_results: list[SentimentResult] = field(default_factory=list)
    scan: ScanResult | None = None
    proposals: list[TradeProposal] = field(default_factory=list)
    decisions: list[RiskDecision] = field(default_factory=list)
    executions: list[tuple[str, ExecutionResult]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked_reason: str | None = None


def drop_incomplete_bars(bars: dict, now: datetime) -> dict:
    """Remove any daily bar newer than the last COMPLETED session (e.g. today's partial bar) - no look-ahead."""
    lcs = last_completed_session(now)
    return {k: v[[d.date() <= lcs for d in v.index]] for k, v in bars.items()}


class TradingFlow:
    def __init__(self, settings: Settings, db: Storage, market: MarketDataProvider, news: NewsProvider | None,
                 execution: ExecutionService | None, scorer: Scorer, llm: LLMClient | None = None,
                 now_fn=utcnow):
        self.s, self.db, self.market, self.news, self.exec, self.scorer = settings, db, market, news, execution, scorer
        self.risk = RiskEngine(settings)
        self.now_fn = now_fn
        mr = settings.llm.max_retries
        self.catalyst_agent = CatalystAnalyst(llm, db, mr)
        self.regime_agent = RegimeAnalyst(llm, db, mr)
        self.pm_agent = PortfolioManager(llm, db, mr)

    # ---- steps ----
    def step_data(self, st: CycleState) -> None:
        syms = sorted(set(self.s.universe.symbols) | {self.s.data.benchmark_symbol})
        start = st.now - timedelta(days=self.s.data.daily_lookback_days)
        try:
            t0 = utcnow()
            st.bars = self.market.daily_bars(syms, start, st.now)
            st.quotes = self.market.latest_quotes(syms)
            self.db.log_provider(self.market.name, True, (utcnow() - t0).total_seconds() * 1e3)
        except ProviderError as e:
            self.db.log_provider(self.market.name, False, 0, str(e))
            self.db.alert("error", "DATA_UNAVAILABLE", f"market data: {e}")
            st.blocked_reason = f"market data unavailable: {e}"
            return
        st.bars = drop_incomplete_bars(st.bars, st.now)
        self._sentiment(st)

    def _sentiment(self, st: CycleState) -> None:
        if self.news is None:
            st.warnings.append("news provider not configured: catalyst strategy has no input")
            return
        try:
            t0 = utcnow()
            since = st.now - timedelta(hours=self.s.data.news_lookback_hours)
            items = self.news.news(self.s.universe.symbols, since)
            self.db.log_provider(self.news.name, True, (utcnow() - t0).total_seconds() * 1e3)
        except ProviderError as e:
            self.db.log_provider(self.news.name, False, 0, str(e))
            st.warnings.append(f"news unavailable: {e}")
            items = []
        known_ids = {r["id"] for r in self.db.q("SELECT id FROM news WHERE published_at>=?", (since.isoformat(),))} \
            if items else set()
        fresh = [i for i in items if i.id not in known_ids]
        if fresh:
            results, dd = analyze(fresh, set(self.s.universe.symbols), self.scorer, self.s.sentiment, st.now,
                                  self.db.known_news_hashes(since))
            self.db.save_news(dd.unique, dd.hashes)
            self.db.save_sentiment(results)
        stored = self.db.recent_sentiment(self.s.data.news_lookback_hours, now=st.now)
        st.sent_results = stored
        st.sentiments = aggregate(stored, self.s.sentiment, st.now)

    def step_scan(self, st: CycleState) -> None:
        if st.blocked_reason:
            return
        sess = session_status(st.now)
        st.scan = scan(self.s, st.bars, st.quotes, st.sentiments, st.now, require_fresh_quotes=sess.is_open)
        r = st.scan
        self.db.save_scan(r.scan_id, st.now, r.universe_size, r.candidates, r.signals, r.regime.regime,
                          r.blocked_reason, r.features,
                          r.no_trade, r.regime.reasons)
        if r.blocked_reason and not r.signals:
            st.blocked_reason = r.blocked_reason

    def _working_exposure(self, extra: list[Position]) -> tuple[list[Position], dict[str, str]]:
        """Working (unfilled) entry orders and proposals approved earlier this cycle count as exposure."""
        pseudo, hz = list(extra), {}
        for o in self.db.open_orders():
            if o.proposal_id and o.status in (OrderStatus.PENDING_SUBMIT, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                prop = self.db.get_proposal(o.proposal_id)
                if prop:
                    rem = max(o.qty - o.filled_qty, 0)
                    sign = 1 if prop.direction.value == "long" else -1
                    pseudo.append(Position(symbol=o.symbol, qty=sign * rem, avg_entry_price=prop.entry_price,
                                           market_value=sign * rem * prop.entry_price, current_price=prop.entry_price))
                    hz[o.symbol] = prop.horizon.value
        return pseudo, hz

    def _context(self, st: CycleState, sig_symbol: str, proposal_id: str,
                 reserved: list[Position] | None = None, reserved_horizons: dict[str, str] | None = None) -> RiskContext:
        acct_hist = self.db.account_history(limit=2000)
        account = acct_hist[-1]
        day_start = datetime.combine(st.now.astimezone(ET).date(), datetime.min.time(), tzinfo=ET).astimezone(UTC)
        q = st.quotes.get(sig_symbol)
        qok, _ = check_quote_freshness(q, st.now, self.s.data.max_quote_age_seconds)
        shortable = None
        pseudo, hz = self._working_exposure(reserved or [])
        horizons = {k: v["horizon"] for k, v in self.db.position_meta().items()} | hz | (reserved_horizons or {})
        return RiskContext(
            now=st.now, account=account, positions=self.db.latest_positions() + pseudo, open_orders=self.db.open_orders(),
            session=session_status(st.now), quote_fresh=qok, bars_fresh=True,
            broker_reconciled=self.exec.is_reconciled() if self.exec else False,
            peak_equity=max(h.equity for h in acct_hist), entries_today=self.db.entries_today(day_start) + len(reserved or []),
            shortable=shortable, operator_halt=bool(self.db.halt_state().get("halted")),
            already_decided=self.db.get_decision(proposal_id) is not None,
            position_horizons=horizons)

    def step_propose_and_risk(self, st: CycleState) -> None:
        valid_no_trade = (
            st.scan is not None
            and not st.scan.signals
            and st.scan.blocked_reason == "no qualifying signals (no-trade is a valid outcome)"
        )
        if st.scan is None or (st.blocked_reason and not valid_no_trade):
            return
        if self.exec is None:
            st.warnings.append("no broker configured: signals recorded, no proposals/risk decisions")
            return
        rep = self.exec.reconcile()
        if not rep.ok:
            st.warnings += [f"reconcile: {i}" for i in rep.issues]
        if not self.db.account_history(limit=1):
            st.warnings.append("no account snapshot available")
            return
        acct = self.db.account_history(limit=1)[-1]
        bs = breaker_state(self.s.risk, acct, max(h.equity for h in self.db.account_history(limit=2000)))
        if bs["daily_loss_tripped"] or bs["drawdown_review"]:
            self.db.alert("critical", "CIRCUIT_BREAKER", f"daily {bs['daily_pnl_pct']:.2%}, drawdown {bs['drawdown_pct']:.2%}: "
                          "new entries blocked; existing positions keep broker-side stops")
            if self.s.risk.existing_positions_policy == "flatten_intraday":
                self.exec.flatten_intraday(force=True)
        regime_view, _ = self.regime_agent.assess(st.scan.regime)
        reserved: list[Position] = []
        reserved_hz: dict[str, str] = {}
        for sig in st.scan.signals[:SHORTLIST]:
            cat_view = None
            if sig.strategy == "catalyst" or sig.catalyst_news_id:
                items = [r for r in st.sent_results if r.symbol == sig.symbol]
                cat_view, _ = self.catalyst_agent.assess(sig.symbol, items, st.sentiments.get(sig.symbol))
            pm, _ = self.pm_agent.decide(sig, cat_view, regime_view)
            if pm.action == "skip" or pm.size_multiplier <= 0:
                st.scan.no_trade.append({"symbol": sig.symbol, "strategy": sig.strategy,
                                         "reasons": [f"portfolio manager skip: {pm.rationale[:120]}"]})
                continue
            tmp_id = new_id("prop")
            ctx = self._context(st, sig.symbol, tmp_id, reserved, reserved_hz)
            if sig.direction.value == "short" and self.exec and hasattr(self.exec.broker, "get_asset"):
                a = self.exec.broker.get_asset(sig.symbol)
                ctx.shortable = bool(a and a.get("shortable") and a.get("tradable"))
            qty, _caps = self.risk.size_for(sig.direction, sig.entry_price, sig.stop_price, sig.horizon, sig.symbol,
                                            sig.avg_volume, ctx.account, ctx.positions, ctx.position_horizons)
            qty = math.floor(qty * pm.size_multiplier)
            if qty < 1:
                st.scan.no_trade.append({"symbol": sig.symbol, "strategy": sig.strategy, "reasons": ["sizing yields 0 shares"]})
                continue
            prop = TradeProposal(id=tmp_id, signal_id=sig.id, symbol=sig.symbol, direction=sig.direction,
                                 strategy=sig.strategy, horizon=sig.horizon, qty=qty, entry_price=sig.entry_price,
                                 stop_price=sig.stop_price, target_price=sig.target_price, created_at=st.now,
                                 expires_at=sig.expires_at, rationale=pm.rationale,
                                 source="agent_pm" if not self.pm_agent.client is None else "quant_only",
                                 agent_confidence=pm.confidence, avg_dollar_volume=sig.avg_dollar_volume,
                                 avg_volume=sig.avg_volume)
            self.db.save_proposal(prop)
            dec = self.risk.evaluate(prop, ctx)
            self.db.save_decision(dec)
            st.proposals.append(prop)
            st.decisions.append(dec)
            if dec.approved:        # reserve exposure so later signals in this cycle see it
                sgn = 1 if prop.direction.value == "long" else -1
                reserved.append(Position(symbol=prop.symbol, qty=sgn * dec.approved_qty, avg_entry_price=prop.entry_price,
                                         market_value=sgn * dec.approved_qty * prop.entry_price,
                                         current_price=prop.entry_price))
                reserved_hz[prop.symbol] = prop.horizon.value

    def step_execute(self, st: CycleState) -> None:
        if self.exec is None:
            return
        for prop, dec in zip(st.proposals, st.decisions):
            if not dec.approved:
                continue
            if st.dry_run:
                st.executions.append((prop.id, ExecutionResult("disabled", "dry-run: no orders submitted")))
                continue
            st.executions.append((prop.id, self.exec.execute(prop, dec)))
        if not st.dry_run:
            self.exec.cancel_stale_entries()
            self.exec.flatten_intraday()

    # ---- full cycle ----
    def run_cycle(self, dry_run: bool = True) -> CycleState:
        st = CycleState(now=self.now_fn(), dry_run=dry_run)
        self.db.record_config(self.s.config_hash(), self.s.safe_dict())
        self.step_data(st)
        self.step_scan(st)
        self.step_propose_and_risk(st)
        self.step_execute(st)
        return st
