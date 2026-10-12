"""data -> sentiment -> signal -> agents -> risk -> execution adapter -> storage, all on fixtures/mocks."""
from datetime import datetime, timedelta

import pytest

from app.data.http import ProviderError
from app.data.providers import DemoMarketData, DemoNews, synthetic_daily_bars
from app.execution.service import ExecutionService
from app.flow.pipeline import TradingFlow
from app.schemas import NewsItem
from app.sentiment.pipeline import RuleBasedScorer
from app.storage.db import Storage
from app.timeutil import UTC
from tests.conftest import MARKET_NOW, make_settings
from tests.fakes import FakeBroker
from tests.unit.test_schemas_and_agents import FakeLLM

LAST_SESSION = datetime(2026, 3, 6, 21, 0, tzinfo=UTC)


def frames(breakout=True, downtrend=False):
    s = make_settings()
    fr = {sym: synthetic_daily_bars(sym, 300, 1, end=LAST_SESSION) for sym in s.universe.symbols}
    if downtrend:
        for df in fr.values():
            import numpy as np
            f = np.linspace(1.5, 0.7, len(df))
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] * f
    elif breakout:
        a = fr["AAPL"]
        a.iloc[-1, a.columns.get_loc("close")] = a["high"].iloc[-21:-1].max() * 1.01
        a.iloc[-1, a.columns.get_loc("high")] = a["close"].iloc[-1] * 1.003
        a.iloc[-1, a.columns.get_loc("volume")] = a["volume"].mean() * 2.6
    return fr


class BrokenMarket:
    name = "broken"
    def daily_bars(self, *a, **k): raise ProviderError("HTTP 503", 503)
    def latest_quotes(self, *a, **k): raise ProviderError("HTTP 503", 503)


def build(paper=True, fb=None, market=None, quote_time=MARKET_NOW, llm=None, news=None, fr=None):
    s = make_settings(ENABLE_PAPER_TRADING="true" if paper else "false")
    db = Storage(":memory:")
    fb = fb or FakeBroker()
    ex = ExecutionService(s, fb, db, now_fn=lambda: MARKET_NOW)
    market = market or DemoMarketData(fr or frames(), quote_time)
    flow = TradingFlow(s, db, market, news if news is not None else DemoNews([]), ex, RuleBasedScorer(), llm,
                       now_fn=lambda: MARKET_NOW)
    return s, db, fb, ex, flow


def test_dry_run_records_everything_but_submits_nothing():
    s, db, fb, ex, flow = build()
    st = flow.run_cycle(dry_run=True)
    assert [g.symbol for g in st.scan.signals][:1] == ["AAPL"]
    assert st.proposals and st.decisions[0].approved and fb.submit_calls == 0
    assert all(r.status == "disabled" for _, r in st.executions)
    for table in ("scans", "signals", "proposals", "risk_decisions", "market_features", "account_snapshots"):
        assert db.q(f"SELECT COUNT(*) n FROM {table}")[0]["n"] >= 1, table
    assert db.q("SELECT COUNT(*) n FROM orders")[0]["n"] == 0
    assert db.q("SELECT COUNT(*) n FROM config_versions")[0]["n"] == 1
    assert db.q("SELECT COUNT(*) n FROM agent_runs")[0]["n"] >= 2                       # regime + PM logged


def test_paper_cycle_submits_risk_sized_order_through_execution_adapter():
    s, db, fb, ex, flow = build()
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls >= 1
    pl = next(x for x in fb.payloads if x["symbol"] == "AAPL")
    assert pl["order_class"] == "bracket"
    qty, stop, limit = int(pl["qty"]), float(pl["stop_loss"]["stop_price"]), float(pl["limit_price"])
    assert qty * (limit - stop) <= 100_000 * s.risk.risk_per_trade_pct * 1.03          # rounding slack
    assert qty * limit <= 100_000 * s.risk.max_position_pct * 1.01
    o = db.get_order(pl["client_order_id"])
    assert o.proposal_id in {p.id for p in st.proposals} and db.get_decision(o.proposal_id).approved


def test_enable_flag_off_means_no_orders_even_when_not_dry_run():
    s, db, fb, ex, flow = build(paper=False)
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0 and st.executions and all(r.status == "disabled" for _, r in st.executions)


def test_second_cycle_does_not_duplicate_position_or_order():
    s, db, fb, ex, flow = build()
    flow.run_cycle(dry_run=False)
    n1 = fb.submit_calls
    flow.run_cycle(dry_run=False)
    assert n1 >= 1 and fb.submit_calls == n1                                             # working orders block re-entry


def test_aggregate_exposure_of_one_cycle_respects_portfolio_limits():
    s, db, fb, ex, flow = build()
    flow.run_cycle(dry_run=False)
    eq = 100_000
    notional = sum(int(p["qty"]) * float(p["limit_price"]) for p in fb.payloads)
    assert len(fb.payloads) <= s.risk.max_new_entries_per_day
    assert notional <= eq * s.risk.max_gross_exposure_pct * 1.01
    for grp in s.universe.correlation_groups.values():
        g = sum(int(p["qty"]) * float(p["limit_price"]) for p in fb.payloads if p["symbol"] in grp)
        assert g <= eq * s.risk.max_group_exposure_pct * 1.01


def test_stale_quotes_block_entries():
    s, db, fb, ex, flow = build(quote_time=MARKET_NOW - timedelta(hours=1))
    st = flow.run_cycle(dry_run=False)
    assert not st.scan.signals and fb.submit_calls == 0


def test_market_data_outage_blocks_and_alerts():
    s, db, fb, ex, flow = build(market=BrokenMarket())
    st = flow.run_cycle(dry_run=False)
    assert "market data unavailable" in st.blocked_reason and fb.submit_calls == 0
    assert db.q("SELECT COUNT(*) n FROM alerts WHERE code='DATA_UNAVAILABLE'")[0]["n"] == 1
    assert db.q("SELECT ok FROM provider_health")[0]["ok"] == 0


def test_broker_outage_blocks_entries():
    fb = FakeBroker(); fb.mode = "down"
    s, db, _, ex, flow = build(fb=fb)
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0 and any("reconcile" in w or "snapshot" in w for w in st.warnings)


def test_daily_loss_breaker_blocks_all_new_entries_and_alerts():
    s, db, fb, ex, flow = build(fb=FakeBroker(equity=97_500, last_equity=100_000))
    st = flow.run_cycle(dry_run=False)
    assert st.decisions and not any(d.approved for d in st.decisions) and fb.submit_calls == 0
    assert not st.decisions[0].checks["daily_loss_breaker"]
    assert db.q("SELECT COUNT(*) n FROM alerts WHERE code='CIRCUIT_BREAKER'")[0]["n"] >= 1


def test_operator_halt_blocks_entries():
    s, db, fb, ex, flow = build()
    db.set_halt(True, "manual")
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0 and not st.decisions[0].checks["operator_not_halted"]


def test_no_trade_is_a_valid_outcome():
    s, db, fb, ex, flow = build(fr=frames(breakout=False, downtrend=True))
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0 and not st.proposals and st.blocked_reason
    assert st.scan.no_trade, "no-trade reasons must be recorded"


def test_malformed_pm_output_results_in_no_trade():
    s, db, fb, ex, flow = build(llm=FakeLLM(["<<not json>>"]))
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0 and not st.proposals
    assert any("portfolio manager skip" in r for x in st.scan.no_trade for r in x["reasons"])


def test_pm_can_scale_size_down_but_never_up():
    ok = {"symbol": "AAPL", "action": "propose", "size_multiplier": 0.5, "confidence": 0.7, "rationale": "half size"}
    s, db, fb, ex, flow = build(llm=FakeLLM([{"regime": "risk_on", "confidence": .8, "rationale": "x"}, ok]))
    full = build()[4].run_cycle(dry_run=True).proposals[0].qty
    st = flow.run_cycle(dry_run=True)
    assert st.proposals[0].qty == int(full * 0.5) or st.proposals[0].qty == full // 2


def test_market_closed_blocks_entries_via_risk_engine():
    s = make_settings(ENABLE_PAPER_TRADING="true")
    sat = datetime(2026, 3, 7, 16, 0, tzinfo=UTC)
    db = Storage(":memory:"); fb = FakeBroker()
    ex = ExecutionService(s, fb, db, now_fn=lambda: sat)
    flow = TradingFlow(s, db, DemoMarketData(frames(), sat), DemoNews([]), ex, RuleBasedScorer(), now_fn=lambda: sat)
    st = flow.run_cycle(dry_run=False)
    assert fb.submit_calls == 0
    assert all(not d.checks["market_open"] for d in st.decisions)


def test_sentiment_is_persisted_and_catalyst_needs_confirmation():
    news = DemoNews([NewsItem(id="n1", headline="Microsoft beats earnings, raises guidance", symbols=["MSFT"],
                              source="t", url="https://x/1", published_at=MARKET_NOW - timedelta(hours=1), ingested_at=MARKET_NOW)])
    s, db, fb, ex, flow = build(news=news)
    st = flow.run_cycle(dry_run=True)
    assert db.q("SELECT COUNT(*) n FROM sentiment WHERE symbol='MSFT'")[0]["n"] == 1
    assert "MSFT" in st.sentiments and st.sentiments["MSFT"].score > 0
    # positive headline alone (no price/volume confirmation) must not create a catalyst signal
    assert not [g for g in st.scan.signals if g.strategy == "catalyst" and g.symbol == "MSFT"]
    # and re-running does not double-count the same story
    flow.run_cycle(dry_run=True)
    assert db.q("SELECT COUNT(*) n FROM sentiment WHERE symbol='MSFT'")[0]["n"] == 1


def test_news_outage_degrades_gracefully():
    class BadNews:
        name = "badnews"
        def news(self, *a, **k): raise ProviderError("HTTP 500", 500)
    s, db, fb, ex, flow = build(news=BadNews())
    st = flow.run_cycle(dry_run=True)
    assert any("news unavailable" in w for w in st.warnings) and st.scan is not None
