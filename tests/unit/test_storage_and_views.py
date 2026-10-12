from datetime import timedelta

import pytest

from app.analytics.summary import performance_summary
from app.dashboard import data as dv
from app.demo import seed_demo
from app.preflight import format_report, paper_mode_confirmed, run_preflight
from app.schemas import OrderRecord, OrderStatus
from app.storage.db import MIGRATIONS, SCHEMA_VERSION, Storage
from tests.conftest import MARKET_NOW, make_settings


def test_migrations_apply_and_are_idempotent(tmp_path):
    p = tmp_path / "t.db"
    a = Storage(p)
    assert a.schema_version() == SCHEMA_VERSION == MIGRATIONS[-1][0]
    a.close()
    b = Storage(p)
    assert b.schema_version() == SCHEMA_VERSION
    b.close()


def test_state_survives_restart(tmp_path):
    p = tmp_path / "t.db"
    a = Storage(p)
    o = OrderRecord(client_order_id="af-1", symbol="AAPL", side="buy", qty=10, submitted_at=MARKET_NOW, updated_at=MARKET_NOW)
    a.upsert_order(o, ref_price=100.0, strategy="momentum")
    a.set_meta("reconciled", {"ok": True})
    a.close()
    b = Storage(p)
    assert b.get_order("af-1").status == OrderStatus.PENDING_SUBMIT and len(b.open_orders()) == 1
    assert b.get_meta("reconciled") == {"ok": True}


def test_order_transitions_are_logged(db):
    o = OrderRecord(client_order_id="c", symbol="A", side="buy", qty=1, submitted_at=MARKET_NOW, updated_at=MARKET_NOW)
    db.upsert_order(o)
    db.upsert_order(o.model_copy(update={"status": OrderStatus.ACCEPTED}))
    db.upsert_order(o.model_copy(update={"status": OrderStatus.FILLED, "filled_qty": 1, "avg_fill_price": 5}))
    assert [e["to_status"] for e in db.order_events("c")] == ["pending_submit", "accepted", "filled"]
    assert db.open_orders() == []


def test_entries_today_ignores_rejected_and_legs(db):
    mk = lambda cid, st, pid: OrderRecord(client_order_id=cid, symbol="A", side="buy", qty=1, status=st, proposal_id=pid,
                                          submitted_at=MARKET_NOW, updated_at=MARKET_NOW)
    db.upsert_order(mk("1", OrderStatus.ACCEPTED, "p1"))
    db.upsert_order(mk("2", OrderStatus.REJECTED, "p2"))
    db.upsert_order(mk("3", OrderStatus.ACCEPTED, None))
    assert db.entries_today(MARKET_NOW - timedelta(hours=1)) == 1


# ---- dashboard empty / unavailable / demo states ----
def test_dashboard_views_empty_on_fresh_db(db):
    assert dv.portfolio_view(db).status == "empty"
    assert dv.strategy_view(db).status == "empty"
    assert dv.agent_view(db).status == "empty"
    assert dv.sentiment_view(db).status == "empty"
    assert dv.ops_view(db).status == "ok"          # ops always renders (shows 'not reconciled')
    assert dv.ops_view(db).data["reconcile"] is None


def test_dashboard_views_unavailable_with_reason_when_db_broken():
    d = Storage(":memory:")
    d.close()
    for fn in (dv.portfolio_view, dv.strategy_view, dv.agent_view, dv.sentiment_view, dv.ops_view):
        v = fn(d)
        assert v.status == "unavailable" and v.reason


def test_performance_summary_empty_does_not_fabricate(db):
    s = performance_summary(db)
    assert s["equity"]["available"] is False and s["trades"]["available"] is False
    assert s["realized_pnl"] is None and s["unrealized_pnl"] is None


def test_demo_db_is_isolated_and_labelled(tmp_path):
    d = seed_demo(make_settings(), str(tmp_path / "demo.db"))
    assert d.origin() == "demo"
    assert dv.banner(d, make_settings())["origin"] == "demo"
    assert dv.portfolio_view(d).data["source"] == "demo"
    assert dv.strategy_view(d).status == "ok"
    live = Storage(tmp_path / "live.db")
    assert live.origin() == "live" and dv.portfolio_view(live).status == "empty"      # demo never leaks into live DB


def test_config_view_hides_credentials():
    v = dv.config_view(make_settings())
    assert "PKTESTKEY1234567890" not in str(v.data) and v.data["secrets"]["ALPACA_API_KEY_ID"] == "set"


# ---- preflight ----
def test_preflight_fails_closed_without_paper_confirmation():
    s = make_settings(ALPACA_PAPER="false", ALPACA_API_KEY_ID="", ALPACA_API_SECRET_KEY="")
    checks = run_preflight(s, online=False)
    d = {c.name: c.status for c in checks}
    assert d["paper_mode_flag"] == "FAIL" and d["env:ALPACA_API_KEY_ID"] == "FAIL"
    assert not paper_mode_confirmed(checks)


def test_preflight_never_prints_secret_values(tmp_path):
    s = make_settings()
    report = format_report(run_preflight(s, online=False))
    assert "PKTESTKEY1234567890" not in report and "testsecretvalue123456" not in report
    assert "paper_order_execution: disabled" in report or "disabled (default)" in report
