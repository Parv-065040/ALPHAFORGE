"""Mocked-broker order lifecycle. No network, no live orders."""
from datetime import datetime, timedelta

import pytest

from app.execution.service import ExecutionService, client_id_for
from app.schemas import OrderStatus, Position, RiskDecision
from app.storage.db import Storage
from app.timeutil import UTC
from tests.conftest import MARKET_NOW, make_proposal, make_settings
from tests.fakes import FakeBroker


def setup(paper=True, tmp=None, broker=None, now=MARKET_NOW, mode="ok"):
    s = make_settings(ENABLE_PAPER_TRADING="true" if paper else "false")
    db = Storage(tmp or ":memory:")
    fb = broker or FakeBroker()
    fb.mode = mode
    clock = {"t": now}
    ex = ExecutionService(s, fb, db, now_fn=lambda: clock["t"])
    ex.reconcile() if mode != "down" else None
    return s, db, fb, ex, clock


def approved(db, p, qty=None, now=MARKET_NOW):
    d = RiskDecision(proposal_id=p.id, approved=True, approved_qty=qty or p.qty, decided_at=now, reasons=["approved"])
    db.save_proposal(p); db.save_decision(d)
    return d


def test_submit_creates_bracket_order_and_persists_state():
    s, db, fb, ex, _ = setup()
    p = make_proposal(qty=40)
    r = ex.execute(p, approved(db, p))
    assert r.status == "submitted" and fb.submit_calls == 1
    pl = fb.payloads[0]
    assert pl["order_class"] == "bracket" and pl["client_order_id"] == client_id_for(p.id) and pl["time_in_force"] == "gtc"
    assert pl["stop_loss"]["stop_price"] == "96.00" and pl["take_profit"]["limit_price"] == "112.00"
    assert pl["type"] == "limit" and float(pl["limit_price"]) == pytest.approx(100.10)
    o = db.get_order(client_order_id=client_id_for(p.id))
    assert o.status == OrderStatus.ACCEPTED and o.broker_order_id and o.qty == 40


def test_duplicate_proposal_never_resubmits():
    s, db, fb, ex, _ = setup()
    p = make_proposal()
    d = approved(db, p)
    ex.execute(p, d)
    r2 = ex.execute(p, d)
    assert r2.status == "duplicate" and fb.submit_calls == 1


def test_execution_requires_stored_matching_approved_decision():
    s, db, fb, ex, _ = setup()
    p = make_proposal()
    forged = RiskDecision(proposal_id=p.id, approved=True, approved_qty=10, decided_at=MARKET_NOW)
    assert ex.execute(p, forged).status == "blocked" and fb.submit_calls == 0      # never stored
    db.save_decision(RiskDecision(proposal_id=p.id, approved=False, approved_qty=0, decided_at=MARKET_NOW))
    assert ex.execute(p, forged).status == "blocked"                               # stored one is a rejection
    other = make_proposal()
    d_other = approved(db, other)
    assert ex.execute(p, d_other).status == "blocked"                              # wrong proposal id
    assert fb.submit_calls == 0


def test_expired_approval_and_expired_proposal_blocked():
    s, db, fb, ex, clock = setup()
    p = make_proposal()
    d = approved(db, p)
    clock["t"] = MARKET_NOW + timedelta(minutes=s.risk.approval_ttl_minutes + 1)
    ex.reconcile()
    assert ex.execute(p, d).status == "blocked"
    p2 = make_proposal(symbol="MSFT", expires_at=MARKET_NOW + timedelta(minutes=2))
    d2 = approved(db, p2, now=clock["t"])                       # fresh approval, but the proposal itself has expired
    r2 = ex.execute(p2, d2)
    assert r2.status == "blocked" and "proposal expired" in r2.detail and fb.submit_calls == 0


def test_unreconciled_broker_blocks_execution():
    s, db, fb, ex, clock = setup()
    p = make_proposal()
    d = approved(db, p)
    clock["t"] = MARKET_NOW + timedelta(minutes=25)                                 # reconcile is stale (>20 min)
    d = approved(db, p, now=clock["t"])
    r = ex.execute(p, d)
    assert r.status == "blocked" and "reconciled" in r.detail


def test_broker_rejection_is_recorded_not_retried():
    s, db, fb, ex, _ = setup(mode="reject")
    p = make_proposal()
    r = ex.execute(p, approved(db, p))
    assert r.status == "rejected" and fb.submit_calls == 1
    assert db.get_order(client_id_for(p.id)).status == OrderStatus.REJECTED
    assert db.q("SELECT COUNT(*) n FROM alerts WHERE code='ORDER_REJECTED'")[0]["n"] == 1


def test_timeout_then_order_exists_is_recovered_without_second_submit():
    s, db, fb, ex, _ = setup(mode="timeout_exists")
    p = make_proposal()
    r = ex.execute(p, approved(db, p))
    assert r.status == "submitted" and fb.submit_calls == 1                        # NOT retried: it existed
    assert "reconciliation" in db.get_order(client_id_for(p.id)).detail


def test_timeout_and_confirmed_absent_retries_exactly_once():
    s, db, fb, ex, _ = setup(mode="timeout_absent_once")
    p = make_proposal()
    r = ex.execute(p, approved(db, p))
    assert r.status == "submitted" and fb.submit_calls == 2
    assert len(fb.orders) == 1


def test_persistent_timeouts_end_unknown_and_block_new_entries():
    s, db, fb, ex, _ = setup(mode="timeout_always")
    p = make_proposal()
    r = ex.execute(p, approved(db, p))
    assert r.status == "unknown" and fb.submit_calls == 2
    assert db.get_order(client_id_for(p.id)).status == OrderStatus.UNKNOWN
    assert not ex.is_reconciled()
    p2 = make_proposal(symbol="MSFT")
    assert ex.execute(p2, approved(db, p2)).status == "blocked"


def test_lookup_failure_after_timeout_is_unknown_not_assumed_failed():
    s, db, fb, ex, _ = setup(mode="timeout_exists")
    fb.mode = "timeout_exists"
    orig = fb.get_order_by_client_id
    fb.get_order_by_client_id = lambda cid: (_ for _ in ()).throw(__import__("app.data.http", fromlist=["x"]).TimeoutOrNetwork("x"))
    p = make_proposal()
    r = ex.execute(p, approved(db, p))
    assert r.status == "unknown" and fb.submit_calls == 1
    fb.get_order_by_client_id = orig
    rep = ex.reconcile()                                                           # later reconciliation resolves it
    assert db.get_order(client_id_for(p.id)).status == OrderStatus.ACCEPTED and rep.ok


def test_partial_then_full_fill_records_fill_deltas_and_position_meta():
    s, db, fb, ex, _ = setup()
    p = make_proposal(qty=100)
    ex.execute(p, approved(db, p))
    cid = client_id_for(p.id)
    fb.fill(cid, 40, 100.0, full=False)
    ex.sync_orders()
    o = db.get_order(cid)
    assert o.status == OrderStatus.PARTIALLY_FILLED and o.filled_qty == 40
    fb.fill(cid, 100, 100.5, full=True)
    ex.sync_orders()
    o = db.get_order(cid)
    assert o.status == OrderStatus.FILLED
    fills = db.q("SELECT qty, price FROM fills WHERE client_order_id=? ORDER BY id", (cid,))
    assert [round(f["qty"]) for f in fills] == [40, 60] and sum(f["qty"] * f["price"] for f in fills) == pytest.approx(100 * 100.5)
    assert db.position_meta()["AAPL"]["strategy"] == "momentum"
    # bracket legs registered for exit tracking
    assert any(o.detail.startswith("bracket leg") for o in db.all_orders())


def test_exit_leg_fill_creates_closed_trade_pnl():
    from app.analytics.metrics import pair_trades
    s, db, fb, ex, _ = setup()
    p = make_proposal(qty=10)
    ex.execute(p, approved(db, p))
    cid = client_id_for(p.id)
    fb.fill(cid, 10, 100.0)
    ex.sync_orders()
    tp = [l for l in fb.orders[cid]["legs"] if l["client_order_id"].startswith("take_profit")][0]
    tp.update(status="filled", filled_qty="10", filled_avg_price="112.0")
    fb.orders[cid]["legs"] = fb.orders[cid]["legs"]
    ex.sync_orders()
    trades = pair_trades(db.fills_with_orders())
    assert len(trades) == 1 and trades[0].pnl == pytest.approx(120.0) and trades[0].strategy == "momentum"


def test_canceled_and_expired_states_map_correctly():
    s, db, fb, ex, _ = setup()
    p = make_proposal()
    ex.execute(p, approved(db, p))
    cid = client_id_for(p.id)
    fb.orders[cid]["status"] = "expired"
    ex.sync_orders()
    assert db.get_order(cid).status == OrderStatus.EXPIRED
    p2 = make_proposal(symbol="MSFT")
    ex.execute(p2, approved(db, p2))
    fb.orders[client_id_for(p2.id)]["status"] = "canceled"
    ex.sync_orders()
    assert db.get_order(client_id_for(p2.id)).status == OrderStatus.CANCELED


def test_stale_unfilled_entries_are_cancelled():
    s, db, fb, ex, clock = setup()
    p = make_proposal()
    ex.execute(p, approved(db, p))
    clock["t"] = MARKET_NOW + timedelta(minutes=s.risk.entry_order_max_age_minutes + 1)
    assert ex.cancel_stale_entries() == 1 and len(fb.cancelled) == 1
    assert db.get_order(client_id_for(p.id)).status == OrderStatus.CANCELED


def test_restart_recovery_resolves_orders_from_broker(tmp_path):
    path = str(tmp_path / "r.db")
    s, db, fb, ex, clock = setup(tmp=path)
    p = make_proposal()
    ex.execute(p, approved(db, p))
    cid = client_id_for(p.id)
    db.close()
    fb.fill(cid, p.qty, 100.0)                                       # fills while the app is down
    db2 = Storage(path)
    ex2 = ExecutionService(s, fb, db2, now_fn=lambda: clock["t"])
    rep = ex2.reconcile()                                            # startup reconciliation
    assert rep.ok and db2.get_order(cid).status == OrderStatus.FILLED
    assert ex2.is_reconciled()


def test_reconcile_reports_untracked_positions_and_broker_outage():
    s, db, fb, ex, _ = setup()
    fb.positions = [Position(symbol="TSLA", qty=5, avg_entry_price=200, market_value=1000)]
    rep = ex.reconcile()
    assert rep.ok and any("TSLA" in w for w in rep.warnings)
    fb.mode = "down"
    rep = ex.reconcile()
    assert not rep.ok and not ex.is_reconciled()
    assert db.q("SELECT COUNT(*) n FROM alerts WHERE code='RECONCILE_FAILED'")[0]["n"] == 1


def test_inactive_account_fails_reconciliation():
    s, db, fb, ex, _ = setup(broker=FakeBroker(status="ACCOUNT_CLOSED"))
    assert not ex.reconcile().ok


def test_intraday_positions_flattened_near_close_only():
    close_near = datetime(2026, 3, 9, 19, 55, tzinfo=UTC)           # 15:55 ET, 5 min to close
    s, db, fb, ex, clock = setup(now=close_near)
    p = make_proposal(horizon=__import__("app.schemas", fromlist=["Horizon"]).Horizon.INTRADAY, strategy="catalyst", qty=10,
                      created_at=close_near, expires_at=close_near + timedelta(minutes=30))
    db.save_proposal(p)
    db.set_position_meta("AAPL", "intraday", "catalyst", p.id, close_near)
    db.set_position_meta("MSFT", "swing", "momentum", "x", close_near)
    fb.positions = [Position(symbol="AAPL", qty=10, avg_entry_price=100, market_value=1000),
                    Position(symbol="MSFT", qty=10, avg_entry_price=100, market_value=1000)]
    closed = ex.flatten_intraday()
    assert closed == ["AAPL"] and fb.payloads[-1]["side"] == "sell" and fb.payloads[-1]["type"] == "market"
    assert ex.flatten_intraday() == []                              # idempotent for the day
    earlier = datetime(2026, 3, 9, 17, 0, tzinfo=UTC)
    clock["t"] = earlier
    db.x("DELETE FROM orders")
    assert ex.flatten_intraday() == []


def test_flatten_does_nothing_when_trading_disabled():
    close_near = datetime(2026, 3, 9, 19, 55, tzinfo=UTC)
    s, db, fb, ex, _ = setup(paper=False, now=close_near)
    db.set_position_meta("AAPL", "intraday", "catalyst", "p", close_near)
    fb.positions = [Position(symbol="AAPL", qty=10, avg_entry_price=100, market_value=1000)]
    assert ex.flatten_intraday() == [] and fb.submit_calls == 0


def test_smoke_test_is_explicit_and_disabled_by_default():
    s, db, fb, ex, _ = setup(paper=False)
    assert ex.paper_smoke_test("SPY", 500)["ok"] is False and fb.submit_calls == 0
    s2, db2, fb2, ex2, _ = setup(paper=True)
    r = ex2.paper_smoke_test("SPY", 500)
    assert r["ok"] and fb2.payloads[0]["limit_price"] == "250.00" and len(fb2.cancelled) == 1
