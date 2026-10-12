from datetime import timedelta

import pytest

from app.risk.engine import RiskContext, RiskEngine, breaker_state, size_position
from app.schemas import AccountSnapshot, Direction, Horizon, OrderRecord, Position
from app.timeutil import SessionStatus
from tests.conftest import MARKET_NOW, make_proposal, make_settings, open_session

S = make_settings()
ENG = RiskEngine(S)
CFG = S.risk


def acct(equity=100_000.0, last=None, bp=200_000.0):
    return AccountSnapshot(ts=MARKET_NOW, equity=equity, cash=equity, buying_power=bp, last_equity=last or equity, status="ACTIVE")


def ctx(**kw):
    d = dict(now=MARKET_NOW, account=acct(), positions=[], open_orders=[], session=open_session(), quote_fresh=True,
             bars_fresh=True, broker_reconciled=True, peak_equity=100_000.0)
    d.update(kw)
    return RiskContext(**d)


def test_sizing_risk_budget_binds():
    qty, caps = size_position(CFG, 100_000, 100, 90, Direction.LONG, 1e6, 0)       # 500 risk / $10 = 50
    assert qty == 50 and caps["risk_budget"] == 50


def test_sizing_position_cap_binds():
    qty, caps = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1e6, 0)       # risk 250 sh; cap 15% = 150
    assert qty == 150 and caps["position_cap"] == 150


def test_sizing_gross_exposure_headroom():
    qty, _ = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1e6, 69_000)     # 70k-69k = 1k -> 10 sh
    assert qty == 10


def test_sizing_buying_power_and_buffer():
    qty, _ = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1_000, 0)        # 950/100 = 9
    assert qty == 9


def test_sizing_liquidity_cap():
    qty, caps = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1e6, 0, avg_volume=3_000)   # 1% = 30
    assert qty == 30 and caps["liquidity"] == 30


def test_sizing_envelope_and_group_headroom():
    q1, _ = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1e6, 0, horizon_notional=24_500, horizon=Horizon.SWING)
    assert q1 == 5                                                                  # 25k envelope - 24.5k
    q2, _ = size_position(CFG, 100_000, 100, 98, Direction.LONG, 1e6, 0, group_headroom=2_000)
    assert q2 == 20


def test_sizing_degenerate_inputs_return_zero():
    assert size_position(CFG, 100_000, 100, 100, Direction.LONG, 1e6, 0)[0] == 0
    assert size_position(CFG, 0, 100, 98, Direction.LONG, 1e6, 0)[0] == 0


def test_sizing_short_uses_absolute_distance():
    assert size_position(CFG, 100_000, 100, 110, Direction.SHORT, 1e6, 0)[0] == 50


def test_approval_happy_path_and_resize():
    d = ENG.evaluate(make_proposal(qty=1000), ctx())
    assert d.approved and 0 < d.approved_qty < 1000 and any("resized" in r for r in d.reasons)
    assert d.approved_qty == 125                                                    # $500 risk / $4 stop distance
    d2 = ENG.evaluate(make_proposal(qty=50), ctx())
    assert d2.approved and d2.approved_qty == 50


def _rejects(proposal=None, **c):
    d = ENG.evaluate(proposal or make_proposal(), ctx(**c))
    assert not d.approved and d.approved_qty == 0
    return d


def test_reject_expired_proposal():
    p = make_proposal(expires_at=MARKET_NOW - timedelta(seconds=1), created_at=MARKET_NOW - timedelta(minutes=31))
    assert "proposal_not_expired" in _rejects(p).reasons[0] or not _rejects(p).checks["proposal_not_expired"]


def test_reject_duplicate_decision_and_working_order_and_position():
    assert not _rejects(already_decided=True).checks["not_already_decided"]
    o = OrderRecord(client_order_id="x", symbol="AAPL", side="buy", qty=1, submitted_at=MARKET_NOW, updated_at=MARKET_NOW)
    assert not _rejects(open_orders=[o]).checks["no_duplicate_or_conflict"]
    pos = Position(symbol="AAPL", qty=10, avg_entry_price=99, market_value=1000)
    assert not _rejects(positions=[pos]).checks["no_duplicate_or_conflict"]


@pytest.mark.parametrize("kw,check", [({"quote_fresh": False}, "data_fresh"), ({"bars_fresh": False}, "data_fresh"),
                                      ({"broker_reconciled": False}, "broker_reconciled"),
                                      ({"operator_halt": True}, "operator_not_halted")])
def test_reject_stale_or_unreconciled_or_halt(kw, check):
    assert not _rejects(**kw).checks[check]


def test_reject_market_closed_and_blackouts():
    closed = SessionStatus(True, False, MARKET_NOW.date(), None, None, None, None, False)
    assert not _rejects(session=closed).checks["market_open"]
    early = SessionStatus(True, True, MARKET_NOW.date(), MARKET_NOW, MARKET_NOW, 2, 300, False)
    assert not _rejects(session=early).checks["session_window"]
    late = SessionStatus(True, True, MARKET_NOW.date(), MARKET_NOW, MARKET_NOW, 380, 10, False)
    assert not _rejects(session=late).checks["session_window"]


def test_intraday_needs_time_left():
    late = SessionStatus(True, True, MARKET_NOW.date(), MARKET_NOW, MARKET_NOW, 350, 20, False)
    d = ENG.evaluate(make_proposal(horizon=Horizon.INTRADAY), ctx(session=late))
    assert not d.checks["intraday_time_left"]


def test_daily_loss_circuit_breaker_boundaries():
    just_ok = _approved(ctx(account=acct(equity=98_100, last=100_000)))      # -1.9%
    assert just_ok
    d = _rejects(account=acct(equity=98_000, last=100_000))                   # -2.0% trips
    assert not d.checks["daily_loss_breaker"]
    assert breaker_state(CFG, acct(98_000, 100_000), 100_000)["daily_loss_tripped"]


def test_drawdown_review_trigger():
    d = _rejects(account=acct(equity=95_000, last=95_000), peak_equity=100_000)
    assert not d.checks["drawdown_review"]
    assert _approved(ctx(account=acct(equity=95_100, last=95_100), peak_equity=100_000))


def _approved(c):
    return ENG.evaluate(make_proposal(), c).approved


def test_short_blocked_by_default_even_when_shortable():
    p = make_proposal(direction=Direction.SHORT, stop_price=104.0, target_price=88.0)
    d = ENG.evaluate(p, ctx(shortable=True))
    assert not d.approved and not d.checks["short_permitted"]


def test_short_allowed_only_with_config_and_shortable_asset():
    perm = RiskEngine(make_settings(ALLOW_SHORTING="true"))
    p = make_proposal(direction=Direction.SHORT, stop_price=104.0, target_price=88.0)
    assert perm.evaluate(p, ctx(shortable=True)).approved
    assert not perm.evaluate(p, ctx(shortable=None)).approved
    assert not perm.evaluate(p, ctx(shortable=False)).approved


def test_structural_validation_wrong_side_stop():
    d = ENG.evaluate(make_proposal(stop_price=104.0), ctx())
    assert not d.approved and not d.checks["proposal_valid"]


def test_max_positions_entries_liquidity_reward_risk():
    many = [Position(symbol=f"S{c}", qty=1, avg_entry_price=1, market_value=100) for c in "ABCDEFGH"]
    assert not _rejects(positions=many).checks["max_positions"]
    assert not _rejects(entries_today=CFG.max_new_entries_per_day).checks["max_entries_today"]
    assert not ENG.evaluate(make_proposal(avg_dollar_volume=1e6), ctx()).checks["liquidity"]
    assert not ENG.evaluate(make_proposal(target_price=101.0), ctx()).checks["reward_risk"]


def test_correlated_group_limit_applies():
    held = [Position(symbol="MSFT", qty=1, avg_entry_price=1, market_value=29_900)]       # mega_tech cap 30k
    d = ENG.evaluate(make_proposal(symbol="AAPL"), ctx(positions=held))
    assert not d.approved and not d.checks["size_nonzero"]


def test_risk_per_trade_is_never_exceeded():
    for entry, stop in [(50, 48.5), (250, 240), (12.3, 11.9)]:
        qty, _ = size_position(CFG, 100_000, entry, stop, Direction.LONG, 1e9, 0)
        assert qty * abs(entry - stop) <= 100_000 * CFG.risk_per_trade_pct + 1e-9
