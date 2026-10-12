import pandas as pd
import pytest

from app.replay.engine import demo_frames, run_replay
from tests.conftest import make_settings

S = make_settings()
REGRESSION_RETURN = 2.5773


SUBSET = S.universe.symbols[:14]          # smaller universe + history keeps the suite fast (pandas overhead)


@pytest.fixture(scope="module")
def frames():
    return demo_frames(SUBSET, 380, 7)


@pytest.fixture(scope="module")
def base(frames):
    return run_replay(S.model_copy(update={"universe": S.universe.model_copy(update={"symbols": SUBSET})}), frames, "demo")


S = S.model_copy(update={"universe": S.universe.model_copy(update={"symbols": SUBSET})})


def test_replay_is_deterministic(frames, base):
    b = run_replay(S, frames, "demo")
    assert base.equity_curve == b.equity_curve and base.metrics == b.metrics


def test_replay_has_no_lookahead(frames, base):
    cut = 330
    day = frames["SPY"].index[cut]
    mutated = {k: v.copy() for k, v in frames.items()}
    for v in mutated.values():
        v.iloc[cut:] = v.iloc[cut:] * 3.0                      # rewrite the future
    alt = run_replay(S, mutated, "demo")
    pre = [(d, e) for d, e in base.equity_curve if pd.Timestamp(d, tz="UTC") < day]
    pre_alt = [(d, e) for d, e in alt.equity_curve if pd.Timestamp(d, tz="UTC") < day]
    assert pre == pre_alt and len(pre) > 50


def test_entries_fill_at_next_open_with_slippage(frames, base):
    r = base
    assert r.trades
    for t in r.trades[:15]:
        op = frames[t.symbol].loc[pd.Timestamp(t.entry_date, tz="UTC"), "open"]
        assert t.entry == pytest.approx(op * 1.0005, rel=1e-6)


def test_trades_respect_risk_per_trade_at_entry(frames, base):
    r = base
    for t in r.trades:
        assert t.qty * t.entry <= 100_000 * 1.6 * S.risk.max_position_pct          # equity drift slack
    assert r.metrics["max_drawdown_pct"] >= 0 and r.origin == "demo"


def test_replay_regression_snapshot(base):
    r = base
    assert 5 <= r.metrics["n_trades"] <= 200                # structural guard, not a performance claim
    assert r.metrics["total_return_pct"] == pytest.approx(REGRESSION_RETURN, abs=0.01)


def test_replay_requires_benchmark():
    with pytest.raises(ValueError):
        run_replay(S, {"AAPL": demo_frames(["AAPL"])["AAPL"]}, "x")
