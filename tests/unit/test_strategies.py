from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.config import load_settings
from app.data.providers import DemoMarketData, check_bar_freshness, check_quote_freshness, synthetic_daily_bars
from app.flow.pipeline import drop_incomplete_bars
from app.schemas import Direction, SymbolSentiment
from app.strategies.features import compute_features
from app.strategies.regime import compute_regime
from app.strategies.scanner import scan
from app.strategies.scoring import components, total_score
from app.strategies.setups import momentum_breakout, news_catalyst, relative_strength
from app.timeutil import UTC
from datetime import datetime
from tests.conftest import make_settings

S = make_settings()
M = S.strategies.momentum


def base_f(**kw):
    f = dict(close=100.0, prev_close=99.0, sma_fast=98.0, sma_slow=95.0, sma_trend=90.0, sma_slow_slope=0.02, atr=2.0,
             high_n=100.0, low_n=90.0, high_50=101.0, vol_ratio=1.8, avg_volume=5e6, avg_dollar_volume=5e8, roc_63=0.12,
             ret_1d=0.01, atr_pct=0.02, rs=0.08)
    f.update(kw)
    return f


def test_features_have_no_lookahead():
    full = synthetic_daily_bars("AAPL", 320, 3)
    n = 250
    a = compute_features(full.iloc[:n], M)
    mutated = full.copy()
    mutated.iloc[n:] = mutated.iloc[n:] * 10          # wildly change the future
    b = compute_features(mutated.iloc[:n], M)
    assert a == b


def test_features_none_on_short_history():
    assert compute_features(synthetic_daily_bars("AAPL", 60, 1), M) is None


def test_breakout_qualifies_with_correct_levels():
    s = momentum_breakout("AAPL", base_f(), 100.4, M)
    assert s.ok and s.direction == Direction.LONG
    assert s.stop == pytest.approx(100.4 - 2 * 2.0) and s.target == pytest.approx(100.4 + 3 * 4.0)


@pytest.mark.parametrize("override,needle", [
    ({"vol_ratio": 1.0}, "volume"), ({"sma_trend": 99.0}, "trend"), ({"atr_pct": 0.09}, "ATR%"),
    ({"high_n": 130.0, "close": 100.0}, "no breakout")])
def test_breakout_blockers(override, needle):
    s = momentum_breakout("AAPL", base_f(**override), 100.4, M)
    assert not s.ok and any(needle in b for b in s.blockers)


def test_breakout_chase_limit():
    s = momentum_breakout("AAPL", base_f(), 103.0, M)       # 3.0 above close/level = 1.5 ATR > 0.5
    assert not s.ok and any("extended" in b for b in s.blockers)


def _sent(**kw):
    d = dict(symbol="AAPL", score=0.6, n_items=2, max_materiality=0.8, min_novelty_of_top=1.0, newest_age_hours=1.0,
             conflicting=False, stale=False, top_news_id="n1", model="rule_based")
    d.update(kw)
    return SymbolSentiment(**d)


def test_positive_sentiment_alone_never_qualifies():
    s = news_catalyst("AAPL", base_f(vol_ratio=0.9), 100.0, _sent(), S.strategies.catalyst)   # no move, no volume
    assert not s.ok
    assert any("price confirmation" in b for b in s.blockers) and any("volume confirmation" in b for b in s.blockers)


def test_catalyst_qualifies_only_with_confirmation():
    s = news_catalyst("AAPL", base_f(vol_ratio=2.0), 101.5, _sent(), S.strategies.catalyst)
    assert s.ok and s.catalyst_news_id == "n1" and s.horizon == "intraday"


@pytest.mark.parametrize("kw,needle", [({"stale": True}, "stale"), ({"conflicting": True}, "conflicting"),
                                       ({"score": 0.1}, "sentiment"), ({"min_novelty_of_top": 0.2}, "novel"),
                                       ({"max_materiality": 0.2}, "materiality")])
def test_catalyst_news_blockers(kw, needle):
    s = news_catalyst("AAPL", base_f(vol_ratio=2.0), 101.5, _sent(**kw), S.strategies.catalyst)
    assert not s.ok and any(needle in b for b in s.blockers)


def test_catalyst_without_sentiment_blocked():
    s = news_catalyst("AAPL", base_f(vol_ratio=2.0), 101.5, None, S.strategies.catalyst)
    assert not s.ok and any("no sentiment" in b for b in s.blockers)


def test_relative_strength_short_signal_blocked_when_shorting_disabled():
    cfg = S.strategies.relative_strength.model_copy(update={"enabled": True, "allow_short_signals": True})
    f = base_f(rs=-0.1, close=85.0, low_n=90.0, sma_slow=95.0, vol_ratio=2.0)
    out = relative_strength("AAPL", f, 84.0, cfg, allow_short=False)
    short = [s for s in out if s.direction == Direction.SHORT][0]
    assert not short.ok and any("shorting disabled" in b for b in short.blockers)
    short2 = [s for s in relative_strength("AAPL", f, 84.0, cfg, allow_short=True) if s.direction == Direction.SHORT][0]
    assert short2.ok and short2.stop > short2.entry > short2.target


def test_score_renormalises_when_catalyst_absent():
    w = S.scoring.weights
    comps = {"trend_momentum": 1.0, "relative_strength": 1.0, "volume_breakout": 1.0, "catalyst": None, "regime": 1.0}
    assert total_score(comps, w) == 1.0
    comps["catalyst"] = 0.0
    assert total_score(comps, w) == pytest.approx(0.8)


def test_components_bounded():
    c = components(base_f(), Direction.LONG, 100.4, 0.9, 0.8, _sent(), 100.0)
    assert all(0 <= v <= 1 for v in c.values() if v is not None)


def _frames(now, seed=1):
    return {s: synthetic_daily_bars(s, 300, seed, end=now) for s in S.universe.symbols}


def test_scanner_blocks_without_benchmark():
    now = datetime(2026, 3, 6, 22, 0, tzinfo=UTC)
    fr = _frames(now)
    del fr["SPY"]
    r = scan(S, fr, {}, {}, now, require_fresh_quotes=False)
    assert r.signals == [] and "benchmark" in r.blocked_reason


def test_scanner_flags_stale_bars_and_missing_quotes():
    now = datetime(2026, 3, 6, 16, 0, tzinfo=UTC)
    fr = _frames(datetime(2026, 2, 20, tzinfo=UTC))               # bars end ~2 weeks before `now`
    r = scan(S, fr, {}, {}, now, require_fresh_quotes=False)
    assert any("stale" in x["reasons"][0] for x in r.no_trade)


def test_scanner_requires_fresh_quotes_when_open():
    now = datetime(2026, 3, 9, 16, 0, tzinfo=UTC)
    fr = _frames(datetime(2026, 3, 6, 12, tzinfo=UTC))
    md = DemoMarketData(fr, now - timedelta(hours=1))               # quotes one hour old
    r = scan(S, fr, md.latest_quotes(list(fr)), {}, now, require_fresh_quotes=True)
    assert r.signals == [] and any("quote" in x["reasons"][0] for x in r.no_trade)


def test_drop_incomplete_bars_removes_todays_partial_bar():
    now = datetime(2026, 3, 9, 16, 0, tzinfo=UTC)                  # Monday, market open
    df = synthetic_daily_bars("AAPL", 40, 1, end=datetime(2026, 3, 9, tzinfo=UTC))
    assert df.index[-1].date().isoformat() == "2026-03-09"
    out = drop_incomplete_bars({"AAPL": df}, now)["AAPL"]
    assert out.index[-1].date().isoformat() == "2026-03-06"


def test_freshness_helpers():
    now = datetime(2026, 3, 9, 16, 0, tzinfo=UTC)
    assert not check_quote_freshness(None, now, 120)[0]
    assert not check_quote_freshness({"mid": 10, "ts": now - timedelta(seconds=500)}, now, 120)[0]
    assert check_quote_freshness({"mid": 10, "ts": now - timedelta(seconds=5)}, now, 120)[0]
    assert not check_bar_freshness(pd.DataFrame(), now.date(), 3)[0]


def test_regime_unclear_without_history_and_risk_off_in_downtrend():
    assert compute_regime(None, S.regime).regime == "unclear"
    idx = pd.bdate_range(end="2026-03-06", periods=120).tz_localize("UTC")
    px = np.linspace(200, 120, 120)
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1e6}, index=idx)
    assert compute_regime(df, S.regime).regime == "risk_off"
