from datetime import datetime, timedelta

import pytest

from app.analytics.metrics import (equity_metrics, exposure_metrics, pair_trades, slippage_bps, trade_metrics,
                                   turnover)
from app.config import SentimentCfg
from app.schemas import AccountSnapshot, NewsItem, Position, SentimentResult
from app.sentiment import pipeline as sp
from app.timeutil import ET, UTC, ensure_utc, last_completed_session, parse_ts, session_status

NOW = datetime(2026, 10, 9, 15, tzinfo=UTC)
CFG = SentimentCfg()


def item(i, head, age=1.0, syms=("AAPL",), url="u"):
    return NewsItem(id=i, headline=head, symbols=list(syms), published_at=NOW - timedelta(hours=age), ingested_at=NOW, url=url)


def test_dedupe_exact_near_and_universe_mapping():
    items = [item("1", "Apple beats earnings, raises guidance", 3), item("2", "APPLE beats earnings; raises guidance!", 2),
             item("3", "Apple beats earnings and raises guidance today", 1.5), item("4", "Unlisted Co soars", 1, ("ZZZZ",)),
             item("1", "Apple beats earnings, raises guidance", 3)]
    dd = sp.normalize_and_dedupe(items, {"AAPL"}, 0.7)
    assert [i.id for i in dd.unique] == ["1"] and dd.duplicate_counts["1"] == 2


def test_known_hashes_reduce_novelty():
    h = sp.headline_hash("Apple beats earnings")
    dd = sp.normalize_and_dedupe([item("1", "Apple beats earnings")], {"AAPL"}, 0.8, {h})
    assert dd.duplicate_counts["1"] == 1 and sp.novelty(1) == 0.5


def test_html_cleaning_and_hash_stability():
    assert sp.clean_text("<p>Apple &amp; Co  beats</p>") == "Apple & Co beats"
    assert sp.headline_hash("Apple, beats!") == sp.headline_hash("apple beats")


def test_time_decay_half_life():
    assert sp.time_decay(NOW - timedelta(hours=6), NOW, 6.0) == pytest.approx(0.5)
    assert sp.time_decay(NOW, NOW, 6.0) == 1.0
    assert sp.time_decay(NOW + timedelta(hours=1), NOW, 6.0) == 1.0          # future timestamps clamp


def test_materiality_ranks_events():
    hi = sp.materiality(item("1", "Company receives FDA approval"))
    lo = sp.materiality(item("2", "Company hosts a webinar"))
    assert hi > 0.8 > lo
    roundup = sp.materiality(item("3", "FDA approval news", syms=("A", "B", "C")))
    assert roundup < hi


def test_rule_scorer_polarity():
    sc = sp.RuleBasedScorer().score(["Apple beats and surges", "Apple misses and plunges", "Apple holds meeting"])
    assert sc[0][0] > 0 > sc[1][0] and sc[2][0] == 0


def test_analyze_preserves_evidence_and_timestamps():
    res, dd = sp.analyze([item("1", "Apple beats earnings", url="https://x/a")], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    r = res[0]
    assert r.url == "https://x/a" and r.published_at == NOW - timedelta(hours=1) and r.model == "rule_based"
    assert r.label == "positive" and -1 <= r.decayed_score <= 1


def test_aggregate_conflict_and_stale_flags():
    pos, _ = sp.analyze([item("1", "Apple beats earnings", 1)], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    neg, _ = sp.analyze([item("2", "Apple plunges on recall probe", 1.1)], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    agg = sp.aggregate(pos + neg, CFG, NOW)["AAPL"]
    assert agg.conflicting and not agg.stale
    old, _ = sp.analyze([item("3", "Apple beats earnings", 40)], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    assert sp.aggregate(old, CFG, NOW)["AAPL"].stale


def test_old_news_decays_in_aggregate():
    fresh, _ = sp.analyze([item("1", "Apple beats earnings", 0.5)], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    old, _ = sp.analyze([item("2", "Apple beats earnings", 20)], {"AAPL"}, sp.RuleBasedScorer(), CFG, NOW)
    assert sp.aggregate(fresh, CFG, NOW)["AAPL"].score > sp.aggregate(old, CFG, NOW)["AAPL"].score


def test_price_volume_confirmation_labels():
    from app.schemas import SymbolSentiment
    s = SymbolSentiment(symbol="A", score=0.5, n_items=1, max_materiality=0.8, stale=False)
    assert sp.price_volume_confirmation(s, 0.02, 1.6) == "confirmed"
    assert sp.price_volume_confirmation(s, -0.02, 1.6) == "divergent"
    assert sp.price_volume_confirmation(s, 0.001, 1.0) == "unconfirmed"
    assert sp.price_volume_confirmation(None, 0.02, 2) == "no_sentiment"


def test_get_scorer_falls_back_when_finbert_unavailable(monkeypatch):
    def boom(*a, **k):
        raise ImportError("no transformers")
    monkeypatch.setattr(sp, "FinBertScorer", boom)
    scorer, warn = sp.get_scorer(True, "ProsusAI/finbert")
    assert scorer.name == "rule_based" and "FinBERT unavailable" in warn


# ---------- time ----------
def test_dst_shifts_market_open_in_utc():
    pre = session_status(datetime(2026, 3, 6, 15, 0, tzinfo=UTC))      # EST: opens 14:30Z
    post = session_status(datetime(2026, 3, 9, 15, 0, tzinfo=UTC))     # EDT: opens 13:30Z
    assert pre.open_utc.hour == 14 and post.open_utc.hour == 13 and pre.is_open and post.is_open


def test_holiday_and_early_close():
    assert not session_status(datetime(2026, 11, 26, 16, 0, tzinfo=UTC)).is_session_day         # Thanksgiving
    e = session_status(datetime(2026, 11, 27, 17, 0, tzinfo=UTC))
    assert e.early_close and e.close_utc.hour == 18 and e.is_open
    assert not session_status(datetime(2026, 11, 27, 18, 30, tzinfo=UTC)).is_open


def test_weekend_closed_and_last_completed_session():
    assert not session_status(datetime(2026, 3, 7, 16, tzinfo=UTC)).is_session_day
    assert last_completed_session(datetime(2026, 3, 8, 12, tzinfo=UTC)).isoformat() == "2026-03-06"
    assert last_completed_session(datetime(2026, 3, 9, 16, tzinfo=UTC)).isoformat() == "2026-03-06"   # open session
    assert last_completed_session(datetime(2026, 3, 9, 21, tzinfo=UTC)).isoformat() == "2026-03-09"   # after close


def test_naive_datetime_rejected_and_parse_z():
    with pytest.raises(ValueError):
        ensure_utc(datetime(2026, 1, 1))
    assert parse_ts("2026-01-02T05:00:00Z").tzinfo is not None


# ---------- metrics ----------
def fills(*rows):
    return [{"symbol": s, "side": side, "qty": q, "price": p, "strategy": st, "ts": i, "ref_price": None, "proposal_id": None}
            for i, (s, side, q, p, st) in enumerate(rows)]


def test_fifo_pairing_long_partial_and_open_remainder():
    t = pair_trades(fills(("A", "buy", 10, 100, "m"), ("A", "buy", 10, 110, "m"), ("A", "sell", 15, 120, "m")))
    assert [(x.qty, x.pnl) for x in t] == [(10, 200.0), (5, 50.0)]


def test_fifo_pairing_short_round_trip():
    t = pair_trades(fills(("A", "sell", 10, 100, "s"), ("A", "buy", 10, 90, "s")))
    assert t[0].direction == "short" and t[0].pnl == 100.0


def test_trade_metrics_values():
    t = pair_trades(fills(("A", "buy", 10, 100, "m"), ("A", "sell", 10, 110, "m"),
                          ("B", "buy", 10, 50, "c"), ("B", "sell", 10, 45, "c")))
    m = trade_metrics(t)
    assert m["n_trades"] == 2 and m["realized_pnl"] == 50.0 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(2.0) and m["avg_win"] == 100 and m["avg_loss"] == -50
    assert m["attribution"]["m"]["pnl"] == 100 and m["source"] == "derived"


def test_trade_metrics_empty_is_honest():
    assert trade_metrics([])["available"] is False


def _snap(i, eq, last=None):
    return AccountSnapshot(ts=NOW + timedelta(days=i), equity=eq, cash=eq, buying_power=eq, last_equity=last or eq, source="broker")


def test_equity_metrics_drawdown_and_return():
    m = equity_metrics([_snap(0, 100_000), _snap(1, 110_000), _snap(2, 99_000, 110_000)], 100_000)
    assert m["max_drawdown_pct"] == pytest.approx(10.0) and m["total_return_pct"] == pytest.approx(-1.0)
    assert m["daily_pnl"] == -11_000 and m["source"] == "broker"
    assert equity_metrics([])["available"] is False


def test_exposure_turnover_slippage():
    pos = [Position(symbol="A", qty=10, avg_entry_price=1, market_value=5_000, unrealized_pl=100),
           Position(symbol="B", qty=-5, avg_entry_price=1, market_value=-2_000, unrealized_pl=-20)]
    ex = exposure_metrics(pos, 100_000)
    assert ex["gross_pct"] == pytest.approx(7.0) and ex["net_pct"] == pytest.approx(3.0) and ex["unrealized_pnl"] == 80
    assert turnover(fills(("A", "buy", 10, 100, None)), 10_000) == pytest.approx(0.1)
    f = [{"side": "buy", "price": 100.10, "ref_price": 100.0, "proposal_id": "p"}]
    assert slippage_bps(f)["avg_bps"] == pytest.approx(10.0, rel=1e-3)
    assert slippage_bps([{"side": "buy", "price": 1, "ref_price": None, "proposal_id": None}])["available"] is False
