"""DEMO seeding: synthetic, clearly labelled data in a SEPARATE database (origin='demo').
Never touches the live/paper database and never mixes with broker metrics."""
from __future__ import annotations

import random
from datetime import timedelta

import numpy as np

from app.config import Settings
from app.data.providers import DemoMarketData, DemoNews, synthetic_daily_bars
from app.schemas import AccountSnapshot, NewsItem
from app.sentiment.pipeline import RuleBasedScorer, aggregate, analyze
from app.storage.db import Storage
from app.strategies.scanner import scan
from app.timeutil import utcnow

DEMO_DB = "data/demo.db"


def seed_demo(s: Settings, path: str = DEMO_DB) -> Storage:
    db = Storage(path)
    db.set_meta("origin", "demo")
    now = utcnow()
    rng = random.Random(3)
    syms = s.universe.symbols
    frames = {x: synthetic_daily_bars(x, 320, 11, end=now) for x in syms}
    a = frames["AAPL"]
    a.iloc[-1, a.columns.get_loc("close")] = a["high"].iloc[-21:-1].max() * 1.01
    a.iloc[-1, a.columns.get_loc("high")] = a["close"].iloc[-1] * 1.003
    a.iloc[-1, a.columns.get_loc("volume")] = a["volume"].mean() * 2.6
    heads = ["{} beats earnings estimates, raises guidance", "{} downgrade by analyst on valuation concerns",
             "{} announces share buyback program", "{} faces investigation over product recall"]
    items = [NewsItem(id=f"demo-{i}", headline=heads[i % 4].format(syms[i % len(syms)]), symbols=[syms[i % len(syms)]],
                      source="DEMO", url=f"https://example.invalid/demo/{i}", summary="Synthetic demo headline.",
                      published_at=now - timedelta(hours=rng.uniform(0.5, 30)), ingested_at=now) for i in range(24)]
    res, dd = analyze(DemoNews(items).news([], now - timedelta(days=3)), set(syms), RuleBasedScorer(), s.sentiment, now)
    db.save_news(dd.unique, dd.hashes)
    db.save_sentiment(res)
    md = DemoMarketData(frames, now)
    r = scan(s, md.daily_bars(syms, now, now), md.latest_quotes(syms), aggregate(res, s.sentiment, now), now,
             require_fresh_quotes=False)
    db.save_scan(r.scan_id, now, r.universe_size, r.candidates, r.signals, r.regime.regime, r.blocked_reason, r.features,
                          r.no_trade, r.regime.reasons)
    eq = 100_000.0
    for k in range(40):
        eq *= 1 + float(np.random.default_rng(k).normal(0.0004, 0.004))
        db.save_account(AccountSnapshot(ts=now - timedelta(days=40 - k), equity=eq, cash=eq * 0.6, buying_power=eq * 2,
                                        last_equity=eq / 1.001, status="DEMO", source="demo"))
    db.alert("info", "DEMO", "Demo database: synthetic data only")
    return db
