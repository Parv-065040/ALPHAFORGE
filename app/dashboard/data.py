"""Dashboard data layer: pure functions over Storage. Every view returns a ViewState so the UI can show
'No data yet' (empty) or 'Unavailable: reason' (error) honestly. Nothing here fabricates values."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from app.analytics.summary import performance_summary
from app.config import Settings
from app.storage.db import Storage


@dataclass
class ViewState:
    status: str                  # ok | empty | unavailable
    data: Any = None
    reason: str = ""


def _safe(fn: Callable[[], ViewState]) -> ViewState:
    try:
        return fn()
    except Exception as e:      # DB locked/corrupt/missing tables etc.
        return ViewState("unavailable", None, f"{type(e).__name__}: {e}")


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows])


def banner(db: Storage, s: Settings) -> dict:
    def f():
        return {"origin": db.origin(), "paper_orders_enabled": s.enable_paper_trading,
                "shorting_enabled": s.allow_shorting, "halted": db.halt_state().get("halted", False)}
    try:
        return f()
    except Exception as e:
        return {"origin": "unknown", "error": str(e), "paper_orders_enabled": s.enable_paper_trading,
                "shorting_enabled": s.allow_shorting, "halted": False}


def portfolio_view(db: Storage, starting_equity: float | None = None) -> ViewState:
    def f():
        hist = db.account_history()
        if not hist:
            return ViewState("empty", None, "No account snapshots yet (run a paper session or reconcile).")
        summ = performance_summary(db, starting_equity)
        return ViewState("ok", {
            "summary": summ,
            "equity_curve": pd.DataFrame({"ts": [h.ts for h in hist], "equity": [h.equity for h in hist]}),
            "positions": pd.DataFrame([p.model_dump() for p in db.latest_positions()]),
            "orders": pd.DataFrame([o.model_dump(mode="json") for o in db.all_orders(100)]),
            "source": hist[-1].source})
    return _safe(f)


def strategy_view(db: Storage) -> ViewState:
    def f():
        scans = db.q("SELECT * FROM scans ORDER BY ts DESC LIMIT 1")
        if not scans:
            return ViewState("empty", None, "No scans recorded yet.")
        sc = scans[0]
        payload = json.loads(sc["payload"] or "{}")
        cands = _df(db.q("SELECT symbol,strategy,score,payload FROM candidates WHERE scan_id=? ORDER BY score DESC",
                         (sc["id"],)))
        if not cands.empty:
            cands["components"] = cands["payload"].map(lambda p: json.loads(p).get("components", {}))
            cands["reasons"] = cands["payload"].map(lambda p: "; ".join(json.loads(p).get("reasons", [])))
            cands = cands.drop(columns=["payload"])
        sig = _df(db.q("SELECT symbol,strategy,direction,score,expires_at,payload FROM signals WHERE scan_id=? "
                       "ORDER BY score DESC", (sc["id"],)))
        if not sig.empty:
            sig["entry"] = sig["payload"].map(lambda p: json.loads(p)["entry_price"])
            sig["stop"] = sig["payload"].map(lambda p: json.loads(p)["stop_price"])
            sig["target"] = sig["payload"].map(lambda p: json.loads(p)["target_price"])
            sig = sig.drop(columns=["payload"])
        return ViewState("ok", {"scan": dict(sc) | {"payload": None}, "regime": payload.get("regime"),
                                "regime_reasons": payload.get("regime_reasons", []), "blocked": payload.get("blocked"),
                                "candidates": cands, "signals": sig,
                                "no_trade": pd.DataFrame(payload.get("no_trade", []))})
    return _safe(f)


def agent_view(db: Storage) -> ViewState:
    def f():
        runs = _df(db.q("SELECT ts,agent,ok,latency_ms,error,used_fallback,output FROM agent_runs ORDER BY id DESC LIMIT 100"))
        dec = _df(db.q("SELECT proposal_id,ts,approved,approved_qty,payload FROM risk_decisions ORDER BY ts DESC LIMIT 100"))
        if runs.empty and dec.empty:
            return ViewState("empty", None, "No agent runs or risk decisions yet.")
        if not dec.empty:
            dec["reasons"] = dec["payload"].map(lambda p: "; ".join(json.loads(p).get("reasons", [])))
            dec = dec.drop(columns=["payload"])
        return ViewState("ok", {"runs": runs, "decisions": dec})
    return _safe(f)


def sentiment_view(db: Storage, hours: float = 72) -> ViewState:
    def f():
        rows = db.recent_sentiment(hours)
        if not rows:
            return ViewState("empty", None, "No sentiment records yet.")
        df = pd.DataFrame([r.model_dump(mode="json") for r in rows])
        df["abs_impact"] = df["decayed_score"].abs()
        high = df[(df["materiality"] >= 0.7) & (df["abs_impact"] > 0.05)].sort_values("abs_impact", ascending=False)
        health = _df(db.q("SELECT provider, MAX(ts) last_ts, SUM(ok) ok, COUNT(*) n FROM provider_health GROUP BY provider"))
        conflicts = df.groupby("symbol")["label"].agg(lambda x: {"positive", "negative"} <= set(x))
        return ViewState("ok", {"feed": df.sort_values("published_at", ascending=False),
                                "distribution": df["label"].value_counts().rename_axis("label").reset_index(name="n"),
                                "high_impact": high, "conflicting_symbols": [k for k, v in conflicts.items() if v],
                                "models": df["model"].value_counts().to_dict(), "provider_health": health})
    return _safe(f)


def ops_view(db: Storage) -> ViewState:
    def f():
        rec = db.get_meta("reconciled")
        alerts = _df(db.q("SELECT ts,level,code,message FROM alerts ORDER BY id DESC LIMIT 50"))
        health = _df(db.q("SELECT ts,provider,ok,latency_ms,detail FROM provider_health ORDER BY id DESC LIMIT 50"))
        return ViewState("ok", {"reconcile": rec, "schema_version": db.schema_version(), "db_path": db.path,
                                "halt": db.halt_state(), "alerts": alerts, "provider_health": health,
                                "open_orders": len(db.open_orders())})
    return _safe(f)


def config_view(s: Settings) -> ViewState:
    return ViewState("ok", s.safe_dict())
