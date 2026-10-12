"""SQLite persistence with versioned migrations and restart recovery support."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.schemas import (AccountSnapshot, OrderRecord, OrderStatus, Position, RiskDecision,
                         SentimentResult, Signal, TradeProposal, NewsItem, TERMINAL_STATUSES)
from app.timeutil import parse_ts, utcnow

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE news(id TEXT PRIMARY KEY, headline TEXT, source TEXT, url TEXT, published_at TEXT,
        ingested_at TEXT, symbols TEXT, hash TEXT, payload TEXT);
    CREATE TABLE sentiment(id INTEGER PRIMARY KEY AUTOINCREMENT, news_id TEXT, symbol TEXT, label TEXT,
        score REAL, confidence REAL, novelty REAL, materiality REAL, decayed_score REAL, model TEXT,
        published_at TEXT, computed_at TEXT, payload TEXT);
    CREATE INDEX ix_sent_sym ON sentiment(symbol, published_at);
    CREATE TABLE scans(id TEXT PRIMARY KEY, ts TEXT, universe_size INTEGER, n_candidates INTEGER,
        n_signals INTEGER, regime TEXT, blocked_reason TEXT, payload TEXT);
    CREATE TABLE candidates(id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id TEXT, symbol TEXT, strategy TEXT,
        score REAL, payload TEXT);
    CREATE TABLE signals(id TEXT PRIMARY KEY, scan_id TEXT, ts TEXT, symbol TEXT, strategy TEXT,
        direction TEXT, score REAL, expires_at TEXT, payload TEXT);
    CREATE TABLE proposals(id TEXT PRIMARY KEY, signal_id TEXT, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,
        strategy TEXT, horizon TEXT, expires_at TEXT, payload TEXT);
    CREATE TABLE risk_decisions(proposal_id TEXT PRIMARY KEY, ts TEXT, approved INTEGER, approved_qty INTEGER,
        payload TEXT);
    CREATE TABLE orders(client_order_id TEXT PRIMARY KEY, broker_order_id TEXT, proposal_id TEXT, symbol TEXT,
        side TEXT, qty REAL, filled_qty REAL, avg_fill_price REAL, status TEXT, raw_status TEXT,
        submitted_at TEXT, updated_at TEXT, detail TEXT);
    CREATE TABLE order_events(id INTEGER PRIMARY KEY AUTOINCREMENT, client_order_id TEXT, ts TEXT,
        from_status TEXT, to_status TEXT, detail TEXT);
    CREATE TABLE fills(id INTEGER PRIMARY KEY AUTOINCREMENT, client_order_id TEXT, ts TEXT, qty REAL, price REAL);
    CREATE TABLE account_snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, equity REAL, cash REAL,
        buying_power REAL, last_equity REAL, status TEXT, source TEXT);
    CREATE TABLE position_snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, qty REAL,
        avg_entry REAL, market_value REAL, unrealized_pl REAL, current_price REAL);
    CREATE TABLE positions_meta(symbol TEXT PRIMARY KEY, horizon TEXT, strategy TEXT, proposal_id TEXT,
        opened_at TEXT);
    CREATE TABLE agent_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent TEXT, ok INTEGER,
        latency_ms REAL, error TEXT, used_fallback INTEGER, output TEXT);
    CREATE TABLE provider_health(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, provider TEXT, ok INTEGER,
        latency_ms REAL, detail TEXT);
    CREATE TABLE alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, code TEXT, message TEXT,
        ack INTEGER DEFAULT 0);
    CREATE TABLE config_versions(hash TEXT PRIMARY KEY, ts TEXT, payload TEXT);
    CREATE TABLE recommendations(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, strategy TEXT, text TEXT,
        evidence TEXT);
    CREATE TABLE market_features(id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id TEXT, ts TEXT, symbol TEXT,
        payload TEXT);
    """),
    (2, """
    ALTER TABLE orders ADD COLUMN ref_price REAL;
    ALTER TABLE orders ADD COLUMN strategy TEXT;
    CREATE INDEX ix_orders_status ON orders(status);
    """),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class Storage:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=15)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # ----- infra -----
    def migrate(self) -> int:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT MAX(version) v FROM schema_version").fetchone()
            current = row["v"] or 0
            for ver, sql in MIGRATIONS:
                if ver > current:
                    with self._conn:
                        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                            self._conn.execute(stmt)
                        self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (ver,))
            return SCHEMA_VERSION

    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"] or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def tx(self):
        with self._lock, self._conn:
            yield self._conn

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def x(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, tuple(params))

    # ----- meta -----
    def set_meta(self, key: str, value: Any) -> None:
        self.x("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, json.dumps(value, default=str)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        r = self.q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def origin(self) -> str:
        return self.get_meta("origin", "live")

    def record_config(self, h: str, payload: dict) -> None:
        self.x("INSERT OR IGNORE INTO config_versions(hash,ts,payload) VALUES(?,?,?)",
               (h, _iso(utcnow()), json.dumps(payload, default=str)))

    # ----- news / sentiment -----
    def save_news(self, items: list[NewsItem], hashes: dict[str, str] | None = None) -> int:
        n = 0
        with self.tx() as c:
            for it in items:
                cur = c.execute(
                    "INSERT OR IGNORE INTO news(id,headline,source,url,published_at,ingested_at,symbols,hash,payload)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (it.id, it.headline, it.source, it.url, _iso(it.published_at), _iso(it.ingested_at),
                     json.dumps(it.symbols), (hashes or {}).get(it.id), it.model_dump_json()))
                n += cur.rowcount
        return n

    def known_news_hashes(self, since: datetime) -> set[str]:
        rows = self.q("SELECT hash FROM news WHERE published_at>=? AND hash IS NOT NULL", (_iso(since),))
        return {r["hash"] for r in rows}

    def save_sentiment(self, results: list[SentimentResult]) -> None:
        with self.tx() as c:
            for r in results:
                c.execute(
                    "INSERT INTO sentiment(news_id,symbol,label,score,confidence,novelty,materiality,decayed_score,"
                    "model,published_at,computed_at,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r.news_id, r.symbol, r.label, r.score, r.confidence, r.novelty, r.materiality,
                     r.decayed_score, r.model, _iso(r.published_at), _iso(r.computed_at), r.model_dump_json()))

    def recent_sentiment(self, hours: float = 48, limit: int = 500, now: datetime | None = None) -> list[SentimentResult]:
        from datetime import timedelta
        since = _iso((now or utcnow()) - timedelta(hours=hours))
        rows = self.q("SELECT payload FROM sentiment WHERE published_at>=? ORDER BY published_at DESC LIMIT ?",
                      (since, limit))
        return [SentimentResult.model_validate_json(r["payload"]) for r in rows]

    # ----- scans / signals / proposals -----
    def save_scan(self, scan_id: str, ts: datetime, universe_size: int, candidates: list, signals: list[Signal],
                  regime: str, blocked_reason: str | None, features: dict[str, dict] | None = None,
                  no_trade: list[dict] | None = None, regime_reasons: list[str] | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO scans VALUES(?,?,?,?,?,?,?,?)",
                      (scan_id, _iso(ts), universe_size, len(candidates), len(signals), regime, blocked_reason,
                       json.dumps({"regime": regime, "blocked": blocked_reason, "regime_reasons": regime_reasons or [],
                                   "no_trade": (no_trade or [])[:400]}, default=str)))
            for cd in candidates:
                c.execute("INSERT INTO candidates(scan_id,symbol,strategy,score,payload) VALUES(?,?,?,?,?)",
                          (scan_id, cd.symbol, cd.strategy, cd.score, cd.model_dump_json()))
            for s in signals:
                c.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
                          (s.id, scan_id, _iso(s.created_at), s.symbol, s.strategy, s.direction.value, s.score,
                           _iso(s.expires_at), s.model_dump_json()))
            for sym, f in (features or {}).items():
                c.execute("INSERT INTO market_features(scan_id,ts,symbol,payload) VALUES(?,?,?,?)",
                          (scan_id, _iso(ts), sym, json.dumps(f, default=float)))

    def save_proposal(self, p: TradeProposal) -> None:
        self.x("INSERT OR IGNORE INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?)",
               (p.id, p.signal_id, _iso(p.created_at), p.symbol, p.side, p.qty, p.strategy, p.horizon.value,
                _iso(p.expires_at), p.model_dump_json()))

    def get_proposal(self, pid: str) -> TradeProposal | None:
        r = self.q("SELECT payload FROM proposals WHERE id=?", (pid,))
        return TradeProposal.model_validate_json(r[0]["payload"]) if r else None

    def save_decision(self, d: RiskDecision) -> None:
        self.x("INSERT OR REPLACE INTO risk_decisions VALUES(?,?,?,?,?)",
               (d.proposal_id, _iso(d.decided_at), int(d.approved), d.approved_qty, d.model_dump_json()))

    def get_decision(self, proposal_id: str) -> RiskDecision | None:
        r = self.q("SELECT payload FROM risk_decisions WHERE proposal_id=?", (proposal_id,))
        return RiskDecision.model_validate_json(r[0]["payload"]) if r else None

    def entries_today(self, day_start_utc: datetime) -> int:
        r = self.q("SELECT COUNT(*) n FROM orders WHERE proposal_id IS NOT NULL AND submitted_at>=? AND "
                   "(status NOT IN ('rejected','canceled','expired') OR filled_qty>0)", (_iso(day_start_utc),))
        return int(r[0]["n"])

    # ----- orders -----
    def upsert_order(self, o: OrderRecord, ref_price: float | None = None, strategy: str | None = None) -> None:
        with self.tx() as c:
            prev = c.execute("SELECT status FROM orders WHERE client_order_id=?", (o.client_order_id,)).fetchone()
            if prev is None:
                c.execute("INSERT INTO orders(client_order_id,broker_order_id,proposal_id,symbol,side,qty,filled_qty,"
                          "avg_fill_price,status,raw_status,submitted_at,updated_at,detail,ref_price,strategy) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (o.client_order_id, o.broker_order_id, o.proposal_id, o.symbol, o.side, o.qty, o.filled_qty,
                           o.avg_fill_price, o.status.value, o.raw_status, _iso(o.submitted_at), _iso(o.updated_at),
                           o.detail, ref_price, strategy))
                c.execute("INSERT INTO order_events(client_order_id,ts,from_status,to_status,detail) VALUES(?,?,?,?,?)",
                          (o.client_order_id, _iso(o.updated_at), None, o.status.value, o.detail))
            else:
                c.execute("UPDATE orders SET broker_order_id=COALESCE(?,broker_order_id), filled_qty=?, "
                          "avg_fill_price=?, status=?, raw_status=?, updated_at=?, detail=? WHERE client_order_id=?",
                          (o.broker_order_id, o.filled_qty, o.avg_fill_price, o.status.value, o.raw_status,
                           _iso(o.updated_at), o.detail, o.client_order_id))
                if prev["status"] != o.status.value:
                    c.execute("INSERT INTO order_events(client_order_id,ts,from_status,to_status,detail) "
                              "VALUES(?,?,?,?,?)",
                              (o.client_order_id, _iso(o.updated_at), prev["status"], o.status.value, o.detail))

    def _order_from_row(self, r: sqlite3.Row) -> OrderRecord:
        return OrderRecord(client_order_id=r["client_order_id"], broker_order_id=r["broker_order_id"],
                           proposal_id=r["proposal_id"], symbol=r["symbol"], side=r["side"], qty=r["qty"],
                           filled_qty=r["filled_qty"] or 0.0, avg_fill_price=r["avg_fill_price"],
                           status=OrderStatus(r["status"]), raw_status=r["raw_status"] or "",
                           submitted_at=parse_ts(r["submitted_at"]), updated_at=parse_ts(r["updated_at"]),
                           detail=r["detail"] or "")

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        r = self.q("SELECT * FROM orders WHERE client_order_id=?", (client_order_id,))
        return self._order_from_row(r[0]) if r else None

    def open_orders(self) -> list[OrderRecord]:
        term = tuple(s.value for s in TERMINAL_STATUSES)
        rows = self.q(f"SELECT * FROM orders WHERE status NOT IN ({','.join('?' * len(term))})", term)
        return [self._order_from_row(r) for r in rows]

    def all_orders(self, limit: int = 500) -> list[OrderRecord]:
        return [self._order_from_row(r) for r in
                self.q("SELECT * FROM orders ORDER BY submitted_at DESC LIMIT ?", (limit,))]

    def order_events(self, client_order_id: str) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM order_events WHERE client_order_id=? ORDER BY id",
                                        (client_order_id,))]

    def add_fill(self, client_order_id: str, ts: datetime, qty: float, price: float) -> None:
        self.x("INSERT INTO fills(client_order_id,ts,qty,price) VALUES(?,?,?,?)",
               (client_order_id, _iso(ts), qty, price))

    def fills_with_orders(self) -> list[dict]:
        rows = self.q("SELECT f.ts, f.qty, f.price, o.symbol, o.side, o.client_order_id, o.strategy, o.ref_price, "
                      "o.proposal_id FROM fills f JOIN orders o ON o.client_order_id=f.client_order_id "
                      "ORDER BY f.ts, f.id")
        return [dict(r) for r in rows]

    # ----- account / positions -----
    def save_account(self, a: AccountSnapshot) -> None:
        self.x("INSERT INTO account_snapshots(ts,equity,cash,buying_power,last_equity,status,source) "
               "VALUES(?,?,?,?,?,?,?)",
               (_iso(a.ts), a.equity, a.cash, a.buying_power, a.last_equity, a.status, a.source))

    def account_history(self, limit: int = 5000) -> list[AccountSnapshot]:
        rows = self.q("SELECT * FROM (SELECT * FROM account_snapshots ORDER BY ts DESC LIMIT ?) ORDER BY ts", (limit,))
        return [AccountSnapshot(ts=parse_ts(r["ts"]), equity=r["equity"], cash=r["cash"],
                                buying_power=r["buying_power"], last_equity=r["last_equity"],
                                status=r["status"] or "", source=r["source"]) for r in rows]

    def save_positions(self, ts: datetime, positions: list[Position]) -> None:
        with self.tx() as c:
            for p in positions:
                c.execute("INSERT INTO position_snapshots(ts,symbol,qty,avg_entry,market_value,unrealized_pl,"
                          "current_price) VALUES(?,?,?,?,?,?,?)",
                          (_iso(ts), p.symbol, p.qty, p.avg_entry_price, p.market_value, p.unrealized_pl,
                           p.current_price))
            c.execute("INSERT INTO meta(key,value) VALUES('positions_ts',?) ON CONFLICT(key) DO UPDATE "
                      "SET value=excluded.value", (json.dumps(_iso(ts)),))

    def latest_positions(self) -> list[Position]:
        ts = self.get_meta("positions_ts")
        if not ts:
            return []
        rows = self.q("SELECT * FROM position_snapshots WHERE ts=?", (ts,))
        return [Position(symbol=r["symbol"], qty=r["qty"], avg_entry_price=r["avg_entry"],
                         market_value=r["market_value"], unrealized_pl=r["unrealized_pl"] or 0.0,
                         current_price=r["current_price"] or 0.0) for r in rows]

    def set_position_meta(self, symbol: str, horizon: str, strategy: str, proposal_id: str, opened_at: datetime) -> None:
        self.x("INSERT OR REPLACE INTO positions_meta VALUES(?,?,?,?,?)",
               (symbol, horizon, strategy, proposal_id, _iso(opened_at)))

    def position_meta(self) -> dict[str, dict]:
        return {r["symbol"]: dict(r) for r in self.q("SELECT * FROM positions_meta")}

    def clear_position_meta(self, symbol: str) -> None:
        self.x("DELETE FROM positions_meta WHERE symbol=?", (symbol,))

    # ----- ops -----
    def log_agent_run(self, agent: str, ok: bool, latency_ms: float, error: str | None, used_fallback: bool,
                      output: dict | None) -> None:
        self.x("INSERT INTO agent_runs(ts,agent,ok,latency_ms,error,used_fallback,output) VALUES(?,?,?,?,?,?,?)",
               (_iso(utcnow()), agent, int(ok), latency_ms, error, int(used_fallback),
                json.dumps(output, default=str) if output is not None else None))

    def log_provider(self, provider: str, ok: bool, latency_ms: float, detail: str = "") -> None:
        self.x("INSERT INTO provider_health(ts,provider,ok,latency_ms,detail) VALUES(?,?,?,?,?)",
               (_iso(utcnow()), provider, int(ok), latency_ms, detail[:500]))

    def alert(self, level: str, code: str, message: str) -> None:
        self.x("INSERT INTO alerts(ts,level,code,message) VALUES(?,?,?,?)", (_iso(utcnow()), level, code, message[:1000]))

    def add_recommendation(self, strategy: str, text: str, evidence: dict) -> None:
        self.x("INSERT INTO recommendations(ts,strategy,text,evidence) VALUES(?,?,?,?)",
               (_iso(utcnow()), strategy, text, json.dumps(evidence, default=str)))

    def set_halt(self, halted: bool, reason: str = "") -> None:
        self.set_meta("operator_halt", {"halted": halted, "reason": reason, "ts": _iso(utcnow())})

    def halt_state(self) -> dict:
        return self.get_meta("operator_halt", {"halted": False, "reason": ""})
