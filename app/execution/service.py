"""Single deterministic order-submission service: idempotent, reconciling, paper-only."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.config import Settings
from app.data.http import ProviderError, RateLimited, TimeoutOrNetwork
from app.execution.broker import BrokerProtocol, LivePathBlocked, map_status
from app.schemas import (Direction, OrderRecord, OrderStatus, RiskDecision, TERMINAL_STATUSES, TradeProposal)
from app.storage.db import Storage
from app.timeutil import parse_ts, session_status, utcnow

log = logging.getLogger("alphaforge.exec")


@dataclass
class ExecutionResult:
    status: Literal["submitted", "disabled", "blocked", "duplicate", "rejected", "unknown"]
    detail: str = ""
    order: OrderRecord | None = None


@dataclass
class ReconcileReport:
    ok: bool
    issues: list[str] = field(default_factory=list)       # blocking
    warnings: list[str] = field(default_factory=list)     # informational


def client_id_for(proposal_id: str) -> str:
    return f"af-{proposal_id}"


def _f(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


class ExecutionService:
    def __init__(self, settings: Settings, broker: BrokerProtocol, storage: Storage, now_fn=utcnow):
        self.s, self.broker, self.db, self.now = settings, broker, storage, now_fn

    # ----- guards -----
    @property
    def trading_enabled(self) -> bool:
        return bool(self.s.enable_paper_trading and self.s.alpaca_paper and getattr(self.broker, "is_paper", False))

    def is_reconciled(self, max_age_min: int = 20) -> bool:
        m = self.db.get_meta("reconciled")
        if not m or not m.get("ok"):
            return False
        return (self.now() - parse_ts(m["ts"])) <= timedelta(minutes=max_age_min)

    # ----- payloads -----
    def build_payload(self, p: TradeProposal, qty: int) -> dict:
        bps = self.s.risk.limit_buffer_bps / 1e4
        long = p.direction == Direction.LONG
        limit = round(p.entry_price * (1 + bps if long else 1 - bps), 2)
        return {
            "symbol": p.symbol, "qty": str(qty), "side": p.side, "type": "limit", "limit_price": f"{limit:.2f}",
            "time_in_force": "day" if p.horizon.value == "intraday" else "gtc",
            "order_class": "bracket", "client_order_id": client_id_for(p.id),
            "take_profit": {"limit_price": f"{round(p.target_price, 2):.2f}"},
            "stop_loss": {"stop_price": f"{round(p.stop_price, 2):.2f}"},
        }

    # ----- submit -----
    def execute(self, p: TradeProposal, d: RiskDecision) -> ExecutionResult:
        if not self.trading_enabled:
            return ExecutionResult("disabled", "ENABLE_PAPER_TRADING is false (or paper mode unconfirmed); no order sent")
        stored = self.db.get_decision(p.id)
        if (not d.approved or d.proposal_id != p.id or stored is None or not stored.approved
                or stored.approved_qty != d.approved_qty or d.approved_qty < 1):
            return ExecutionResult("blocked", "no stored, matching, approved risk decision")
        now = self.now()
        if now - stored.decided_at > timedelta(minutes=self.s.risk.approval_ttl_minutes):
            return ExecutionResult("blocked", "risk approval expired")
        if p.expires_at <= now:
            return ExecutionResult("blocked", "proposal expired")
        if not self.is_reconciled():
            return ExecutionResult("blocked", "broker state not reconciled")
        qty = min(d.approved_qty, p.qty)
        cid = client_id_for(p.id)
        existing = self.db.get_order(cid)
        if existing is not None:
            self.sync_orders()
            return ExecutionResult("duplicate", "order already exists for this proposal", self.db.get_order(cid))
        rec = OrderRecord(client_order_id=cid, proposal_id=p.id, symbol=p.symbol, side=p.side, qty=qty,
                          status=OrderStatus.PENDING_SUBMIT, submitted_at=now, updated_at=now, detail="pending submit")
        self.db.upsert_order(rec, ref_price=p.entry_price, strategy=p.strategy)
        payload = self.build_payload(p, qty)
        return self._submit_with_reconcile(rec, payload)

    def _apply_raw(self, rec: OrderRecord, raw: dict, detail: str = "") -> OrderRecord:
        new = rec.model_copy(update={
            "broker_order_id": raw.get("id") or rec.broker_order_id, "raw_status": raw.get("status", ""),
            "status": map_status(raw.get("status")), "filled_qty": _f(raw.get("filled_qty")),
            "avg_fill_price": _f(raw.get("filled_avg_price"), 0.0) or None, "updated_at": self.now(),
            "detail": detail or rec.detail})
        self._record_fill_delta(rec, new)
        self.db.upsert_order(new)
        return new

    def _submit_with_reconcile(self, rec: OrderRecord, payload: dict) -> ExecutionResult:
        for attempt in range(2):
            try:
                raw = self.broker.submit_order(payload)
                new = self._apply_raw(rec, raw, "submitted")
                return ExecutionResult("submitted", "submitted to Alpaca paper", new)
            except ProviderError as e:
                transient = isinstance(e, (TimeoutOrNetwork, RateLimited)) or (e.status or 0) >= 500
                if not transient:      # 4xx other than 429: broker rejected the order
                    new = rec.model_copy(update={"status": OrderStatus.REJECTED, "updated_at": self.now(),
                                                 "detail": f"rejected: {e}"[:300]})
                    self.db.upsert_order(new)
                    self.db.alert("warning", "ORDER_REJECTED", f"{rec.symbol}: {e}")
                    return ExecutionResult("rejected", str(e), new)
                # Timeout/unknown outcome: NEVER assume failure. Reconcile by client_order_id first.
                try:
                    raw = self.broker.get_order_by_client_id(rec.client_order_id)
                except ProviderError as e2:
                    new = rec.model_copy(update={"status": OrderStatus.UNKNOWN, "updated_at": self.now(),
                                                 "detail": f"submit outcome unknown; reconcile failed: {e2}"[:300]})
                    self.db.upsert_order(new)
                    self.db.alert("error", "ORDER_UNKNOWN", f"{rec.symbol}: outcome unknown")
                    self.db.set_meta("reconciled", {"ok": False, "ts": self.now().isoformat()})
                    return ExecutionResult("unknown", "submit outcome unknown", new)
                if raw is not None:
                    new = self._apply_raw(rec, raw, "recovered after submit timeout via reconciliation")
                    return ExecutionResult("submitted", "order found at broker after timeout", new)
                if attempt == 0:
                    log.warning("order %s absent at broker after timeout; retrying once", rec.client_order_id)
                    continue          # confirmed absent -> single safe retry (same idempotency key)
        new = rec.model_copy(update={"status": OrderStatus.UNKNOWN, "updated_at": self.now(),
                                     "detail": "submit failed twice; outcome unconfirmed"})
        self.db.upsert_order(new)
        self.db.set_meta("reconciled", {"ok": False, "ts": self.now().isoformat()})
        return ExecutionResult("unknown", "submit failed twice; outcome unconfirmed", new)

    # ----- order sync -----
    def _record_fill_delta(self, old: OrderRecord, new: OrderRecord) -> None:
        dq = new.filled_qty - old.filled_qty
        if dq > 1e-9 and new.avg_fill_price:
            old_val = old.filled_qty * (old.avg_fill_price or 0.0)
            px = (new.filled_qty * new.avg_fill_price - old_val) / dq
            self.db.add_fill(new.client_order_id, self.now(), dq, px if px > 0 else new.avg_fill_price)

    def sync_orders(self) -> list[OrderRecord]:
        open_orders = self.db.open_orders()
        recent = self.db.all_orders(limit=50)
        if not open_orders and not recent:
            return []
        since = min([o.submitted_at for o in (open_orders or recent)]) - timedelta(minutes=5)
        raws = self.broker.list_orders(after=since, nested=True)
        flat: dict[str, tuple[dict, str | None]] = {}
        for r in raws:
            flat[r.get("client_order_id", "")] = (r, None)
            for leg in r.get("legs") or []:
                flat[leg.get("client_order_id", "")] = (leg, r.get("client_order_id"))
        # register bracket legs of known parents
        for cid, (raw, parent_cid) in list(flat.items()):
            if parent_cid and self.db.get_order(cid) is None and self.db.get_order(parent_cid) is not None:
                par = self.db.get_order(parent_cid)
                strat = next((x["strategy"] for x in self.db.q("SELECT strategy FROM orders WHERE client_order_id=?",
                                                              (parent_cid,))), None)
                leg = OrderRecord(client_order_id=cid, broker_order_id=raw.get("id"), proposal_id=None,
                                  symbol=raw.get("symbol", par.symbol), side=raw.get("side", ""),
                                  qty=_f(raw.get("qty")), status=OrderStatus.ACCEPTED, submitted_at=self.now(),
                                  updated_at=self.now(), detail=f"bracket leg of {parent_cid}")
                self.db.upsert_order(leg, strategy=strat)
        updated: list[OrderRecord] = []
        for o in self.db.open_orders():
            raw = flat.get(o.client_order_id, (None, None))[0]
            if raw is None:
                raw = self.broker.get_order_by_client_id(o.client_order_id)
            if raw is None:
                if o.status == OrderStatus.PENDING_SUBMIT and self.now() - o.submitted_at > timedelta(minutes=2):
                    n = o.model_copy(update={"status": OrderStatus.UNKNOWN, "updated_at": self.now(),
                                             "detail": "pending order not found at broker"})
                    self.db.upsert_order(n)
                    updated.append(n)
                continue
            n = self._apply_raw(o, raw)
            updated.append(n)
            if n.proposal_id and n.filled_qty > 0:
                prop = self.db.get_proposal(n.proposal_id)
                if prop:
                    self.db.set_position_meta(prop.symbol, prop.horizon.value, prop.strategy, prop.id, self.now())
        return updated

    # ----- reconciliation -----
    def reconcile(self) -> ReconcileReport:
        rep = ReconcileReport(True)
        try:
            acct = self.broker.get_account()
            positions = self.broker.get_positions()
            self.db.save_account(acct)
            self.db.save_positions(self.now(), positions)
            self.sync_orders()
        except (ProviderError, LivePathBlocked) as e:
            rep.ok = False
            rep.issues.append(f"broker unreachable/unreadable: {e}")
        else:
            for o in self.db.open_orders():
                if o.status == OrderStatus.UNKNOWN:
                    rep.ok = False
                    rep.issues.append(f"order {o.client_order_id} ({o.symbol}) unresolved")
            held = {p.symbol for p in positions}
            meta = self.db.position_meta()
            for sym in list(meta):
                if sym not in held:
                    self.db.clear_position_meta(sym)
            for sym in held - set(meta):
                rep.warnings.append(f"position {sym} not opened by ALPHAFORGE (untracked; excluded from intraday management)")
            if acct.status and acct.status.upper() not in {"ACTIVE", "ACCOUNT_UPDATED"}:
                rep.ok = False
                rep.issues.append(f"account status {acct.status}")
        self.db.set_meta("reconciled", {"ok": rep.ok, "ts": self.now().isoformat(), "issues": rep.issues})
        if not rep.ok:
            self.db.alert("error", "RECONCILE_FAILED", "; ".join(rep.issues))
        return rep

    # ----- trade management -----
    def cancel_stale_entries(self) -> int:
        n = 0
        limit = timedelta(minutes=self.s.risk.entry_order_max_age_minutes)
        for o in self.db.open_orders():
            if (o.proposal_id and o.status == OrderStatus.ACCEPTED and o.broker_order_id
                    and self.now() - o.submitted_at > limit and self.trading_enabled):
                try:
                    self.broker.cancel_order(o.broker_order_id)
                    n += 1
                except ProviderError as e:
                    self.db.alert("warning", "CANCEL_FAILED", f"{o.symbol}: {e}")
        if n:
            self.sync_orders()
        return n

    def flatten_intraday(self, force: bool = False) -> list[str]:
        """Close intraday-tagged positions near the close (or immediately when force). Exit orders only."""
        st = session_status(self.now())
        due = force or (st.is_open and st.minutes_to_close is not None
                        and st.minutes_to_close <= self.s.session.intraday_flatten_minutes_before_close)
        if not due:
            return []
        closed: list[str] = []
        positions = {p.symbol: p for p in self.broker.get_positions()}
        for sym, m in self.db.position_meta().items():
            pos = positions.get(sym)
            if m["horizon"] != "intraday" or pos is None or pos.qty == 0:
                continue
            cid = f"af-flat-{sym}-{st.session_date.isoformat()}"
            if self.db.get_order(cid) is not None:
                continue
            if not self.trading_enabled:
                log.info("would flatten %s (trading disabled)", sym)
                continue
            for o in self.db.open_orders():
                if o.symbol == sym and o.broker_order_id:
                    self.broker.cancel_order(o.broker_order_id)
            side = "sell" if pos.qty > 0 else "buy"
            rec = OrderRecord(client_order_id=cid, symbol=sym, side=side, qty=abs(pos.qty),
                              submitted_at=self.now(), updated_at=self.now(), detail="intraday flatten")
            self.db.upsert_order(rec, strategy=m["strategy"])
            self._submit_with_reconcile(rec, {"symbol": sym, "qty": str(int(abs(pos.qty))), "side": side,
                                              "type": "market", "time_in_force": "day", "client_order_id": cid})
            closed.append(sym)
        return closed

    def paper_smoke_test(self, symbol: str, ref_price: float) -> dict:
        """Explicit operator action: far-from-market 1-share limit buy, then cancel. Never run automatically."""
        if not self.trading_enabled:
            return {"ok": False, "detail": "ENABLE_PAPER_TRADING is false"}
        cid = f"af-smoke-{int(self.now().timestamp())}"
        payload = {"symbol": symbol, "qty": "1", "side": "buy", "type": "limit",
                   "limit_price": f"{max(round(ref_price * 0.5, 2), 0.01):.2f}", "time_in_force": "day",
                   "client_order_id": cid}
        raw = self.broker.submit_order(payload)
        if raw.get("id"):
            self.broker.cancel_order(raw["id"])
        return {"ok": True, "detail": f"submitted and cancelled test order {cid}", "status": raw.get("status")}
