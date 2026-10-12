"""Test doubles. FakeBroker simulates Alpaca paper order behaviour incl. timeouts/rejections. NEVER networked."""
from __future__ import annotations

import uuid
from datetime import datetime

from app.data.http import ProviderError, TimeoutOrNetwork
from app.schemas import AccountSnapshot, Position
from app.timeutil import utcnow


class FakeBroker:
    is_paper = True

    def __init__(self, equity: float = 100_000.0, last_equity: float | None = None, status: str = "ACTIVE"):
        self.equity, self.last_equity, self.status = equity, last_equity or equity, status
        self.positions: list[Position] = []
        self.orders: dict[str, dict] = {}          # by client_order_id
        self.submit_calls = 0
        self.cancelled: list[str] = []
        self.mode = "ok"      # ok | reject | timeout_exists | timeout_absent_once | timeout_always | down
        self.shortable = True
        self.payloads: list[dict] = []

    # ---- reads
    def get_account(self) -> AccountSnapshot:
        if self.mode == "down":
            raise TimeoutOrNetwork("broker down", None, "network")
        return AccountSnapshot(ts=utcnow(), equity=self.equity, cash=self.equity * 0.9, buying_power=self.equity * 2,
                               last_equity=self.last_equity, status=self.status, source="broker")

    def get_positions(self):
        if self.mode == "down":
            raise TimeoutOrNetwork("broker down", None, "network")
        return list(self.positions)

    def list_orders(self, after=None, nested=True):
        if self.mode == "down":
            raise TimeoutOrNetwork("broker down", None, "network")
        return [o for o in self.orders.values() if not o.get("_leg")]

    def get_order_by_client_id(self, cid):
        if self.mode == "down" or (self.mode == "timeout_always_lookup"):
            raise TimeoutOrNetwork("lookup failed", None, "network")
        if cid in self.orders:
            return self.orders[cid]
        for o in list(self.orders.values()):
            for leg in o.get("legs", []):
                if leg["client_order_id"] == cid:
                    return leg
        return None

    def get_asset(self, symbol):
        return {"symbol": symbol, "tradable": True, "shortable": self.shortable}

    # ---- writes
    def _create(self, payload: dict) -> dict:
        oid = str(uuid.uuid4())
        o = {"id": oid, "client_order_id": payload["client_order_id"], "status": "accepted", "symbol": payload["symbol"],
             "side": payload["side"], "qty": payload["qty"], "filled_qty": "0", "filled_avg_price": None, "legs": []}
        if payload.get("order_class") == "bracket":
            opp = "sell" if payload["side"] == "buy" else "buy"
            for kind in ("take_profit", "stop_loss"):
                o["legs"].append({"id": str(uuid.uuid4()), "client_order_id": f"{kind}-{oid[:8]}", "status": "held",
                                  "symbol": payload["symbol"], "side": opp, "qty": payload["qty"], "filled_qty": "0",
                                  "filled_avg_price": None})
        self.orders[payload["client_order_id"]] = o
        return o

    def submit_order(self, payload: dict) -> dict:
        self.submit_calls += 1
        self.payloads.append(payload)
        if self.mode == "reject":
            raise ProviderError("HTTP 422: insufficient buying power", 422, "http_error")
        if self.mode == "timeout_exists":
            self._create(payload)
            raise TimeoutOrNetwork("read timeout", None, "network")
        if self.mode == "timeout_absent_once" and self.submit_calls == 1:
            raise TimeoutOrNetwork("connect timeout", None, "network")
        if self.mode == "timeout_always":
            raise TimeoutOrNetwork("timeout", None, "network")
        return self._create(payload)

    def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)
        for o in self.orders.values():
            if o["id"] == broker_order_id:
                o["status"] = "canceled"

    # ---- test helpers
    def fill(self, cid: str, qty: float, price: float, full: bool = True):
        o = self.orders[cid]
        o["filled_qty"], o["filled_avg_price"] = str(qty), str(price)
        o["status"] = "filled" if full else "partially_filled"
