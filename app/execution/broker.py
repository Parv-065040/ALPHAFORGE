"""Alpaca PAPER broker adapter (REST). The live endpoint is not reachable through this code:
the base URL is a hard-coded constant and construction fails closed unless paper mode is confirmed."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.config import PAPER_BASE_URL, Settings
from app.data.http import HttpClient, NotFound, ProviderError
from app.schemas import AccountSnapshot, OrderStatus, Position
from app.timeutil import utcnow


class LivePathBlocked(RuntimeError):
    """Raised whenever paper mode cannot be confirmed. There is no live-trading path."""


class BrokerProtocol(Protocol):
    is_paper: bool
    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[Position]: ...
    def list_orders(self, after: datetime | None = None, nested: bool = True) -> list[dict]: ...
    def get_order_by_client_id(self, client_order_id: str) -> dict | None: ...
    def submit_order(self, payload: dict) -> dict: ...
    def cancel_order(self, broker_order_id: str) -> None: ...
    def get_asset(self, symbol: str) -> dict | None: ...


_STATUS_MAP = {
    "new": OrderStatus.ACCEPTED, "accepted": OrderStatus.ACCEPTED, "pending_new": OrderStatus.ACCEPTED,
    "accepted_for_bidding": OrderStatus.ACCEPTED, "held": OrderStatus.ACCEPTED, "pending_replace": OrderStatus.ACCEPTED,
    "replaced": OrderStatus.ACCEPTED, "pending_cancel": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED, "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED, "done_for_day": OrderStatus.EXPIRED, "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED, "suspended": OrderStatus.REJECTED, "stopped": OrderStatus.REJECTED,
    "calculated": OrderStatus.UNKNOWN,
}


def map_status(raw: str | None) -> OrderStatus:
    return _STATUS_MAP.get((raw or "").lower(), OrderStatus.UNKNOWN)


def assert_paper(settings: Settings) -> None:
    if not settings.alpaca_paper:
        raise LivePathBlocked("ALPACA_PAPER must be true; refusing to continue (no live path exists)")
    if not settings.has_alpaca_keys():
        raise LivePathBlocked("Alpaca paper keys missing")


class AlpacaPaperBroker:
    is_paper = True

    def __init__(self, settings: Settings, http: HttpClient | None = None):
        assert_paper(settings)
        self.http = http or HttpClient(
            PAPER_BASE_URL,
            {"APCA-API-KEY-ID": settings.secrets.alpaca_key_id, "APCA-API-SECRET-KEY": settings.secrets.alpaca_secret},
            timeout=settings.data.http_timeout_seconds, retries=settings.data.max_retries,
            backoff=settings.data.backoff_seconds, secrets=settings.secrets.values())
        if self.http.base_url != PAPER_BASE_URL:
            raise LivePathBlocked(f"broker base URL must be {PAPER_BASE_URL}")

    def get_account(self) -> AccountSnapshot:
        a = self.http.get("/v2/account")
        return AccountSnapshot(ts=utcnow(), equity=float(a["equity"]), cash=float(a["cash"]),
                               buying_power=float(a["buying_power"]),
                               last_equity=float(a.get("last_equity") or a["equity"]),
                               status=str(a.get("status", "")), source="broker")

    def get_positions(self) -> list[Position]:
        out = []
        for p in self.http.get("/v2/positions"):
            out.append(Position(symbol=p["symbol"], qty=float(p["qty"]), avg_entry_price=float(p["avg_entry_price"]),
                                market_value=float(p["market_value"]), unrealized_pl=float(p.get("unrealized_pl") or 0),
                                current_price=float(p.get("current_price") or 0)))
        return out

    def list_orders(self, after: datetime | None = None, nested: bool = True) -> list[dict]:
        params: dict[str, Any] = {"status": "all", "limit": 500, "direction": "desc", "nested": str(nested).lower()}
        if after:
            params["after"] = after.isoformat()
        return self.http.get("/v2/orders", params)

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        try:
            return self.http.get("/v2/orders:by_client_order_id", {"client_order_id": client_order_id})
        except NotFound:
            return None

    def submit_order(self, payload: dict) -> dict:
        return self.http.request_once("POST", "/v2/orders", json_body=payload)   # never auto-retried

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self.http.request_once("DELETE", f"/v2/orders/{broker_order_id}")
        except NotFound:
            return

    def get_asset(self, symbol: str) -> dict | None:
        try:
            return self.http.get(f"/v2/assets/{symbol}")
        except NotFound:
            return None

    def get_clock(self) -> dict:
        return self.http.get("/v2/clock")
