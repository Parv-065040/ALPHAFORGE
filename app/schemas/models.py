"""Typed contracts shared by backend, storage and dashboard."""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def normalize_symbol(s: str) -> str:
    """Normalize vendor symbols: upper-case, strip, BRK-B / BRK/B -> BRK.B."""
    s = s.strip().upper().replace("/", ".").replace("-", ".")
    if not SYMBOL_RE.match(s):
        raise ValueError(f"invalid symbol: {s!r}")
    return s


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (UTC)")
    return v


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class Horizon(str, Enum):
    INTRADAY = "intraday"
    SWING = "swing"


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "pending_submit"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


TERMINAL_STATUSES = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


# ---------- news / sentiment ----------
class NewsItem(_M):
    id: str
    headline: str
    summary: str = ""
    source: str = ""            # publisher
    url: str = ""
    symbols: list[str] = []
    published_at: datetime
    ingested_at: datetime

    _v1 = field_validator("published_at", "ingested_at")(_aware)


class SentimentResult(_M):
    news_id: str
    symbol: str
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=-1, le=1)          # p_pos - p_neg; NOT a calibrated return probability
    confidence: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    materiality: float = Field(ge=0, le=1)
    decayed_score: float = Field(ge=-1, le=1)
    model: str
    published_at: datetime
    computed_at: datetime
    headline: str = ""
    url: str = ""
    source: str = ""
    evidence: str = ""


class SymbolSentiment(_M):
    symbol: str
    score: float = Field(ge=-1, le=1)
    n_items: int
    max_materiality: float
    min_novelty_of_top: float = 0.0
    newest_age_hours: float | None = None
    conflicting: bool = False
    stale: bool = True
    top_news_id: str | None = None
    model: str = ""


# ---------- scanning ----------
class Candidate(_M):
    symbol: str
    strategy: str
    direction: Direction
    score: float
    components: dict[str, float]
    features: dict[str, float]
    reasons: list[str] = []


class Signal(_M):
    id: str = Field(default_factory=lambda: new_id("sig"))
    symbol: str
    strategy: str
    direction: Direction
    horizon: Horizon
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)
    score: float
    components: dict[str, float]
    created_at: datetime
    expires_at: datetime
    reasons: list[str] = []
    catalyst_news_id: str | None = None
    avg_dollar_volume: float = 0.0
    avg_volume: float = 0.0


class TradeProposal(_M):
    id: str = Field(default_factory=lambda: new_id("prop"))
    signal_id: str
    symbol: str
    direction: Direction
    strategy: str
    horizon: Horizon
    qty: int = Field(gt=0)
    entry_price: float = Field(gt=0)      # reference price
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)
    created_at: datetime
    expires_at: datetime
    rationale: str = ""
    source: str = "quant_only"            # quant_only | agent_pm
    agent_confidence: float | None = None
    avg_dollar_volume: float = 0.0
    avg_volume: float = 0.0

    @property
    def side(self) -> str:
        return "buy" if self.direction == Direction.LONG else "sell"


class RiskDecision(_M):
    proposal_id: str
    approved: bool
    approved_qty: int = 0
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    decided_at: datetime
    limits_snapshot: dict[str, float] = {}


# ---------- broker ----------
class AccountSnapshot(_M):
    ts: datetime
    equity: float
    cash: float
    buying_power: float
    last_equity: float          # prior-close equity as reported by broker (start-of-day proxy)
    status: str = ""
    source: Literal["broker", "demo"] = "broker"


class Position(_M):
    symbol: str
    qty: float                  # signed
    avg_entry_price: float
    market_value: float
    unrealized_pl: float = 0.0
    current_price: float = 0.0


class OrderRecord(_M):
    client_order_id: str
    broker_order_id: str | None = None
    proposal_id: str | None = None
    symbol: str
    side: str
    qty: float
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING_SUBMIT
    raw_status: str = ""
    submitted_at: datetime
    updated_at: datetime
    detail: str = ""


# ---------- agent outputs (validated LLM structures) ----------
class CatalystAssessment(_M):
    symbol: str
    materiality: Literal["low", "medium", "high"]
    direction: Literal["bullish", "bearish", "neutral", "unclear"]
    summary: str = Field(max_length=500)
    evidence_urls: list[str] = Field(default_factory=list, max_length=5)
    uncertainty: float = Field(ge=0, le=1)


class RegimeAssessment(_M):
    regime: Literal["risk_on", "neutral", "risk_off", "unclear"]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=500)


class PMDecision(_M):
    symbol: str
    action: Literal["propose", "skip"]
    size_multiplier: float = Field(ge=0.0, le=1.0)   # can only scale size DOWN
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=500)


class AgentResult(_M):
    agent: str
    ok: bool
    output: dict | None = None
    error: str | None = None
    latency_ms: float = 0.0
    used_fallback: bool = False
