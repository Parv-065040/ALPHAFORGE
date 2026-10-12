from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings  # noqa: E402
from app.schemas import AccountSnapshot, Direction, Horizon, RiskDecision, TradeProposal  # noqa: E402
from app.storage.db import Storage  # noqa: E402
from app.timeutil import UTC, SessionStatus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARKET_NOW = datetime(2026, 3, 9, 16, 0, tzinfo=UTC)      # Monday 12:00 ET (EDT), regular session


def make_settings(**env):
    base = {"ALPACA_PAPER": "true", "ALPACA_API_KEY_ID": "PKTESTKEY1234567890", "ALPACA_API_SECRET_KEY": "testsecretvalue123456"}
    base.update({k: str(v) for k, v in env.items()})
    return load_settings(str(ROOT / "config" / "default.yaml"), env=base, load_dotenv_file=False)


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def paper_settings():
    return make_settings(ENABLE_PAPER_TRADING="true")


@pytest.fixture
def db():
    d = Storage(":memory:")
    yield d
    d.close()


@pytest.fixture
def now():
    return MARKET_NOW


def open_session(now=MARKET_NOW) -> SessionStatus:
    return SessionStatus(True, True, now.date(), now - timedelta(minutes=150), now + timedelta(minutes=240), 150, 240, False)


def make_proposal(now=MARKET_NOW, **kw) -> TradeProposal:
    d = dict(signal_id="sig-1", symbol="AAPL", direction=Direction.LONG, strategy="momentum", horizon=Horizon.SWING,
             qty=100, entry_price=100.0, stop_price=96.0, target_price=112.0, created_at=now,
             expires_at=now + timedelta(minutes=30), avg_dollar_volume=5e8, avg_volume=5e6)
    d.update(kw)
    return TradeProposal(**d)
