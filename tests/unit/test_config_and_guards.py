import logging

import pytest
from pydantic import ValidationError

from app.config import PAPER_BASE_URL, RiskCfg, ScoringCfg, load_settings
from app.data.http import HttpClient
from app.execution.broker import AlpacaPaperBroker, LivePathBlocked
from app.execution.service import ExecutionService
from app.logging_setup import REDACTED, RedactingFilter, redact
from app.storage.db import Storage
from tests.conftest import make_proposal, make_settings
from tests.fakes import FakeBroker


def test_defaults_are_safe():
    s = load_settings("config/default.yaml", env={}, load_dotenv_file=False)
    assert s.enable_paper_trading is False and s.alpaca_paper is False and s.allow_shorting is False
    assert s.risk.risk_per_trade_pct == 0.005
    assert s.strategies.relative_strength.enabled is False


def test_higher_risk_requires_explicit_approval():
    with pytest.raises(ValidationError):
        RiskCfg(risk_per_trade_pct=0.0075)
    assert RiskCfg(risk_per_trade_pct=0.0075, higher_risk_approved=True).risk_per_trade_pct == 0.0075


def test_scoring_weight_keys_validated():
    with pytest.raises(ValidationError):
        ScoringCfg(weights={"trend_momentum": 1.0})


def test_safe_dict_never_contains_secret_values():
    s = make_settings()
    blob = str(s.safe_dict()) + repr(s.secrets) + str(s.secrets)
    assert "PKTESTKEY1234567890" not in blob and "testsecretvalue123456" not in blob
    assert s.safe_dict()["secrets"]["ALPACA_API_KEY_ID"] == "set"


def test_broker_refuses_when_paper_flag_false():
    s = make_settings(ALPACA_PAPER="false")
    with pytest.raises(LivePathBlocked):
        AlpacaPaperBroker(s)


def test_broker_refuses_non_paper_base_url():
    s = make_settings()
    with pytest.raises(LivePathBlocked):
        AlpacaPaperBroker(s, http=HttpClient("https://api.alpaca.markets", {}))


def test_broker_accepts_paper_url_only():
    s = make_settings()
    b = AlpacaPaperBroker(s, http=HttpClient(PAPER_BASE_URL, {}))
    assert b.is_paper


def test_broker_refuses_without_keys():
    s = make_settings(ALPACA_API_KEY_ID="", ALPACA_API_SECRET_KEY="")
    with pytest.raises(LivePathBlocked):
        AlpacaPaperBroker(s)


def test_execution_disabled_by_default_submits_nothing(db, now):
    s = make_settings()                                   # ENABLE_PAPER_TRADING unset -> false
    fb = FakeBroker()
    ex = ExecutionService(s, fb, db, now_fn=lambda: now)
    p = make_proposal(now)
    from app.schemas import RiskDecision
    d = RiskDecision(proposal_id=p.id, approved=True, approved_qty=10, decided_at=now)
    db.save_proposal(p); db.save_decision(d)
    r = ex.execute(p, d)
    assert r.status == "disabled" and fb.submit_calls == 0


def test_redact_removes_secret_values_and_key_patterns():
    out = redact("key=PKABCDEFGHIJKLMNOPQR secret: abc123supersecret token=zzzz1234 value testsecretvalue123456",
                 ["testsecretvalue123456"])
    assert "PKABCDEFGHIJKLMNOPQR" not in out and "abc123supersecret" not in out
    assert "testsecretvalue123456" not in out and REDACTED in out


def test_logging_filter_redacts(caplog):
    lg = logging.getLogger("alphaforge.test")
    lg.addFilter(RedactingFilter(["mysecretvalue99"]))
    with caplog.at_level(logging.INFO, logger="alphaforge.test"):
        lg.info("calling api with mysecretvalue99")
    assert "mysecretvalue99" not in caplog.text
