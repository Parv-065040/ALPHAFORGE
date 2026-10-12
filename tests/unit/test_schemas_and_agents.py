import json
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.agents.agents import CatalystAnalyst, PortfolioManager, RegimeAnalyst
from app.agents.llm import run_structured
from app.schemas import (CatalystAssessment, Direction, Horizon, PMDecision, RegimeAssessment, SentimentResult, Signal,
                         SymbolSentiment, TradeProposal, normalize_symbol, NewsItem)
from app.strategies.regime import RegimeState
from app.timeutil import UTC
from tests.conftest import MARKET_NOW, make_proposal


class FakeLLM:
    model = "fake"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def chat(self, system, user, schema=None):
        self.calls += 1
        r = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def sig(sym="AAPL"):
    return Signal(symbol=sym, strategy="momentum", direction=Direction.LONG, horizon=Horizon.SWING, entry_price=100,
                  stop_price=96, target_price=112, score=0.7, components={"trend_momentum": 0.7}, created_at=MARKET_NOW,
                  expires_at=MARKET_NOW + timedelta(minutes=30), reasons=["breakout"])


def test_symbol_normalization():
    assert normalize_symbol(" brk-b ") == "BRK.B" and normalize_symbol("aapl") == "AAPL"
    with pytest.raises(ValueError):
        normalize_symbol("TOOLONGSYM1")


def test_naive_timestamps_rejected():
    with pytest.raises(ValidationError):
        NewsItem(id="1", headline="x", published_at=datetime(2026, 1, 1), ingested_at=datetime(2026, 1, 1, tzinfo=UTC))


def test_extra_fields_forbidden_and_qty_positive():
    with pytest.raises(ValidationError):
        PMDecision(symbol="A", action="propose", size_multiplier=1.0, confidence=0.5, rationale="x", bogus=1)
    with pytest.raises(ValidationError):
        make_proposal(qty=0)


def test_pm_cannot_scale_up():
    with pytest.raises(ValidationError):
        PMDecision(symbol="A", action="propose", size_multiplier=1.5, confidence=0.5, rationale="x")


def test_run_structured_malformed_json_is_safe_failure():
    obj, res = run_structured(FakeLLM(["not json"]), "s", "u", PMDecision, max_retries=1)
    assert obj is None and not res.ok and "validation" in res.error


def test_run_structured_exception_is_safe_failure():
    obj, res = run_structured(FakeLLM([TimeoutError("slow")]), "s", "u", PMDecision, max_retries=0)
    assert obj is None and "TimeoutError" in res.error


def test_run_structured_bounded_retries():
    llm = FakeLLM(["bad"])
    run_structured(llm, "s", "u", PMDecision, max_retries=2)
    assert llm.calls == 3


def test_pm_malformed_output_becomes_skip(db):
    pm = PortfolioManager(FakeLLM(["{{garbage"]), db, 0)
    d, r = pm.decide(sig(), None, RegimeAssessment(regime="risk_on", confidence=0.8, rationale="x"))
    assert d.action == "skip" and d.size_multiplier == 0 and not r.ok


def test_pm_symbol_mismatch_becomes_skip(db):
    bad = {"symbol": "MSFT", "action": "propose", "size_multiplier": 1.0, "confidence": 0.9, "rationale": "x"}
    d, r = PortfolioManager(FakeLLM([bad]), db, 0).decide(sig(), None, RegimeAssessment(regime="neutral", confidence=0.5, rationale="x"))
    assert d.action == "skip"


def test_pm_valid_output_passes_and_is_logged(db):
    ok = {"symbol": "AAPL", "action": "propose", "size_multiplier": 0.5, "confidence": 0.7, "rationale": "fine"}
    d, r = PortfolioManager(FakeLLM([ok]), db, 0).decide(sig(), None, RegimeAssessment(regime="neutral", confidence=0.5, rationale="x"))
    assert d.action == "propose" and d.size_multiplier == 0.5 and r.ok
    assert db.q("SELECT COUNT(*) n FROM agent_runs")[0]["n"] == 1


def test_pm_without_llm_is_labelled_quant_only(db):
    d, r = PortfolioManager(None, db).decide(sig(), None, RegimeAssessment(regime="neutral", confidence=0.5, rationale="x"))
    assert d.action == "propose" and r.used_fallback and d.rationale.startswith("quant-only")


def _sr(url="https://x/1"):
    return SentimentResult(news_id="1", symbol="AAPL", label="positive", score=0.6, confidence=0.8, novelty=1, materiality=0.8,
                           decayed_score=0.4, model="rule_based", published_at=MARKET_NOW, computed_at=MARKET_NOW,
                           headline="Apple beats", url=url, source="t")


def test_catalyst_evidence_is_grounded_to_known_urls(db):
    out = {"symbol": "AAPL", "materiality": "high", "direction": "bullish", "summary": "ok",
           "evidence_urls": ["https://x/1", "https://invented.example/fake"], "uncertainty": 0.2}
    agg = SymbolSentiment(symbol="AAPL", score=0.5, n_items=1, max_materiality=0.8, stale=False)
    a, r = CatalystAnalyst(FakeLLM([out]), db, 0).assess("AAPL", [_sr()], agg)
    assert a.evidence_urls == ["https://x/1"] and r.ok


def test_catalyst_failure_falls_back_without_inventing(db):
    agg = SymbolSentiment(symbol="AAPL", score=0.5, n_items=1, max_materiality=0.8, stale=False, model="rule_based")
    a, r = CatalystAnalyst(FakeLLM(["nope"]), db, 0).assess("AAPL", [_sr()], agg)
    assert r.used_fallback and not r.ok and a.direction == "bullish" and "no LLM used" in a.summary


def test_catalyst_no_evidence_is_unclear(db):
    a, r = CatalystAnalyst(None, db).assess("AAPL", [], None)
    assert a.direction == "unclear" and a.uncertainty == 1.0


def test_regime_llm_cannot_override_deterministic_label(db):
    llm_out = {"regime": "risk_on", "confidence": 0.99, "rationale": "LLM says bullish"}
    reg = RegimeState("risk_off", 0.2, ["benchmark below 50d SMA"])
    a, r = RegimeAnalyst(FakeLLM([llm_out]), db, 0).assess(reg)
    assert a.regime == "risk_off"

def test_regime_agent_rejects_contradictory_rationale():
    llm_out = {
        "regime": "risk_on",
        "confidence": 0.99,
        "rationale": "This is a risk off bear market."
    }
    reg = RegimeState("risk_on", 0.8, ["benchmark above 50d SMA"])
    assessment, result = RegimeAnalyst(FakeLLM([llm_out]), None, 0).assess(reg)

    assert result.ok is True
    assert assessment.regime == "risk_on"
    assert assessment.rationale == "benchmark above 50d SMA"


def test_regime_agent_keeps_consistent_rationale():
    llm_out = {
        "regime": "risk_on",
        "confidence": 0.99,
        "rationale": "The benchmark trend supports a risk-on environment."
    }
    reg = RegimeState("risk_on", 0.8, ["benchmark above 50d SMA"])
    assessment, result = RegimeAnalyst(FakeLLM([llm_out]), None, 0).assess(reg)

    assert result.ok is True
    assert assessment.regime == "risk_on"
    assert assessment.rationale == llm_out["rationale"]
