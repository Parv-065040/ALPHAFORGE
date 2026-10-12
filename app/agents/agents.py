"""Bounded, typed agents. Numeric work stays in Python; LLMs only classify/explain and may only
SKIP or SCALE DOWN a trade. Any failure -> deterministic fallback or safe no-trade."""
from __future__ import annotations

from app.agents.llm import LLMClient, run_structured
from app.schemas import (AgentResult, CatalystAssessment, Direction, PMDecision, RegimeAssessment, Signal,
                         SentimentResult, SymbolSentiment)
from app.storage.db import Storage
from app.strategies.regime import RegimeState

SYSTEM_RULES = ("You are a cautious financial analyst assistant. Use ONLY the facts provided. If evidence is thin "
                "or conflicting answer 'unclear'/'skip'. Respond with JSON matching the schema, nothing else.")


class _Base:
    name = "agent"

    def __init__(self, client: LLMClient | None, db: Storage | None, max_retries: int = 1):
        self.client, self.db, self.max_retries = client, db, max_retries

    def _log(self, ok: bool, latency: float, err: str | None, fb: bool, out: dict | None) -> None:
        if self.db:
            self.db.log_agent_run(self.name, ok, latency, err, fb, out)


class CatalystAnalyst(_Base):
    """Research & Catalyst Analyst: assesses selected high-impact/ambiguous stories for ONE symbol."""
    name = "catalyst_analyst"

    def assess(self, symbol: str, items: list[SentimentResult], agg: SymbolSentiment | None
               ) -> tuple[CatalystAssessment, AgentResult]:
        fallback = self._fallback(symbol, items, agg)
        if self.client is None or not items:
            self._log(True, 0.0, None, True, fallback.model_dump())
            return fallback, AgentResult(agent=self.name, ok=True, output=fallback.model_dump(), used_fallback=True)
        facts = "\n".join(f"- [{i.published_at.isoformat()}] ({i.source}) {i.headline} | model={i.model} "
                          f"score={i.score:+.2f} materiality={i.materiality:.2f} url={i.url}" for i in items[:6])
        obj, res = run_structured(self.client, SYSTEM_RULES,
                                  f"Symbol: {symbol}\nStories:\n{facts}\nAssess materiality and direction.",
                                  CatalystAssessment, self.max_retries)
        if obj is None or obj.symbol.upper() != symbol:
            err = res.error or "symbol mismatch in output"
            self._log(False, res.latency_ms, err, True, fallback.model_dump())
            return fallback, AgentResult(agent=self.name, ok=False, error=err, latency_ms=res.latency_ms,
                                         output=fallback.model_dump(), used_fallback=True)
        known = {i.url for i in items}
        obj = obj.model_copy(update={"evidence_urls": [u for u in obj.evidence_urls if u in known]})  # grounding
        self._log(True, res.latency_ms, None, False, obj.model_dump())
        return obj, AgentResult(agent=self.name, ok=True, output=obj.model_dump(), latency_ms=res.latency_ms)

    @staticmethod
    def _fallback(symbol: str, items: list[SentimentResult], agg: SymbolSentiment | None) -> CatalystAssessment:
        if not items or agg is None:
            return CatalystAssessment(symbol=symbol, materiality="low", direction="unclear",
                                      summary="No news evidence available.", uncertainty=1.0)
        mat = "high" if agg.max_materiality >= 0.7 else "medium" if agg.max_materiality >= 0.45 else "low"
        d = "unclear" if agg.conflicting else "bullish" if agg.score > 0.15 else "bearish" if agg.score < -0.15 else "neutral"
        return CatalystAssessment(symbol=symbol, materiality=mat, direction=d, uncertainty=0.5 if not agg.conflicting else 0.9,
                                  summary=f"Rule-derived from {agg.n_items} stories (model={agg.model}); no LLM used.",
                                  evidence_urls=[i.url for i in items[:3] if i.url])


class RegimeAnalyst(_Base):
    """Market Regime Analyst: deterministic regime is authoritative; LLM adds commentary only."""
    name = "regime_analyst"

    def assess(self, regime: RegimeState) -> tuple[RegimeAssessment, AgentResult]:
        base = RegimeAssessment(regime=regime.regime, confidence=min(abs(regime.score - 0.5) * 2 + 0.3, 1.0),
                                rationale="; ".join(regime.reasons)[:480])
        if self.client is None:
            self._log(True, 0.0, None, True, base.model_dump())
            return base, AgentResult(agent=self.name, ok=True, output=base.model_dump(), used_fallback=True)
        obj, res = run_structured(self.client, SYSTEM_RULES,
                                  f"Benchmark regime facts: {regime.reasons}; score={regime.score}. Summarise.",
                                  RegimeAssessment, self.max_retries)
        if obj is None:
            self._log(False, res.latency_ms, res.error, True, base.model_dump())
            return base, AgentResult(agent=self.name, ok=False, error=res.error, output=base.model_dump(),
                                     latency_ms=res.latency_ms, used_fallback=True)
        # The deterministic regime and rationale are authoritative. Do not accept
        # LLM commentary that explicitly contradicts the computed classification.
        rationale = (obj.rationale or "").strip()
        contradiction_terms = {
            "risk_on": ("risk off", "risk_off", "bear market", "bearish regime"),
            "risk_off": ("risk on", "risk_on", "bull market", "bullish regime"),
            "neutral": ("risk on", "risk_on", "risk off", "risk_off", "bull market", "bear market"),
            "unclear": (),
        }
        normalized = rationale.lower().replace("-", " ").replace("_", " ")
        conflicts = any(
            term.replace("_", " ") in normalized
            for term in contradiction_terms.get(base.regime, ())
        )
        if not rationale or conflicts:
            rationale = base.rationale
        merged = base.model_copy(update={"rationale": rationale[:500]})
        self._log(True, res.latency_ms, None, False, merged.model_dump())
        return merged, AgentResult(agent=self.name, ok=True, output=merged.model_dump(), latency_ms=res.latency_ms)


class PortfolioManager(_Base):
    """Trade Proposal Agent. May only skip or scale size down (<=1.0); Python builds the proposal and the
    risk engine remains the authority. Without an LLM: deterministic quant-only pass-through."""
    name = "portfolio_manager"

    def decide(self, sig: Signal, catalyst: CatalystAssessment | None, regime: RegimeAssessment
               ) -> tuple[PMDecision, AgentResult]:
        quant = PMDecision(symbol=sig.symbol, action="propose", size_multiplier=1.0, confidence=sig.score,
                           rationale=f"quant-only: {sig.strategy} score {sig.score:.2f}; " + "; ".join(sig.reasons)[:300])
        if self.client is None:
            self._log(True, 0.0, None, True, quant.model_dump())
            return quant, AgentResult(agent=self.name, ok=True, output=quant.model_dump(), used_fallback=True)
        prompt = (f"Signal: {sig.symbol} {sig.direction.value} via {sig.strategy}, score {sig.score:.2f}, "
                  f"reasons {sig.reasons}. Regime: {regime.regime} ({regime.rationale}). "
                  f"Catalyst view: {catalyst.model_dump() if catalyst else 'n/a'}. "
                  "Decide propose or skip, and a size_multiplier in [0,1] (1 = full size).")
        obj, res = run_structured(self.client, SYSTEM_RULES, prompt, PMDecision, self.max_retries)
        if obj is None or obj.symbol.upper() != sig.symbol:
            # malformed/ambiguous -> SAFE no-trade (never invent a proposal)
            skip = PMDecision(symbol=sig.symbol, action="skip", size_multiplier=0.0, confidence=0.0,
                              rationale=f"agent output invalid: {res.error or 'symbol mismatch'}"[:480])
            self._log(False, res.latency_ms, res.error, False, skip.model_dump())
            return skip, AgentResult(agent=self.name, ok=False, error=res.error or "symbol mismatch",
                                     output=skip.model_dump(), latency_ms=res.latency_ms)
        self._log(True, res.latency_ms, None, False, obj.model_dump())
        return obj, AgentResult(agent=self.name, ok=True, output=obj.model_dump(), latency_ms=res.latency_ms)
