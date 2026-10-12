"""Sentiment Intelligence pipeline: normalize -> dedupe -> ticker map -> score -> novelty/materiality/decay
-> per-symbol aggregation -> price/volume confirmation. Sentiment is NOT a calibrated return probability."""
from __future__ import annotations

import hashlib
import html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from app.config import SentimentCfg
from app.schemas import NewsItem, SentimentResult, SymbolSentiment
from app.timeutil import utcnow

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9$%.]+")

MATERIALITY_KEYWORDS: dict[str, float] = {
    "earnings": 0.8, "guidance": 0.85, "forecast": 0.6, "fda": 0.9, "approval": 0.8, "merger": 0.9,
    "acquisition": 0.9, "acquire": 0.85, "buyout": 0.9, "takeover": 0.9, "bankruptcy": 1.0, "downgrade": 0.7,
    "upgrade": 0.7, "price target": 0.55, "lawsuit": 0.6, "sec ": 0.6, "investigation": 0.7, "recall": 0.7,
    "buyback": 0.65, "dividend": 0.5, "contract": 0.6, "layoffs": 0.6, "revenue": 0.65, "profit": 0.6,
    "outlook": 0.65, "partnership": 0.45, "launch": 0.4, "beats": 0.75, "misses": 0.75, "plunge": 0.7, "surge": 0.65,
}
POS_WORDS = {"beats", "surge", "soars", "jumps", "upgrade", "upgraded", "record", "growth", "approval", "approved",
             "raises", "raised", "strong", "outperform", "buyback", "wins", "profit", "gains", "rallies", "bullish"}
NEG_WORDS = {"misses", "plunge", "plunges", "falls", "drops", "downgrade", "downgraded", "lawsuit", "probe",
             "investigation", "recall", "cuts", "cut", "weak", "layoffs", "bankruptcy", "fraud", "warns", "slump",
             "bearish", "loses", "decline", "declines"}


def clean_text(s: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def headline_hash(headline: str) -> str:
    toks = " ".join(_TOK.findall(clean_text(headline).lower()))
    return hashlib.sha1(toks.encode()).hexdigest()[:16]


def _tokens(s: str) -> set[str]:
    return set(_TOK.findall(clean_text(s).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class DedupResult:
    unique: list[NewsItem]
    duplicate_counts: dict[str, int]       # kept news id -> number of duplicates folded into it
    hashes: dict[str, str]


def normalize_and_dedupe(items: list[NewsItem], universe: set[str], similarity: float = 0.8,
                         known_hashes: set[str] | None = None) -> DedupResult:
    """Normalize text, map tickers to the universe, fold exact/near duplicates into the earliest story."""
    known = known_hashes or set()
    cleaned: list[NewsItem] = []
    for it in items:
        syms = sorted({s for s in it.symbols if s in universe})
        h = clean_text(it.headline)
        if not h or not syms:
            continue
        cleaned.append(it.model_copy(update={"headline": h, "summary": clean_text(it.summary), "symbols": syms}))
    cleaned.sort(key=lambda i: i.published_at)
    kept: list[NewsItem] = []
    toks: dict[str, set[str]] = {}
    dups: dict[str, int] = {}
    hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for it in cleaned:
        if it.id in seen_ids:
            continue
        seen_ids.add(it.id)
        hh = headline_hash(it.headline)
        t = _tokens(it.headline)
        match = next((k for k in kept if hashes[k.id] == hh or
                      (set(k.symbols) & set(it.symbols) and jaccard(toks[k.id], t) >= similarity)), None)
        if match is not None:
            dups[match.id] = dups.get(match.id, 0) + 1
            continue
        kept.append(it)
        hashes[it.id] = hh
        toks[it.id] = t
        dups.setdefault(it.id, 0)
        if hh in known:                         # seen in a previous batch: counts as a prior duplicate
            dups[it.id] += 1
    return DedupResult(kept, dups, hashes)


# ---------- scorers ----------
class Scorer(Protocol):
    name: str
    def score(self, texts: list[str]) -> list[tuple[float, float]]: ...   # (score in [-1,1], confidence)


class RuleBasedScorer:
    """Transparent lexicon fallback. Much weaker than FinBERT; labelled model='rule_based'."""
    name = "rule_based"

    def score(self, texts):
        out = []
        for t in texts:
            tk = _tokens(t)
            p, n = len(tk & POS_WORDS), len(tk & NEG_WORDS)
            if p + n == 0:
                out.append((0.0, 0.3))
            else:
                out.append(((p - n) / (p + n), min(0.4 + 0.15 * (p + n), 0.8)))
        return out


class FinBertScorer:
    """ProsusAI/finbert via transformers (lazy import; needs `pip install transformers torch`)."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        import torch  # type: ignore
        self._torch = torch
        self.name = "finbert"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.labels = {int(i): l.lower() for i, l in self.model.config.id2label.items()}

    def score(self, texts):
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i:i + 16]
                enc = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                for row in probs:
                    d = {self.labels[j]: float(row[j]) for j in range(len(row))}
                    out.append((d.get("positive", 0.0) - d.get("negative", 0.0), float(max(row))))
        return out


def get_scorer(use_finbert: bool, model_name: str) -> tuple[Scorer, str | None]:
    """Return (scorer, warning). Falls back to rules if FinBERT is unavailable."""
    if use_finbert:
        try:
            return FinBertScorer(model_name), None
        except Exception as e:  # missing deps, no cached weights offline, etc.
            return RuleBasedScorer(), f"FinBERT unavailable ({type(e).__name__}); using rule-based fallback"
    return RuleBasedScorer(), "FinBERT disabled; using rule-based scorer"


# ---------- materiality / novelty / decay ----------
def materiality(item: NewsItem) -> float:
    text = f" {item.headline.lower()} {item.summary.lower()[:200]} "
    base = max([w for k, w in MATERIALITY_KEYWORDS.items() if k in text] or [0.15])
    focus = 1.0 if len(item.symbols) <= 2 else 0.6      # broad multi-ticker roundups are less specific
    return round(min(base * focus, 1.0), 3)


def novelty(dup_count: int) -> float:
    return round(1.0 / (1.0 + dup_count), 3)


def time_decay(published_at: datetime, now: datetime, half_life_h: float) -> float:
    age_h = max((now - published_at).total_seconds() / 3600, 0.0)
    return float(0.5 ** (age_h / half_life_h))


def analyze(items: list[NewsItem], universe: set[str], scorer: Scorer, cfg: SentimentCfg, now: datetime | None = None,
            known_hashes: set[str] | None = None) -> tuple[list[SentimentResult], DedupResult]:
    now = now or utcnow()
    dd = normalize_and_dedupe(items, universe, cfg.dedupe_similarity, known_hashes)
    if not dd.unique:
        return [], dd
    texts = [f"{i.headline}. {i.summary[:200]}".strip() for i in dd.unique]
    scored = scorer.score(texts)
    results: list[SentimentResult] = []
    for it, (sc, conf) in zip(dd.unique, scored):
        sc = float(np.clip(sc, -1, 1))
        label = "positive" if sc > 0.15 else "negative" if sc < -0.15 else "neutral"
        nov, mat = novelty(dd.duplicate_counts.get(it.id, 0)), materiality(it)
        decay = time_decay(it.published_at, now, cfg.half_life_hours)
        impact = sc * conf * decay * nov * (0.3 + 0.7 * mat)
        for sym in it.symbols:
            results.append(SentimentResult(
                news_id=it.id, symbol=sym, label=label, score=round(sc, 4), confidence=round(float(conf), 4),
                novelty=nov, materiality=mat, decayed_score=round(float(np.clip(impact, -1, 1)), 4),
                model=scorer.name, published_at=it.published_at, computed_at=now, headline=it.headline,
                url=it.url, source=it.source, evidence=it.summary[:300]))
    return results, dd


def aggregate(results: list[SentimentResult], cfg: SentimentCfg, now: datetime | None = None) -> dict[str, SymbolSentiment]:
    """Per-symbol view. Aggregated score = tanh(sum of decayed impacts); conflicting if strong +/- coexist."""
    now = now or utcnow()
    by: dict[str, list[SentimentResult]] = {}
    for r in results:
        by.setdefault(r.symbol, []).append(r)
    out: dict[str, SymbolSentiment] = {}
    for sym, rs in by.items():
        # recompute decay at `now` so stored results age correctly
        impacts = []
        for r in rs:
            base = r.score * r.confidence * r.novelty * (0.3 + 0.7 * r.materiality)
            impacts.append(base * time_decay(r.published_at, now, cfg.half_life_hours))
        agg = float(math.tanh(sum(impacts) * 2.0))
        newest = min((now - r.published_at).total_seconds() / 3600 for r in rs)
        top = max(rs, key=lambda r: abs(r.decayed_score) * r.materiality)
        pos = any(i > 0.08 for i in impacts)
        neg = any(i < -0.08 for i in impacts)
        out[sym] = SymbolSentiment(
            symbol=sym, score=round(agg, 4), n_items=len(rs), max_materiality=max(r.materiality for r in rs),
            min_novelty_of_top=top.novelty, newest_age_hours=round(newest, 2), conflicting=pos and neg,
            stale=newest > cfg.stale_after_hours, top_news_id=top.news_id, model=top.model)
    return out


def price_volume_confirmation(sent: SymbolSentiment | None, ret_since_close: float, vol_ratio: float) -> str:
    """confirmed | divergent | unconfirmed | no_sentiment."""
    if sent is None or sent.n_items == 0:
        return "no_sentiment"
    if abs(sent.score) < 0.1:
        return "unconfirmed"
    aligned = (sent.score > 0) == (ret_since_close > 0)
    if aligned and abs(ret_since_close) >= 0.005 and vol_ratio >= 1.2:
        return "confirmed"
    if not aligned and abs(ret_since_close) >= 0.005:
        return "divergent"
    return "unconfirmed"
