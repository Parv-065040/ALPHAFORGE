"""Typed, validated configuration. Secrets come only from the environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"
BASELINE_RISK_PCT = 0.005


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppCfg(_Strict):
    db_path: str = "data/alphaforge.db"
    log_dir: str = "logs"
    scan_interval_minutes: int = Field(15, ge=1, le=240)
    starting_equity_target: float = 100000


class DataCfg(_Strict):
    feed: str = "iex"
    daily_lookback_days: int = 420
    max_quote_age_seconds: int = 120
    max_bar_age_sessions: int = 3
    news_lookback_hours: int = 48
    http_timeout_seconds: float = 10
    max_retries: int = 3
    backoff_seconds: float = 1.0
    benchmark_symbol: str = "SPY"


class UniverseCfg(_Strict):
    symbols: list[str]
    correlation_groups: dict[str, list[str]] = {}


class SessionCfg(_Strict):
    entry_blackout_open_minutes: int = 5
    entry_blackout_close_minutes: int = 15
    intraday_flatten_minutes_before_close: int = 10


class ScoringCfg(_Strict):
    weights: dict[str, float]
    min_score: float = 0.6

    @model_validator(mode="after")
    def _weights(self) -> "ScoringCfg":
        expected = {"trend_momentum", "relative_strength", "volume_breakout", "catalyst", "regime"}
        if set(self.weights) != expected:
            raise ValueError(f"scoring.weights keys must be {sorted(expected)}")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("scoring weights must be non-negative with a positive sum")
        return self


class MomentumCfg(_Strict):
    enabled: bool = True
    breakout_lookback: int = 20
    sma_fast: int = 20
    sma_slow: int = 50
    sma_trend: int = 100
    min_volume_ratio: float = 1.3
    atr_period: int = 14
    stop_atr_multiple: float = 2.0
    target_r_multiple: float = 3.0
    max_chase_atr: float = 0.5
    min_atr_pct: float = 0.01
    max_atr_pct: float = 0.06
    horizon: str = "swing"


class CatalystCfg(_Strict):
    enabled: bool = True
    min_decayed_sentiment: float = 0.35
    min_materiality: float = 0.5
    min_novelty: float = 0.5
    min_price_move_pct: float = 0.01
    min_volume_ratio: float = 1.5
    stop_atr_multiple: float = 1.5
    target_r_multiple: float = 2.0
    horizon: str = "intraday"


class RelStrengthCfg(_Strict):
    enabled: bool = False
    rs_lookback: int = 63
    min_rs_long: float = 0.05
    breakdown_lookback: int = 20
    min_volume_ratio: float = 1.3
    stop_atr_multiple: float = 2.0
    target_r_multiple: float = 2.5
    horizon: str = "swing"
    allow_short_signals: bool = False


class StrategiesCfg(_Strict):
    momentum: MomentumCfg = MomentumCfg()
    catalyst: CatalystCfg = CatalystCfg()
    relative_strength: RelStrengthCfg = RelStrengthCfg()


class RegimeCfg(_Strict):
    block_longs_in_risk_off: bool = True
    max_benchmark_atr_pct: float = 0.025


class Envelopes(_Strict):
    intraday: float = 40000
    swing: float = 25000
    reserve: float = 25000
    uncommitted: float = 10000


class RiskCfg(_Strict):
    risk_per_trade_pct: float = BASELINE_RISK_PCT
    higher_risk_approved: bool = False
    max_position_pct: float = 0.15
    max_gross_exposure_pct: float = 0.70
    daily_loss_limit_pct: float = 0.02
    drawdown_review_pct: float = 0.05
    buying_power_buffer_pct: float = 0.05
    min_avg_dollar_volume: float = 20_000_000
    max_adv_participation: float = 0.01
    min_reward_risk: float = 1.5
    max_positions: int = 8
    max_new_entries_per_day: int = 4
    max_group_exposure_pct: float = 0.30
    proposal_ttl_minutes: int = 30
    approval_ttl_minutes: int = 5
    limit_buffer_bps: float = 10
    entry_order_max_age_minutes: int = 45
    existing_positions_policy: str = "hold_with_stops"
    envelopes: Envelopes = Envelopes()

    @model_validator(mode="after")
    def _guard_risk(self) -> "RiskCfg":
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 0.02:
            raise ValueError("risk_per_trade_pct must be in (0, 0.02]")
        if self.risk_per_trade_pct > BASELINE_RISK_PCT and not self.higher_risk_approved:
            raise ValueError(
                "risk_per_trade_pct above 0.5% requires risk.higher_risk_approved: true "
                "(explicit approval + documented validation)"
            )
        if self.existing_positions_policy not in {"hold_with_stops", "flatten_intraday"}:
            raise ValueError("existing_positions_policy must be hold_with_stops or flatten_intraday")
        return self


class SentimentCfg(_Strict):
    half_life_hours: float = 6.0
    dedupe_similarity: float = 0.8
    high_impact_materiality: float = 0.7
    stale_after_hours: float = 24


class LLMCfg(_Strict):
    timeout_seconds: float = 60
    max_retries: int = 1
    temperature: float = 0.0


class Secrets(BaseModel):
    """Held in memory only. repr/str never reveal values."""
    alpaca_key_id: str = ""
    alpaca_secret: str = ""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Secrets(<redacted>)"

    __str__ = __repr__

    def values(self) -> list[str]:
        return [v for v in (self.alpaca_key_id, self.alpaca_secret) if v]


class Settings(_Strict):
    app: AppCfg = AppCfg()
    data: DataCfg = DataCfg()
    universe: UniverseCfg
    session: SessionCfg = SessionCfg()
    scoring: ScoringCfg
    strategies: StrategiesCfg = StrategiesCfg()
    regime: RegimeCfg = RegimeCfg()
    risk: RiskCfg = RiskCfg()
    sentiment: SentimentCfg = SentimentCfg()
    llm: LLMCfg = LLMCfg()

    # Environment-derived (not part of the YAML file)
    alpaca_paper: bool = False
    enable_paper_trading: bool = False
    allow_shorting: bool = False
    use_finbert: bool = True
    finbert_model: str = "ProsusAI/finbert"
    use_llm: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_fallback_model: str = "qwen3:4b"
    log_level: str = "INFO"
    secrets: Secrets = Secrets()
    config_path: str = ""

    def has_alpaca_keys(self) -> bool:
        return bool(self.secrets.alpaca_key_id and self.secrets.alpaca_secret)

    def safe_dict(self) -> dict[str, Any]:
        """Configuration view with credentials replaced by presence flags."""
        d = self.model_dump(exclude={"secrets"})
        d["secrets"] = {
            "ALPACA_API_KEY_ID": "set" if self.secrets.alpaca_key_id else "missing",
            "ALPACA_API_SECRET_KEY": "set" if self.secrets.alpaca_secret else "missing",
        }
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.safe_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _truthy(v: str | None, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(config_path: str | None = None, env: dict[str, str] | None = None,
                  load_dotenv_file: bool = True) -> Settings:
    """Load YAML + environment. `env` overrides os.environ (used by tests)."""
    if load_dotenv_file and env is None:
        load_dotenv(override=False)
    e = dict(os.environ) if env is None else env
    path = Path(config_path or e.get("ALPHAFORGE_CONFIG") or "config/default.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("app", {})
    if e.get("ALPHAFORGE_DB_PATH"):
        raw["app"]["db_path"] = e["ALPHAFORGE_DB_PATH"]
    uni = raw.get("universe", {})
    if "symbols" in uni:
        uni["symbols"] = [str(s).upper().strip() for s in uni["symbols"]]
    raw["alpaca_paper"] = _truthy(e.get("ALPACA_PAPER"), False)
    raw["enable_paper_trading"] = _truthy(e.get("ENABLE_PAPER_TRADING"), False)
    raw["allow_shorting"] = _truthy(e.get("ALLOW_SHORTING"), False)
    raw["use_finbert"] = _truthy(e.get("USE_FINBERT"), True)
    raw["use_llm"] = _truthy(e.get("USE_LLM"), False)
    raw["finbert_model"] = e.get("FINBERT_MODEL", "ProsusAI/finbert")
    raw["ollama_host"] = e.get("OLLAMA_HOST", "http://localhost:11434")
    raw["ollama_model"] = e.get("OLLAMA_MODEL", "qwen3:8b")
    raw["ollama_fallback_model"] = e.get("OLLAMA_FALLBACK_MODEL", "qwen3:4b")
    raw["log_level"] = e.get("ALPHAFORGE_LOG_LEVEL", "INFO")
    raw["secrets"] = {
        "alpaca_key_id": e.get("ALPACA_API_KEY_ID", "").strip(),
        "alpaca_secret": e.get("ALPACA_API_SECRET_KEY", "").strip(),
    }
    raw["config_path"] = str(path)
    return Settings.model_validate(raw)
