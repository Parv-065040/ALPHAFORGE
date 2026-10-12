"""Preflight checks. Reports presence of secrets only (never values). Fails closed on paper-mode doubt."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import PAPER_BASE_URL, Settings


@dataclass
class Check:
    name: str
    status: str      # PASS | WARN | FAIL | SKIP
    detail: str


def _import_ok(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def run_preflight(s: Settings, online: bool = True) -> list[Check]:
    out: list[Check] = []
    add = lambda n, st, d: out.append(Check(n, st, d))
    add("python", "PASS" if sys.version_info >= (3, 10) else "FAIL", sys.version.split()[0])
    for m in ["pydantic", "yaml", "pandas", "numpy", "requests", "exchange_calendars", "plotly"]:
        add(f"dep:{m}", "PASS" if _import_ok(m) else "FAIL", "importable" if _import_ok(m) else "missing: pip install -r requirements.txt")
    add("dep:streamlit", "PASS" if _import_ok("streamlit") else "WARN", "dashboard needs streamlit")
    add("dep:transformers+torch", "PASS" if _import_ok("transformers") and _import_ok("torch") else "WARN",
        "FinBERT available" if _import_ok("transformers") and _import_ok("torch") else "optional: rule-based sentiment fallback will be used")
    add("dep:crewai", "PASS" if _import_ok("crewai") else "SKIP", "optional CrewAI Flow wrapper")
    add("env:ALPACA_API_KEY_ID", "PASS" if s.secrets.alpaca_key_id else "FAIL", "set" if s.secrets.alpaca_key_id else "missing")
    add("env:ALPACA_API_SECRET_KEY", "PASS" if s.secrets.alpaca_secret else "FAIL", "set" if s.secrets.alpaca_secret else "missing")
    add("paper_mode_flag", "PASS" if s.alpaca_paper else "FAIL",
        "ALPACA_PAPER=true" if s.alpaca_paper else "ALPACA_PAPER must be true (no live path exists)")
    add("paper_order_execution", "WARN" if s.enable_paper_trading else "PASS",
        "ENABLED: orders WILL be submitted to the Alpaca PAPER account in a paper session" if s.enable_paper_trading
        else "disabled (default); no orders can be submitted")
    add("shorting", "PASS", "enabled by config" if s.allow_shorting else "disabled (default)")
    try:
        from app.storage.db import Storage
        db = Storage(s.app.db_path)
        add("storage", "PASS", f"{s.app.db_path} schema v{db.schema_version()}")
        db.close()
    except Exception as e:
        add("storage", "FAIL", f"{type(e).__name__}: {e}")
    try:
        from app.timeutil import session_status, utcnow
        st = session_status(utcnow())
        add("market_calendar", "PASS", f"XNYS ok; session_day={st.is_session_day} open={st.is_open}")
    except Exception as e:
        add("market_calendar", "FAIL", f"{type(e).__name__}: {e}")
    add("data_source", "PASS" if s.has_alpaca_keys() else "WARN",
        f"Alpaca market data feed={s.data.feed}; news via Alpaca (Benzinga) with the same keys")
    if online and s.has_alpaca_keys() and s.alpaca_paper:
        try:
            from app.execution.broker import AlpacaPaperBroker
            b = AlpacaPaperBroker(s)
            a = b.get_account()
            ok = a.status.upper() in {"ACTIVE", "ACCOUNT_UPDATED"}
            add("alpaca_paper_account", "PASS" if ok else "FAIL",
                f"endpoint {PAPER_BASE_URL}; status={a.status}; equity=${a.equity:,.2f} (target ${s.app.starting_equity_target:,.0f})")
        except Exception as e:
            add("alpaca_paper_account", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")
    else:
        add("alpaca_paper_account", "SKIP", "offline or keys/paper flag missing")
    if online and s.use_llm:
        try:
            from app.agents.llm import OllamaClient
            models = OllamaClient(s.ollama_host, s.ollama_model).available_models()
            has = any(m.startswith(s.ollama_model) for m in models)
            fb = any(m.startswith(s.ollama_fallback_model) for m in models)
            add("ollama_model", "PASS" if has else ("WARN" if fb else "FAIL"),
                f"{s.ollama_model}: {'found' if has else 'missing (ollama pull ' + s.ollama_model + ')'}; fallback {s.ollama_fallback_model}: {'found' if fb else 'missing'}")
        except Exception as e:
            add("ollama_model", "WARN", f"Ollama unreachable at {s.ollama_host} ({type(e).__name__}); agents use deterministic fallbacks")
    else:
        add("ollama_model", "SKIP", "USE_LLM=false: deterministic fallbacks only")
    return out


def format_report(checks: list[Check]) -> str:
    icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}
    lines = [f"{icon[c.status]} {c.name}: {c.detail}" for c in checks]
    fails = sum(c.status == "FAIL" for c in checks)
    lines.append(f"\nResult: {'FAILED' if fails else 'OK'} ({fails} failing)")
    return "\n".join(lines)


def paper_mode_confirmed(checks: list[Check]) -> bool:
    d = {c.name: c.status for c in checks}
    return d.get("paper_mode_flag") == "PASS" and d.get("alpaca_paper_account") == "PASS"
