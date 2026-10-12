"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

REDACTED = "***REDACTED***"
# Alpaca-style key ids (PK/AK + 18+ alphanumerics) and long secret-like tokens after key= / secret=
_PATTERNS = [
    re.compile(r"\b(?:PK|AK|CK)[A-Z0-9]{16,}\b"),
    re.compile(r"(?i)(secret|apikey|api_key|api-key|authorization|token)(\s*[=:]\s*)([^\s,;\"']+)"),
]


def redact(text: str, secrets: list[str] | None = None) -> str:
    out = str(text)
    for s in secrets or []:
        if s and len(s) >= 4:
            out = out.replace(s, REDACTED)
    out = _PATTERNS[0].sub(REDACTED, out)
    out = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    return out


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self._secrets = secrets or []

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), self._secrets)
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload["ctx"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: str | None = None,
                  secrets: list[str] | None = None) -> logging.Logger:
    root = logging.getLogger("alphaforge")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    flt = RedactingFilter(secrets)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "alphaforge.log", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        h.addFilter(flt)
        root.addHandler(h)
    root.propagate = False
    return root
