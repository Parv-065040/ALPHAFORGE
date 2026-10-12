"""Local LLM access (Ollama) and strict structured-output validation. Malformed output -> safe failure."""
from __future__ import annotations

import time
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    model: str
    def chat(self, system: str, user: str, schema: dict | None = None) -> str: ...


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: float = 60, temperature: float = 0.0, session=None):
        self.host, self.model, self.timeout, self.temperature = host.rstrip("/"), model, timeout, temperature
        self.session = session or requests.Session()

    def available_models(self) -> list[str]:
        r = self.session.get(f"{self.host}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, system: str, user: str, schema: dict | None = None) -> str:
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": self.temperature}}
        if schema:
            body["format"] = schema
        r = self.session.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class StructuredResult(BaseModel):
    ok: bool
    value: dict | None = None
    error: str | None = None
    latency_ms: float = 0.0
    raw: str | None = None


def run_structured(client: LLMClient, system: str, user: str, model_cls: type[T], max_retries: int = 1
                   ) -> tuple[T | None, StructuredResult]:
    """Call the model, validate against a Pydantic schema, bounded retries. Never raises."""
    t0 = time.perf_counter()
    last_err, raw = "unknown", None
    for _ in range(max_retries + 1):
        try:
            raw = client.chat(system, user, model_cls.model_json_schema())
            obj = model_cls.model_validate_json(raw)
            return obj, StructuredResult(ok=True, value=obj.model_dump(), latency_ms=(time.perf_counter() - t0) * 1e3,
                                         raw=raw)
        except ValidationError as e:
            last_err = f"schema validation failed: {str(e)[:200]}"
        except Exception as e:  # network, timeout, bad JSON envelope
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
    return None, StructuredResult(ok=False, error=last_err, latency_ms=(time.perf_counter() - t0) * 1e3, raw=raw)
