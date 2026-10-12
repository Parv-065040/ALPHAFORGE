"""Benchmark local models on YOUR machine. Measures throughput/latency/memory; reports labelled-sample accuracy
ONLY if you supply labels. Nothing here proves trading value.

Usage:
  python scripts/benchmark_models.py finbert --csv tests/fixtures/headlines_sample.csv
  python scripts/benchmark_models.py ollama --model qwen3:8b --runs 5
CSV columns: headline[,label]   (label in positive|negative|neutral)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gpu_mem_mb():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
    except Exception:
        return None


def bench_finbert(args):
    from app.sentiment.pipeline import FinBertScorer, RuleBasedScorer
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    texts = [r["headline"] for r in rows]
    for name, build in (("finbert", lambda: FinBertScorer(args.model)), ("rule_based", RuleBasedScorer)):
        try:
            t0 = time.perf_counter(); sc = build(); load = time.perf_counter() - t0
        except Exception as e:
            print(f"{name}: unavailable ({type(e).__name__}: {e})"); continue
        sc.score(texts[:2])                                    # warm-up
        t0 = time.perf_counter(); out = sc.score(texts * args.repeat); dt = time.perf_counter() - t0
        res = {"scorer": name, "load_s": round(load, 2), "headlines": len(texts) * args.repeat,
               "headlines_per_s": round(len(texts) * args.repeat / dt, 1), "gpu_peak_mb": gpu_mem_mb()}
        if all(r.get("label") for r in rows):
            lab = lambda s: "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
            pred = [lab(s) for s, _ in out[:len(texts)]]
            res["accuracy_on_your_labels"] = round(sum(p == r["label"] for p, r in zip(pred, rows)) / len(rows), 3)
            res["n_labelled"] = len(rows)
        print(json.dumps(res))


def bench_ollama(args):
    from app.agents.llm import OllamaClient, run_structured
    from app.schemas import PMDecision
    c = OllamaClient(args.host, args.model, timeout=args.timeout)
    try:
        print("installed models:", c.available_models())
    except Exception as e:
        print(f"Ollama unreachable: {e}"); return
    lat, ok = [], 0
    for _ in range(args.runs):
        obj, res = run_structured(c, "Respond with JSON only.", "Symbol AAPL momentum breakout score 0.7. Propose or skip?", PMDecision, 0)
        lat.append(res.latency_ms); ok += obj is not None
    print(json.dumps({"model": args.model, "runs": args.runs, "valid_json_rate": ok / args.runs,
                      "latency_ms_median": round(statistics.median(lat)), "latency_ms_max": round(max(lat))}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="c", required=True)
    f = sub.add_parser("finbert"); f.add_argument("--csv", required=True); f.add_argument("--model", default="ProsusAI/finbert"); f.add_argument("--repeat", type=int, default=20); f.set_defaults(fn=bench_finbert)
    o = sub.add_parser("ollama"); o.add_argument("--model", default="qwen3:8b"); o.add_argument("--host", default="http://localhost:11434"); o.add_argument("--runs", type=int, default=5); o.add_argument("--timeout", type=float, default=120); o.set_defaults(fn=bench_ollama)
    a = p.parse_args(); a.fn(a)
