# Model selection (to be confirmed by benchmarking on your hardware)
Hardware target: Windows laptop, AMD Ryzen 7, RTX 4050 Laptop 6 GB VRAM.
- **Sentiment:** `ProsusAI/finbert` (~110M params) fits easily on GPU or CPU. Run
  `python scripts/benchmark_models.py finbert --csv <your labelled headlines.csv>` for throughput and accuracy on
  YOUR labels (the shipped 6-row sample is only a format example; accuracy on it means nothing).
- **Agent LLM:** primary candidate `qwen3:8b` (4-bit quantised ≈ 5 GB) - tight on 6 GB VRAM with context, expect partial
  CPU offload and slower responses; fallback `qwen3:4b`. Benchmark with
  `python scripts/benchmark_models.py ollama --model qwen3:8b --runs 5` (reports valid-JSON rate and latency).
  Choose the 4B model if the 8B is too slow or unreliable at structured output. Agents run a few times per cycle, so
  latency is acceptable; correctness is protected by schema validation and deterministic fallbacks.
These are candidates, not measured recommendations: no model was run during this build.
