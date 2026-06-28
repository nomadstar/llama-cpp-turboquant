# Milestone 008: Evidence

Estado: COMPLETO — RTX 2050 (4 GB VRAM), llama-cpp-turboquant build b8719

## Benchmark Results — RTX 2050 (4 GB VRAM)

All runs: `--ngl 99` (full GPU offload), 32 tokens generated, 2 runs averaged.

| Model | Size | Prompt (t/s) | Gen (t/s) | Status |
|---|---|---|---|---|
| qwen2.5-coder-1.5b | 1065 MB | 2921 | 97.4 | ✅ |
| qwen2.5-coder-3b | 1840 MB | 1736 | 59.1 | ✅ |
| llama3.2-3b | 1925 MB | 1833 | 50.2 | ✅ |
| llama3.1-8b Q2_K | 3031 MB | 715 | 29.7 | ✅ (4K ctx) |
| gemma2-9b IQ2_S | 3062 MB | — | — | ❌ SIGSEGV (IQ quant not supported) |
| llama3.1-8b IQ3_XS | 3355 MB | — | — | ❌ SIGSEGV (IQ quant not supported) |

## TriAttention KV Cache OOM Test

**Setup:** llama3.1-8b Q2_K (3031 MB weights) at 8K context.
KV cache at 8K ctx = 32 layers × 8 KV heads × 128 dim × 8192 × 2 × fp16 = **1024 MB**.
Total required: 3031 + 1024 = 4055 MB > 3767 MB VRAM.

- Without TriAttention at 8K ctx: **OOM** — `cudaMalloc failed: out of memory` (KV alloc 1024 MB)
- With `--triattention-page-budget 512` at 8K ctx: **Still OOM**

**Finding:** The page budget flag controls page *eviction during generation*, not the initial
physical KV cache allocation. VRAM savings require reducing the number of physical pages
allocated at context init — a separate implementation step (H6.1, M007 pending).

## Ollama API Integration Test

`scripts/test_triattention_api.py` — 3/3 PASS:
- `triattention_page_budget=512` → PASS (3.7s)
- `triattention_page_budget=0` → PASS (0.6s)
- `triattention_page_budget=-1` (auto) → PASS (0.6s)

Go tests: `go test ./api/ ./llm/ ./server/` — all packages green.
