# Milestone 008: Conclusions

Estado: COMPLETO

## H3.1 — Native Paged FA reduces attention latency 12–18% vs gather for >8K context

**Result: INCONCLUSIVE on RTX 2050.**
The benchmark infrastructure is in place and validated for sub-4K contexts. The 8K context
test was blocked by OOM (KV cache allocation exceeds VRAM). H3.1 requires dedicated testing
on a GPU with ≥8 GB VRAM to measure the attention kernel latency difference at >8K tokens.

## H8.1 — TurboQuant achieves higher intelligence/GB than q4_0/q8_0

**Result: INFRASTRUCTURE VALIDATED, quantitative comparison deferred.**
Generation speed on RTX 2050 with q2_k (3031 MB, 29.7 t/s) confirms 8B models run at
usable speed in 4 GB. Perplexity comparison vs q4_0/q8_0 requires the M007 calibration
run (triattention_calibrate.py) which needs a suitable corpus file.

## TriAttention Page Budget — Key Finding

Page eviction (via `--triattention-page-budget`) operates at generation time (which pages
to retain in the active attention window). It does NOT reduce the initial physical KV
cache CUDA allocation. To achieve VRAM savings at load time, the physical page pool size
must be bounded at context init — this is the M007 / H6.1 implementation gap.

## IQ Quantizations (IQ2_S, IQ3_XS)

These quantization formats are not yet supported by this build (SIGSEGV at load time).
They require the importance-based quantization kernel implementation in ggml, which is
present in upstream llama.cpp but not yet merged into this fork.

## Benchmark Infrastructure (M008 Goal — ACHIEVED)

`scripts/benchmark.py` is operational and produces reproducible prompt/generation t/s
measurements across all supported model formats. Results are persisted to JSON for
longitudinal tracking. The ollama integration layer (`triattention_page_budget` API
option, auto-fit scheduler logic) is complete and tested.
