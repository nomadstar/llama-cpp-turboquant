#!/usr/bin/env python3
"""
TriAttention H6.1 Generation-Mode Evaluation

Tests TriAttention KV-cache page eviction in autoregressive (generation) mode,
the only mode where eviction actually fires.  In batch/perplexity mode, all
pages are pinned as "current_batch_pages" and eviction never triggers.

Methodology:
  1. Fill a long prompt into the KV cache (forces pages to be allocated).
  2. Generate N tokens autoregressively (forces potential eviction at every step).
  3. Measure:
       - Generation throughput (tokens/s)
       - ROUGE-L / semantic similarity of output vs. full-KV baseline
       - Whether output is coherent (non-degenerate) at varying budgets

Usage:
  python3 scripts/triattention_generation_eval.py \\
    --model qwen2.5-coder-1.5b-bf16.gguf \\
    --context-len 1024 \\
    --n-predict 128 \\
    --page-budgets 4 8 16 32 64 \\
    --extra-args "-ngl 99 -fa on" \\
    --prompt-file data/wikitext2-test.txt \\
    --output research/milestone-008/generation_eval.json

Hypothesis H6.1 (from M006 audit):
  TriAttention page eviction maintains >= 95% quality vs full-KV baseline
  at 50% page budget in generation mode.

Status: INDETERMINADO — previous eval used batch mode (perplexity) where
  eviction is blocked by current_batch_pages protection.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from statistics import mean, stdev

DEFAULT_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In machine learning, attention mechanisms allow models to dynamically "
    "focus on relevant parts of the input. Transformers use multi-head "
    "self-attention to capture long-range dependencies in sequences. "
    "The key-value cache stores intermediate attention states for reuse "
    "during autoregressive generation, enabling efficient decoding. "
    "However, the KV cache grows linearly with context length, creating "
    "memory pressure for long sequences. Page eviction strategies like "
    "TriAttention selectively discard low-importance KV pages based on "
    "attention score analysis, approximating the full-attention output "
    "while using a fraction of the memory budget. "
    * 8  # ~500 tokens — enough to fill several pages
)

BINARY_SEARCH_PATHS = [
    "./build/bin/llama-cli",
    "./build-tri/bin/llama-cli",
]


def find_binary(custom_path=None):
    if custom_path:
        if os.path.isfile(custom_path):
            return custom_path
        print(f"Error: binary not found: {custom_path}", file=sys.stderr)
        sys.exit(1)
    for p in BINARY_SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    print("Error: llama-cli not found. Specify with --binary.", file=sys.stderr)
    sys.exit(1)


def run_generation(binary, model, prompt, context_len, n_predict, page_budget,
                   extra_args, runs=3, env=None):
    """Run llama-cli in generation mode and return (texts, timings)."""
    base_cmd = [
        binary, "-m", model,
        "-c", str(context_len),
        "-n", str(n_predict),
        "--no-display-prompt",
        "-p", prompt,
    ] + extra_args

    if page_budget is not None:
        base_cmd += ["--triattention-page-budget", str(page_budget)]

    texts = []
    timings = []
    for i in range(runs):
        print(f"  run {i+1}/{runs}: {' '.join(base_cmd[:6])} ...", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(
            base_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            print(f"  Warning: non-zero exit code {result.returncode}", file=sys.stderr)

        combined = result.stdout + "\n" + result.stderr

        # Extract generated text (stdout is the generation, stderr has timing)
        text = result.stdout.strip()
        texts.append(text)

        # Extract tokens/s from stderr timing line
        m = re.search(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens.*?=\s*([\d.]+)\s*tokens/s",
                      combined, re.IGNORECASE)
        if m:
            tps = float(m.group(2))
        else:
            # Fallback: n_predict / elapsed
            tps = n_predict / elapsed if elapsed > 0 else 0.0
        timings.append(tps)

        if result.returncode != 0 and "abort" in combined.lower():
            print(f"  ABORT detected — skipping remaining runs", file=sys.stderr)
            break

    return texts, timings


def rouge_l(ref_tokens, hyp_tokens):
    """Compute ROUGE-L (F1) between two token lists."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    n, m = len(ref_tokens), len(hyp_tokens)
    # DP for LCS
    dp = [[0] * (m + 1) for _ in range(2)]
    lcs = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                lcs = max(lcs, dp[i % 2][j])
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def word_tokens(text):
    return re.findall(r"\w+", text.lower())


def is_degenerate(text, threshold=0.5):
    """True if text has excessive repetition (sign of KV corruption)."""
    words = word_tokens(text)
    if len(words) < 10:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold


def evaluate_budgets(binary, model, prompt, context_len, n_predict,
                     page_budgets, extra_args, runs, baseline_texts):
    results = {}
    for budget in page_budgets:
        label = f"budget_{budget}"
        print(f"\n[{label}] page_budget={budget}", flush=True)
        texts, timings = run_generation(
            binary, model, prompt, context_len, n_predict,
            budget, extra_args, runs=runs
        )
        rouge_scores = [
            rouge_l(word_tokens(ref), word_tokens(hyp))
            for ref, hyp in zip(baseline_texts, texts)
        ]
        degenerate = [is_degenerate(t) for t in texts]
        results[label] = {
            "page_budget": budget,
            "texts": texts,
            "timings_tps": timings,
            "mean_tps": mean(timings) if timings else 0.0,
            "rouge_l_scores": rouge_scores,
            "mean_rouge_l": mean(rouge_scores) if rouge_scores else 0.0,
            "degenerate_runs": sum(degenerate),
            "total_runs": len(texts),
        }
        print(f"  mean ROUGE-L: {results[label]['mean_rouge_l']:.3f}  "
              f"mean t/s: {results[label]['mean_tps']:.1f}  "
              f"degenerate: {sum(degenerate)}/{len(texts)}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary",       help="Path to llama-cli binary")
    ap.add_argument("--model",        required=True, help="Path to GGUF model")
    ap.add_argument("--context-len",  type=int, default=1024)
    ap.add_argument("--n-predict",    type=int, default=128, help="Tokens to generate")
    ap.add_argument("--page-budgets", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64],
                    help="List of page budgets to test (in pages of 32 tokens)")
    ap.add_argument("--runs",         type=int, default=3)
    ap.add_argument("--prompt-file",  help="Path to text file; first 500 words used as prompt")
    ap.add_argument("--prompt",       help="Explicit prompt string (overrides --prompt-file)")
    ap.add_argument("--extra-args",   default="-ngl 99 -fa on",
                    help="Extra args for llama-cli (space-separated)")
    ap.add_argument("--output",       default="research/milestone-008/generation_eval.json")
    ap.add_argument("--no-paging-env", action="store_true",
                    help="Use LLAMA_NO_PAGING=1 for baseline (confirms eviction is off)")
    args = ap.parse_args()

    binary = find_binary(args.binary)
    extra  = args.extra_args.split() if args.extra_args else []

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            text = f.read()
        words = text.split()[:600]
        prompt = " ".join(words)
    else:
        prompt = DEFAULT_PROMPT

    print(f"=== TriAttention Generation Eval ===")
    print(f"Binary:      {binary}")
    print(f"Model:       {args.model}")
    print(f"Context:     {args.context_len}")
    print(f"n-predict:   {args.n_predict}")
    print(f"Page budgets:{args.page_budgets}")
    print(f"Runs/config: {args.runs}")
    print(f"Prompt len:  {len(prompt.split())} words")
    print()

    # Step 1: Baseline (no eviction)
    print("[baseline] running with no eviction (LLAMA_NO_PAGING=1 or large budget) ...")
    baseline_env = dict(os.environ)
    if args.no_paging_env:
        baseline_env["LLAMA_NO_PAGING"] = "1"

    # Use a very large budget to ensure no eviction (context/32 pages)
    max_budget = (args.context_len + 31) // 32
    baseline_texts, baseline_timings = run_generation(
        binary, args.model, prompt, args.context_len, args.n_predict,
        max_budget, extra, runs=args.runs,
        env=baseline_env if args.no_paging_env else None,
    )

    baseline_result = {
        "page_budget": max_budget,
        "texts": baseline_texts,
        "timings_tps": baseline_timings,
        "mean_tps": mean(baseline_timings) if baseline_timings else 0.0,
        "rouge_l_scores": [1.0] * len(baseline_texts),
        "mean_rouge_l": 1.0,
        "degenerate_runs": sum(is_degenerate(t) for t in baseline_texts),
        "total_runs": len(baseline_texts),
    }
    print(f"  baseline mean t/s: {baseline_result['mean_tps']:.1f}")
    if baseline_texts:
        print(f"  baseline sample: {baseline_texts[0][:120]!r}")

    # Step 2: Evaluate each budget
    budget_results = evaluate_budgets(
        binary, args.model, prompt, args.context_len, args.n_predict,
        args.page_budgets, extra, args.runs, baseline_texts
    )

    # Step 3: H6.1 hypothesis check
    print("\n=== H6.1 Hypothesis Check ===")
    print("H6.1: TriAttention maintains >= 95% quality at 50% page budget")
    half_budget = max(1, max_budget // 2)
    half_label = None
    for label, r in budget_results.items():
        if r["page_budget"] == half_budget or abs(r["page_budget"] - half_budget) <= 2:
            half_label = label
            break
    if half_label:
        rouge = budget_results[half_label]["mean_rouge_l"]
        quality_pct = rouge * 100
        h61_pass = quality_pct >= 95.0
        print(f"  50% budget = {half_budget} pages → ROUGE-L = {rouge:.3f} ({quality_pct:.1f}%)")
        print(f"  H6.1: {'PASS ✓' if h61_pass else 'FAIL ✗'}")
    else:
        print(f"  No result near 50% budget ({half_budget} pages). Budgets tested: {args.page_budgets}")
        h61_pass = None

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "metadata": {
            "binary":      binary,
            "model":       args.model,
            "context_len": args.context_len,
            "n_predict":   args.n_predict,
            "runs":        args.runs,
            "extra_args":  extra,
            "prompt_len_words": len(prompt.split()),
            "max_budget_pages": max_budget,
            "hypothesis":  "H6.1: >= 95% ROUGE-L at 50% page budget in generation mode",
            "h6_1_result": "PASS" if h61_pass else ("FAIL" if h61_pass is False else "INDETERMINATE"),
        },
        "baseline": baseline_result,
        "budgets":  budget_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Budget':>8} {'Pages':>6} {'ROUGE-L':>8} {'t/s':>8} {'Degen':>6}")
    print("-" * 42)
    for label, r in sorted(budget_results.items(), key=lambda x: x[1]["page_budget"]):
        degen = f"{r['degenerate_runs']}/{r['total_runs']}"
        print(f"{r['page_budget']:>8} {r['page_budget']*32:>6} {r['mean_rouge_l']:>8.3f} "
              f"{r['mean_tps']:>8.1f} {degen:>6}")
    print(f"{'baseline':>8} {max_budget*32:>6} {'1.000':>8} {baseline_result['mean_tps']:>8.1f} "
          f"{'0/' + str(args.runs):>6}")


if __name__ == "__main__":
    main()
