#!/usr/bin/env python3
"""
TriAttention Calibration and Numerical Validation Script
"""

import argparse
import os
import sys
import subprocess
import re
import json
import tempfile
import shutil
import statistics

def find_binary(custom_path=None):
    if custom_path:
        if os.path.exists(custom_path) and os.path.isfile(custom_path):
            return custom_path
        else:
            print(f"Error: Specified binary path not found: {custom_path}", file=sys.stderr)
            sys.exit(1)

    # Prefer llama-perplexity; fall back to llama-cli -ppl
    search_paths = [
        "./build/bin/llama-perplexity",
        "./build-tri/bin/llama-perplexity",
        "./build/bin/llama-cli",
        "./build-tri/bin/llama-cli",
    ]
    for path in search_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return path

    print("Error: Could not find llama-perplexity or llama-cli binary in build directories.", file=sys.stderr)
    print("Please specify the path using --binary <path>.", file=sys.stderr)
    sys.exit(1)

def run_perplexity_run(cmd):
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    combined_output = result.stdout + "\n" + result.stderr

    # Match nan/inf/scientific notation in addition to plain decimals
    _num = r"[-+]?(?:nan|inf|[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
    match = re.search(rf"Final estimate:\s*PPL\s*=\s*({_num})", combined_output, re.IGNORECASE)
    if match:
        return float(match.group(1)), combined_output

    match = re.search(rf"PPL\s*=\s*({_num})", combined_output, re.IGNORECASE)
    if match:
        return float(match.group(1)), combined_output

    return None, combined_output

def run_benchmark(binary, model, prompt_file, context_len, page_budget, runs, chunks, extra_args):
    # Construct base command
    cmd = [binary]

    # Append perplexity flag -ppl for llama-cli as required
    if "llama-cli" in os.path.basename(binary):
        cmd.append("-ppl")

    cmd.extend(["-m", model, "-f", prompt_file, "-c", str(context_len)])

    if chunks > 0:
        cmd.extend(["--chunks", str(chunks)])

    if extra_args:
        cmd.extend(extra_args.strip().split())

    # Run baseline without --triattention-page-budget, and eviction with it
    if page_budget > 0:
        cmd.extend(["--triattention-page-budget", str(page_budget)])

    ppl_values = []
    for run_idx in range(1, runs + 1):
        print(f"--- Run {run_idx}/{runs} for page budget {page_budget} ---")
        ppl, output = run_perplexity_run(cmd)
        if ppl is not None:
            print(f"Run {run_idx} PPL: {ppl:.4f}")
            ppl_values.append(ppl)
        else:
            print("Error: Failed to parse perplexity from binary output.", file=sys.stderr)
            print("--- Output Snippet ---", file=sys.stderr)
            print("\n".join(output.splitlines()[-20:]), file=sys.stderr)
            sys.exit(1)

    mean_ppl = statistics.mean(ppl_values)
    return mean_ppl

def main():
    parser = argparse.ArgumentParser(description="TriAttention Calibration and Numerical Validation Script")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--prompt", type=str, help="Direct prompt text to evaluate")
    parser.add_argument("--prompt-file", type=str, help="Path to a file containing prompt/corpus text")
    parser.add_argument("--page-budget", type=int, default=0, help="Physical page budget (default 0 = disabled)")
    parser.add_argument("--context-len", type=int, default=4096, help="Context length (default 4096)")
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions to average results (default 3)")
    parser.add_argument("--chunks", type=int, default=0, help="Max number of chunks to process (default 0 = all)")
    parser.add_argument("--binary", type=str, help="Explicit path to llama-cli binary")
    parser.add_argument("--extra-args", type=str, default="", help="Extra arguments to pass to the binary (e.g. '-ngl 0 -fit off')")

    args = parser.parse_args()

    # Validate model
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Strict Validation of prompt parameters (avoid silent fallbacks)
    if not args.prompt and not args.prompt_file:
        print("Error: Must specify either --prompt or --prompt-file.", file=sys.stderr)
        sys.exit(1)

    prompt_file = None
    temp_dir = None
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file does not exist: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        prompt_file = args.prompt_file
    elif args.prompt:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_prompt.txt")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(args.prompt)
        prompt_file = temp_file_path

    # Find the binary
    binary_path = find_binary(args.binary)
    print(f"Using binary: {binary_path}")

    try:
        # Run Baseline (budget = 0, runs without --triattention-page-budget flag)
        print("\n=== STAGE 1: Running Baseline (No Eviction) ===")
        baseline_ppl = run_benchmark(
            binary=binary_path,
            model=args.model,
            prompt_file=prompt_file,
            context_len=args.context_len,
            page_budget=0,
            runs=args.runs,
            chunks=args.chunks,
            extra_args=args.extra_args
        )
        print(f"Baseline Average PPL: {baseline_ppl:.4f}")

        # Run Eviction (runs with --triattention-page-budget flag)
        print(f"\n=== STAGE 2: Running Eviction (Page Budget = {args.page_budget}) ===")
        eviction_ppl = run_benchmark(
            binary=binary_path,
            model=args.model,
            prompt_file=prompt_file,
            context_len=args.context_len,
            page_budget=args.page_budget,
            runs=args.runs,
            chunks=args.chunks,
            extra_args=args.extra_args
        )
        print(f"Eviction Average PPL: {eviction_ppl:.4f}")

        # Calculate retention quality percentage
        if eviction_ppl > 0:
            quality_retention_pct = (baseline_ppl / eviction_ppl) * 100.0
        else:
            quality_retention_pct = 0.0

        # Create output JSON results
        results = {
            "baseline_ppl": baseline_ppl,
            "eviction_ppl": eviction_ppl,
            "page_budget": args.page_budget,
            "context_len": args.context_len,
            "quality_retention_pct": quality_retention_pct
        }

        # Write to JSON
        output_dir = "research/milestone-007"
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "calibration_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults written to {json_path}")

        # Print structured ASCII summary table
        print("\n" + "="*89)
        print(f" {'TRIATTENTION CALIBRATION SUMMARY':^87}")
        print("="*89)
        print(f" Model: {os.path.basename(args.model)}")
        print(f" Context Length: {args.context_len} | Page Budget: {args.page_budget} | Runs: {args.runs} | Chunks: {args.chunks if args.chunks > 0 else 'All'}")
        print("="*89)
        print("+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+")
        print(f"| {'Metric':<19} | {'Baseline':<19} | {'Eviction':<19} | {'Retention %':<19} |")
        print("+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+")
        print(f"| {'Perplexity (PPL)':<19} | {baseline_ppl:<19.4f} | {eviction_ppl:<19.4f} | {f'{quality_retention_pct:.2f}%':<19} |")
        print("+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+" + "-"*21 + "+")

    finally:
        # Cleanup temp directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
