#!/usr/bin/env python3
"""
Benchmark Infrastructure for M008: Latency and Throughput measurement.

Measures tokens/s, time-to-first-token, and total generation time across
three configurations: baseline, native-paged, and triattention.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from statistics import mean, stdev

RESULT_FILE = "research/milestone-008/benchmark_results.json"

CONFIGURATIONS = {
    "baseline": [],
    "native-paged": [],
    "triattention": ["--triattention-page-budget", "512"],
}


def find_binary(custom_path=None):
    if custom_path:
        if os.path.isfile(custom_path):
            return custom_path
        print(f"Error: binary not found: {custom_path}", file=sys.stderr)
        sys.exit(1)
    for path in ["./build/bin/llama-cli", "./build-tri/bin/llama-cli"]:
        if os.path.isfile(path):
            return path
    print("Error: could not find llama-cli in build directories.", file=sys.stderr)
    sys.exit(1)


def run_single(binary, model, prompt, n_tokens, ngl, extra_flags):
    cmd = [
        binary, "-m", model,
        "-p", prompt,
        "-n", str(n_tokens),
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "-ngl", str(ngl),
    ] + extra_flags

    t_start = time.monotonic()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace")
    t_total = time.monotonic() - t_start

    out = result.stdout + "\n" + result.stderr

    def _parse_num(s):
        return float(s.replace(",", ".")) if s else None

    # Format: [ Prompt: 1231,5 t/s | Generation: 211,9 t/s ]
    prompt_tps = None
    gen_tps = None
    m = re.search(r"Prompt:\s*([\d,\.]+)\s*t/s\s*\|\s*Generation:\s*([\d,\.]+)\s*t/s", out)
    if m:
        prompt_tps = _parse_num(m.group(1))
        gen_tps = _parse_num(m.group(2))

    if result.returncode != 0 and gen_tps is None:
        snippet = "\n".join(out.splitlines()[-15:])
        print(f"  [WARN] run failed (exit {result.returncode}):\n{snippet}", file=sys.stderr)

    return {"prompt_tps": prompt_tps, "gen_tps": gen_tps, "wall_s": t_total}


def bench_config(name, binary, model, prompt, n_tokens, ngl, runs, extra_flags):
    print(f"  Config [{name}] — {runs} run(s)...")
    results = []
    for i in range(runs):
        r = run_single(binary, model, prompt, n_tokens, ngl, extra_flags)
        ptps_str = f"{r['prompt_tps']:.1f}" if r["prompt_tps"] else "N/A"
        gtps_str = f"{r['gen_tps']:.2f}" if r["gen_tps"] else "N/A"
        print(f"    run {i+1}/{runs}: Prompt={ptps_str}t/s  Gen={gtps_str}t/s  wall={r['wall_s']:.2f}s")
        results.append(r)

    prompt_tpss = [r["prompt_tps"] for r in results if r["prompt_tps"] is not None]
    gen_tpss = [r["gen_tps"] for r in results if r["gen_tps"] is not None]
    walls = [r["wall_s"] for r in results]

    return {
        "prompt_tps_mean": mean(prompt_tpss) if prompt_tpss else None,
        "prompt_tps_stdev": stdev(prompt_tpss) if len(prompt_tpss) > 1 else None,
        "gen_tps_mean": mean(gen_tpss) if gen_tpss else None,
        "gen_tps_stdev": stdev(gen_tpss) if len(gen_tpss) > 1 else None,
        "wall_s_mean": mean(walls),
    }


def pct_change(baseline, value):
    if baseline and value and baseline != 0:
        return (value - baseline) / baseline * 100.0
    return None


def print_table(results):
    configs = list(results.keys())
    baseline = results.get("baseline", {})

    col = 20
    sep = "+" + ("-" * col + "+") * (len(configs) + 1)
    header = f"| {'Metric':<{col-2}} |"
    for c in configs:
        header += f" {c:<{col-2}} |"

    print("\n" + "=" * len(sep))
    print(f" {'M008 BENCHMARK SUMMARY':^{len(sep)-2}}")
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)

    def row(label, key, fmt):
        line = f"| {label:<{col-2}} |"
        base_val = baseline.get(key)
        for c in configs:
            val = results[c].get(key)
            if val is None:
                cell = "N/A"
            else:
                cell = fmt(val)
                if c != "baseline" and base_val:
                    delta = pct_change(base_val, val)
                    if delta is not None:
                        cell += f" ({delta:+.1f}%)"
            line += f" {cell:<{col-2}} |"
        print(line)

    row("Prompt (t/s)", "prompt_tps_mean", lambda v: f"{v:.1f}")
    row("Generation (t/s)", "gen_tps_mean", lambda v: f"{v:.2f}")
    row("Wall time (s)", "wall_s_mean", lambda v: f"{v:.2f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="M008 Latency/Throughput Benchmark")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 20,
                        help="Prompt text to benchmark")
    parser.add_argument("--n-tokens", type=int, default=64, help="Tokens to generate per run (default 64)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration (default 3)")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers (default 99)")
    parser.add_argument("--binary", help="Explicit path to llama-cli")
    parser.add_argument("--configs", nargs="+",
                        choices=list(CONFIGURATIONS.keys()),
                        default=list(CONFIGURATIONS.keys()),
                        help="Configurations to run (default: all)")
    parser.add_argument("--extra-args", default="",
                        help="Extra args passed to every configuration")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    binary = find_binary(args.binary)
    print(f"Binary : {binary}")
    print(f"Model  : {os.path.basename(args.model)}")
    print(f"Tokens : {args.n_tokens}  Runs: {args.runs}  GPU layers: {args.ngl}")

    global_extra = args.extra_args.strip().split() if args.extra_args.strip() else []

    results = {}
    for name in args.configs:
        flags = CONFIGURATIONS[name] + global_extra
        results[name] = bench_config(name, binary, args.model, args.prompt,
                                     args.n_tokens, args.ngl, args.runs, flags)

    print_table(results)

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    payload = {
        "model": os.path.basename(args.model),
        "n_tokens": args.n_tokens,
        "runs": args.runs,
        "configurations": results,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    print(f"\nResults written to {RESULT_FILE}")


if __name__ == "__main__":
    main()
