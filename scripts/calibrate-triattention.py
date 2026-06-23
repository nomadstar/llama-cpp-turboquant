#!/usr/bin/env python3
"""
calibrate-triattention.py — Genera el archivo .triattention para TriAttention.

TriAttention comprime el KV cache usando estadísticas de atención por cabeza.
Las cabezas con patrones locales/predecibles se comprimen con turbo3;
las cabezas globales se mantienen en q8_0.

Uso:
    python3 scripts/calibrate-triattention.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --corpus corpus.txt \
        --output modelo.triattention

Requiere: pip install -r scripts/requirements-triattention.txt
"""

import argparse
import json
import math
import os
import random
import sys
import time

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibra estadísticas TriAttention y genera .triattention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",       required=True,  help="ID de HuggingFace o ruta local")
    p.add_argument("--corpus",      required=True,  help="Archivo de texto plano UTF-8")
    p.add_argument("--output",      required=True,  help="Ruta del archivo .triattention de salida")
    p.add_argument("--n-samples",   type=int,   default=512,    help="Ventanas aleatorias a procesar (default: 512)")
    p.add_argument("--max-seq-len", type=int,   default=2048,   help="Tokens por ventana (default: 2048)")
    p.add_argument("--sample-heads",type=int,   default=0,      help="Cabezas a muestrear, 0 = todas (default: 0)")
    p.add_argument("--vram-gb",     type=float, default=2.0,    help="Límite VRAM en GB (default: 2)")
    p.add_argument("--ram-gb",      type=float, default=20.0,   help="Límite RAM CPU en GB (default: 20)")
    p.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"],
                                    help="Precisión del modelo (default: bfloat16)")
    p.add_argument("--seed",        type=int,   default=42,     help="Semilla aleatoria (default: 42)")
    return p.parse_args()


def load_model(model_id, dtype_str, vram_gb, ram_gb):
    """Carga el modelo con restricciones de memoria."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: Instala dependencias con: pip install -r scripts/requirements-triattention.txt", file=sys.stderr)
        sys.exit(1)

    dtype = getattr(__import__("torch"), dtype_str)

    # Construir max_memory para device_map
    max_memory = {}
    if __import__("torch").cuda.is_available():
        n_gpus = __import__("torch").cuda.device_count()
        vram_per_gpu = int(vram_gb * 1024)  # MB
        for i in range(n_gpus):
            max_memory[i] = f"{vram_per_gpu}MB"
    ram_mb = int(ram_gb * 1024)
    max_memory["cpu"] = f"{ram_mb}MB"

    print(f"[1/3] Cargando tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Cargando modelo ({dtype_str}, VRAM≤{vram_gb}GB, RAM≤{ram_gb}GB)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        output_attentions=False,  # manejaremos hooks manualmente
    )
    model.eval()
    print(f"       Modelo cargado. Dispositivos: {set(str(p.device) for p in model.parameters())}", flush=True)
    return tokenizer, model


def read_corpus_windows(corpus_path, tokenizer, n_samples, max_seq_len, seed):
    """Lee el corpus y genera ventanas de tokens aleatorias."""
    print(f"[3/3] Leyendo corpus: {corpus_path}", flush=True)
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"       Corpus: {len(text):,} caracteres", flush=True)

    # Tokenizar todo el corpus de una vez (sin padding, sin truncación)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    print(f"       Tokens totales: {total_tokens:,}", flush=True)

    if total_tokens < max_seq_len:
        print(f"ADVERTENCIA: corpus demasiado corto ({total_tokens} tokens < {max_seq_len}). "
              "Reduciendo max_seq_len.", file=sys.stderr)
        max_seq_len = total_tokens // 2

    rng = random.Random(seed)
    windows = []
    for _ in range(n_samples):
        start = rng.randint(0, total_tokens - max_seq_len)
        window = token_ids[start : start + max_seq_len]
        windows.append(window)

    return windows, max_seq_len


def collect_attention_stats(model, tokenizer, windows, sample_heads, max_seq_len):
    """
    Recolecta estadísticas de atención por capa y cabeza.

    Por cada ventana, registra para cada capa/cabeza:
      - mean_distance: distancia promedio ponderada (en tokens) a la que atiende
      - entropy: entropía de la distribución de atención
    """
    import torch

    config = model.config
    n_layers  = config.num_hidden_layers
    n_heads   = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    # Acumuladores: shape [n_layers, n_heads]
    sum_distance = [[0.0] * n_heads for _ in range(n_layers)]
    sum_entropy  = [[0.0] * n_heads for _ in range(n_layers)]
    count        = [[0]   * n_heads for _ in range(n_layers)]

    # Hook para capturar atención por capa
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # outputs puede ser tuple; la atención es el segundo elemento si output_attentions=True
            # Alternativa: capturar los pesos desde el módulo interno
            # Usamos el patrón: interceptar justo después del softmax
            pass
        return hook

    # Estrategia más robusta: usar output_attentions en el forward
    n_samples = len(windows)

    print(f"\nRecolectando estadísticas ({n_samples} ventanas, {n_layers} capas, {n_heads} cabezas)...",
          flush=True)

    t0 = time.time()
    processed = 0

    for win_idx, window in enumerate(windows):
        if win_idx % max(1, n_samples // 20) == 0:
            elapsed = time.time() - t0
            pct = win_idx / n_samples * 100
            eta = (elapsed / max(win_idx, 1)) * (n_samples - win_idx)
            print(f"  {pct:5.1f}%  ventana {win_idx}/{n_samples}  ETA: {eta:.0f}s", flush=True)

        input_ids = torch.tensor([window], dtype=torch.long)
        # Mover al mismo dispositivo que el primer parámetro del modelo
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
            )

        # outputs.attentions: tuple de [1, n_heads, seq, seq] por capa
        attentions = outputs.attentions  # None si modelo no soporta

        if attentions is None:
            print("ADVERTENCIA: el modelo no retorna atenciones. "
                  "Prueba con un modelo HuggingFace que soporte output_attentions.", file=sys.stderr)
            sys.exit(1)

        for layer_idx, attn in enumerate(attentions):
            # attn: [batch=1, n_heads, seq_len, seq_len]  (float32 o bfloat16)
            attn = attn[0].float()  # [n_heads, seq_len, seq_len]

            heads_to_process = range(n_heads)
            if sample_heads > 0:
                heads_to_process = random.sample(range(n_heads), min(sample_heads, n_heads))

            for h in heads_to_process:
                a = attn[h]  # [seq_len, seq_len] — atención causal

                # Distancia promedio: para cada query q, promedio ponderado de (q - k)
                positions = torch.arange(seq_len, dtype=torch.float32, device=a.device)
                # a[q, k] = peso de q → k  (triangular inferior, normalizada)
                # distancia = q - k (cuántos tokens atrás mira)
                query_pos = positions.unsqueeze(1)   # [seq, 1]
                key_pos   = positions.unsqueeze(0)   # [1, seq]
                dist_mat  = (query_pos - key_pos).clamp(min=0)  # [seq, seq]

                # Solo mirar la parte causal válida (a ya es causal del modelo)
                mean_dist_per_query = (a * dist_mat).sum(dim=1)  # [seq]
                mean_dist = mean_dist_per_query.mean().item()

                # Entropía: H = -sum(a * log(a+eps))
                eps = 1e-9
                entropy_per_query = -(a * (a + eps).log()).sum(dim=1)  # [seq]
                mean_entropy = entropy_per_query.mean().item()

                sum_distance[layer_idx][h] += mean_dist
                sum_entropy[layer_idx][h]  += mean_entropy
                count[layer_idx][h]        += 1

        processed += 1
        del outputs

    elapsed = time.time() - t0
    print(f"  100.0%  Completado en {elapsed:.1f}s", flush=True)

    # Promediar
    mean_distance = []
    mean_entropy  = []
    for layer_idx in range(n_layers):
        row_d = []
        row_e = []
        for h in range(n_heads):
            c = count[layer_idx][h]
            row_d.append(sum_distance[layer_idx][h] / c if c > 0 else 0.0)
            row_e.append(sum_entropy[layer_idx][h]  / c if c > 0 else 0.0)
        mean_distance.append(row_d)
        mean_entropy.append(row_e)

    return mean_distance, mean_entropy, n_layers, n_heads, n_kv_heads


def compute_importance(mean_distance, mean_entropy, n_layers, n_heads):
    """
    Calcula un score de importancia por capa [0..1].
    Capas con heads que miran lejos (alta distancia) son más importantes → q8_0.
    Capas con heads locales (baja distancia) son comprimibles → turbo3.

    importance = 0 → comprimible (turbo3)
    importance = 1 → importante (q8_0)
    """
    # Score por capa = promedio de mean_distance de todas las cabezas
    layer_scores = []
    for layer_idx in range(n_layers):
        avg_dist = sum(mean_distance[layer_idx]) / n_heads
        layer_scores.append(avg_dist)

    # Normalizar a [0, 1]
    min_s = min(layer_scores)
    max_s = max(layer_scores)
    rng = max_s - min_s if max_s > min_s else 1.0

    importance = [(s - min_s) / rng for s in layer_scores]
    return importance


def assign_cache_types(importance, n_layers, budget_tokens, max_seq_len):
    """
    Asigna cache_type_k y cache_type_v por capa basándose en importancia y budget.

    El budget_tokens controla qué fracción de capas usa q8_0:
    - budget = max_seq_len → todo q8_0 (sin compresión)
    - budget = 0           → todo turbo3 (máxima compresión)
    - presupuesto intermedio → las capas más importantes usan q8_0

    Simplificación: fracción de capas en q8_0 = budget / max_seq_len
    Hasta max 1.0, mín 0.0.
    """
    fraction_q8 = min(1.0, max(0.0, budget_tokens / max_seq_len))
    n_q8_layers = max(1, round(fraction_q8 * n_layers))

    # Las n_q8_layers capas con mayor importancia → q8_0
    sorted_layers = sorted(range(n_layers), key=lambda i: importance[i], reverse=True)
    q8_set = set(sorted_layers[:n_q8_layers])

    cache_types = []
    for layer_idx in range(n_layers):
        if layer_idx in q8_set:
            cache_types.append({"cache_type_k": "q8_0", "cache_type_v": "q8_0"})
        else:
            cache_types.append({"cache_type_k": "q8_0", "cache_type_v": "turbo3"})

    return cache_types, n_q8_layers


def save_triattention(output_path, model_id, args, n_layers, n_heads, n_kv_heads,
                      mean_distance, mean_entropy, importance, cache_types):
    """Guarda el archivo .triattention (JSON)."""

    # Redondear a 4 decimales para mantener archivo legible
    def r4(lst):
        if isinstance(lst[0], list):
            return [[round(x, 4) for x in row] for row in lst]
        return [round(x, 4) for x in lst]

    data = {
        "format":    "triattention-v1",
        "model":     model_id,
        "n_layers":  n_layers,
        "n_heads":   n_heads,
        "n_kv_heads": n_kv_heads,
        "calibration": {
            "n_samples":   args.n_samples,
            "max_seq_len": args.max_seq_len,
            "corpus":      os.path.basename(args.corpus),
        },
        "stats": {
            "mean_distance": r4(mean_distance),
            "entropy":       r4(mean_entropy),
            "layer_importance": r4(importance),
        },
        "layers": cache_types,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Guardado: {output_path}", flush=True)


def print_summary(importance, cache_types, n_layers):
    """Imprime un resumen de la calibración."""
    n_q8   = sum(1 for t in cache_types if t["cache_type_v"] == "q8_0")
    n_turbo = n_layers - n_q8

    print("\n─── Resumen de calibración ──────────────────────────────")
    print(f"  Capas totales    : {n_layers}")
    print(f"  Capas q8_0  (K+V): {n_q8}")
    print(f"  Capas turbo3 (V) : {n_turbo}")
    print()
    print("  Importancia por capa (↑ = más importante = q8_0):")
    for i, (imp, ct) in enumerate(zip(importance, cache_types)):
        bar = "█" * int(imp * 20)
        tag = "q8_0 " if ct["cache_type_v"] == "q8_0" else "turbo3"
        print(f"  L{i:3d}  [{bar:<20s}] {imp:.3f}  →  {tag}")
    print("─────────────────────────────────────────────────────────")


def main():
    args = parse_args()
    random.seed(args.seed)

    # Verificar corpus
    if not os.path.isfile(args.corpus):
        print(f"ERROR: no se encuentra el corpus: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  TriAttention Calibration")
    print("=" * 60)
    print(f"  Modelo   : {args.model}")
    print(f"  Corpus   : {args.corpus}")
    print(f"  Salida   : {args.output}")
    print(f"  Muestras : {args.n_samples}  |  SeqLen: {args.max_seq_len}")
    print(f"  VRAM     : {args.vram_gb} GB  |  RAM: {args.ram_gb} GB")
    print("=" * 60)
    print()

    tokenizer, model = load_model(args.model, args.dtype, args.vram_gb, args.ram_gb)

    windows, max_seq_len = read_corpus_windows(
        args.corpus, tokenizer, args.n_samples, args.max_seq_len, args.seed
    )

    mean_distance, mean_entropy, n_layers, n_heads, n_kv_heads = collect_attention_stats(
        model, tokenizer, windows, args.sample_heads, max_seq_len
    )

    importance = compute_importance(mean_distance, mean_entropy, n_layers, n_heads)

    # Budget por defecto = max_seq_len / 2 (50% de capas en q8_0)
    budget = max_seq_len // 2
    cache_types, n_q8 = assign_cache_types(importance, n_layers, budget, max_seq_len)

    save_triattention(
        args.output, args.model, args,
        n_layers, n_heads, n_kv_heads,
        mean_distance, mean_entropy, importance, cache_types
    )

    print_summary(importance, cache_types, n_layers)

    print(f"""
Para usar en inferencia:
  ./build/bin/llama-cli \\
      -m tu_modelo.gguf \\
      --triattention-stats {args.output} \\
      --triattention-budget {budget}
""")


if __name__ == "__main__":
    main()
