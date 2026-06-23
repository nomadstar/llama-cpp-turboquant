#!/bin/bash
set -x

CCACHE_FLAGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# ── Detectar GPU: AMD (ROCm/HIP) o NVIDIA (CUDA) ────────────────
detect_gpu_flags() {
    # Preferir variable de entorno si ya está fijada
    if [ -n "$GPU_FLAGS" ]; then
        return
    fi

    # Detectar AMD: rocm-smi o /dev/kfd presente
    if command -v rocm-smi &>/dev/null || [ -e /dev/kfd ]; then
        # Intentar obtener la arquitectura automáticamente
        local arch=""
        if command -v rocminfo &>/dev/null; then
            arch=$(rocminfo 2>/dev/null | awk '/gfx[0-9]/{print $NF; exit}')
        fi
        if [ -n "$arch" ]; then
            GPU_FLAGS="-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=${arch}"
        else
            # Fallback: RDNA3 es la más común hoy en dia; ajustar si hace falta
            GPU_FLAGS="-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100"
        fi
        GPU_VENDOR="AMD"
    # Detectar NVIDIA: nvidia-smi o /dev/nvidia0 presente
    elif command -v nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
        GPU_FLAGS="-DGGML_CUDA=ON"
        GPU_VENDOR="NVIDIA"
    else
        echo "ADVERTENCIA: No se detectó GPU. Compilando solo CPU."
        GPU_FLAGS=""
        GPU_VENDOR="CPU"
    fi

    echo "=== GPU detectada: ${GPU_VENDOR} | Flags: ${GPU_FLAGS} ==="
}

build_if_changed() {
    local branch="$1"
    local build_dir="$2"
    local hash_file=".last_build_hash_${build_dir}"

    git checkout "$branch"
    local current_hash
    current_hash=$(git rev-parse HEAD)

    if [ -f "$hash_file" ] && [ "$(cat "$hash_file")" = "$current_hash" ]; then
        echo "=== $branch: sin cambios ($(git log -1 --format='%h %s')), saltando compilación ==="
    else
        cmake -B "$build_dir" $GPU_FLAGS $CCACHE_FLAGS
        cmake --build "$build_dir" --config Release --parallel 6
        echo "$current_hash" > "$hash_file"
    fi
}

detect_gpu_flags

# ── Compilar ambas ramas (solo si hubo cambios) ─────────────────
build_if_changed feature/triattention-paged build-tri
build_if_changed feature/supermerge         build-super

# ── Correctitud (una vez por rama) ─────────────────────────────
./build-tri/bin/test-quantize-fns  > triattention_results.txt     2>&1 || true
./build-tri/bin/test-backend-ops   > triattention_backend_ops.txt  2>&1 || true
./build-super/bin/test-quantize-fns > supermerge_results.txt       2>&1 || true
./build-super/bin/test-backend-ops  > supermerge_backend_ops.txt   2>&1 || true

# ── Benchmarks por modelo ───────────────────────────────────────
OLLAMA_MANIFESTS_DIR="$HOME/.ollama/models/manifests"

if [ ! -d "$OLLAMA_MANIFESTS_DIR" ]; then
    echo "No se encontró el directorio de manifiestos de Ollama."
    exit 1
fi

find "$OLLAMA_MANIFESTS_DIR" -type f | while read -r manifest; do
    model_path="${manifest#$OLLAMA_MANIFESTS_DIR/}"
    model_name=$(echo "$model_path" | tr '/' '_')

    # Buscar el layer de tipo "model" (no license/template/params)
    digest=$(python3 -c "
import json, sys
try:
    data = json.load(open('$manifest'))
    for layer in data.get('layers', []):
        if layer.get('mediaType','').endswith('.model'):
            print(layer['digest']); break
except: pass
" 2>/dev/null)

    if [ -z "$digest" ]; then
        echo "Sin blob de modelo en $model_name, omitiendo."
        continue
    fi

    blob_file="$HOME/.ollama/models/blobs/${digest/:/-}"
    if [ ! -f "$blob_file" ]; then
        echo "Blob no encontrado: $blob_file"
        continue
    fi

    echo "=== $model_name ==="

    ./build-tri/bin/llama-bench \
      -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "triattention_bench_${model_name}.txt" 2>&1

    ./build-super/bin/llama-bench \
      -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "supermerge_bench_${model_name}.txt" 2>&1

    TRIATTENTION_STATS="${model_name}.triattention"
    if [ -f "$TRIATTENTION_STATS" ]; then
        ./build-super/bin/llama-bench \
          -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 \
          --triattention-stats "$TRIATTENTION_STATS" \
          --triattention-budget 2048 -r 3 \
          > "supermerge_bench_triattention_${model_name}.txt" 2>&1
    fi
done

echo "=== Local Qwen2.5-Coder-1.5B ==="
LOCAL_BLOB="qwen2.5-coder-1.5b-bf16.gguf"
LOCAL_STATS="qwen2.5-coder-1.5b.triattention"
if [ -f "$LOCAL_BLOB" ]; then
    ./build-tri/bin/llama-bench \
      -m "$LOCAL_BLOB" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "triattention_bench_local_qwen.txt" 2>&1

    ./build-super/bin/llama-bench \
      -m "$LOCAL_BLOB" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "supermerge_bench_local_qwen.txt" 2>&1

    if [ -f "$LOCAL_STATS" ]; then
        ./build-super/bin/llama-bench \
          -m "$LOCAL_BLOB" --cache-type-v turbo3 --cache-type-k q8_0 \
          --triattention-stats "$LOCAL_STATS" \
          --triattention-budget 1024 -r 3 \
          > "supermerge_bench_triattention_local_qwen.txt" 2>&1
    fi
fi

echo "Listo."
ls -lh *bench*.txt *results*.txt *backend*.txt 2>/dev/null
zip results.zip *bench*.txt *results*.txt *backend*.txt