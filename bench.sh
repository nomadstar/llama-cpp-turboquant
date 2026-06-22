#!/bin/bash
set -x  # Mostrar los comandos mientras se ejecutan

# 1. Asegurarnos de estar en la primera rama
git checkout feature/triattention-paged

# 2. Configurar y compilar con límite de 6 hilos para no colapsar la RAM
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel 6

# 3. Ejecutar los tests de compatibilidad de memoria (quantize) y backend (GPU)
# Usamos "|| true" para que el script no se detenga si una prueba falla
./build/bin/test-quantize-fns > triattention_results.txt 2>&1 || true
./build/bin/test-backend-ops > triattention_backend_ops.txt 2>&1 || true


# 4. Guardar cualquier cambio local y cambiar a la segunda rama
git stash
git checkout feature/supermerge

# 5. Recompilar la segunda rama usando el mismo límite de hilos
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel 6

# 6. Ejecutar las mismas pruebas para la segunda rama
./build/bin/test-quantize-fns > supermerge_results.txt 2>&1 || true
./build/bin/test-backend-ops > supermerge_backend_ops.txt 2>&1 || true

# Agrega esto al final del script, en cada rama:

# Performance real — tokens por segundo
./build/bin/llama-bench \
  -m /ruta/al/modelo.gguf \
  --cache-type-v turbo3 \
  --cache-type-k q8_0 \
  -r 3 \
  > supermerge_bench.txt 2>&1

# Con TriAttention (solo supermerge):
./build/bin/llama-bench \
  -m /ruta/al/modelo.gguf \
  --cache-type-v turbo3 \
  --cache-type-k q8_0 \
  --triattention-stats modelo.triattention \
  --triattention-budget 2048 \
  -r 3 \
  > supermerge_bench_triattention.txt 2>&1
  
echo "¡Proceso terminado! Revisa los archivos .txt generados para ver los resultados."
