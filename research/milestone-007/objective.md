# Milestone 007: TriAttention Calibration and Numerical Validation

## Objetivo

Calibrar el scoring de páginas de TriAttention sobre corpus representativos y realizar la validación numérica de la calidad del contexto y perplejidad del modelo bajo presupuestos decrecientes de páginas físicas de la caché KV. El script de calibración ejecutará el binario `llama-cli` con el flag `-ppl` para medir la perplejidad con y sin desalojo de la caché KV.

## Hipótesis

**H6.1**: La remoción adaptativa de páginas usando scoring RoPE (TriAttention) permite mantener el contexto efectivo al 95% de la calidad de perplexidad original usando sólo el 50% de las páginas físicas.

## Criterios de Éxito

1. Lograr un script de calibración automatizado `scripts/triattention_calibrate.py` que permita medir y contrastar la perplejidad del modelo con y sin eviction de KV cache usando el binario `llama-cli` con el flag `-ppl`.
2. Comprobar que bajo un presupuesto del 50% de páginas físicas (64 páginas físicas de KV cache para un contexto de 4096 tokens, basado en `pg_block_size = 32`), la perplejidad obtenida retiene al menos el 95% de la calidad de la perplejidad base (sin desalojo).
3. Generar un reporte estructurado y automatizado en formato JSON (`research/milestone-007/calibration_results.json`) con las métricas clave.

## Archivos a Modificar / Crear

- `tools/cli/cli.cpp` (Modificar para delegar `-ppl` a `llama-perplexity`)
- `src/llama-graph.cpp` (Modificar para buscar el tensor de flash attention subyacente antes de adjuntar la tabla de páginas)
- `scripts/triattention_calibrate.py` (Crear/Modificar)
- `research/milestone-007/calibration_results.json` (Crear mediante script)
- `research/milestone-007/evidence.md` (Crear stub)
- `research/milestone-007/conclusions.md` (Crear stub)
