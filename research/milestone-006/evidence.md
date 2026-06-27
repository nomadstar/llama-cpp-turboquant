# Milestone 006: Evidence

*Estado: IMPLEMENTADO — build OK, pendiente validación GPU/numerical.*

## Cambios por Archivo

| Archivo | Cambio |
|---|---|
| `include/llama.h` | Exposición pública del presupuesto de páginas TriAttention |
| `src/llama-cparams.h` | Campo de parámetros para `triattention_page_budget` |
| `src/llama-context.cpp` | Cableado de parámetros de contexto |
| `common/common.h` | Declaración de opción CLI/común |
| `common/common.cpp` | Default y propagación del presupuesto |
| `common/arg.cpp` | Parsing de `--triattention-page-budget` |
| `src/llama-kv-cache.h` | Declaraciones de estado y API de eviction |
| `src/llama-kv-cache.cpp` | Reserva de bloque 0, `pg_score_and_evict()`, desalojo bajo presupuesto |
| `src/llama-model.cpp` | Integración de configuración en runtime/modelo |

## Resumen de Evidencia

- Total de archivos modificados: 9
- Delta reportado: 244 líneas
- Feature flag: `--triattention-page-budget N`
- Build: pasa

## Fix Pass Posterior a la Crítica

- La crítica encontró un issue **P1 BLOCKER** en `get_unrotated_key()`
  (`src/llama-kv-cache.cpp:2793`, uso efectivo en `src/llama-kv-cache.cpp:2833-2834`): la
  un-rotation RoPE estaba apoyándose en `hparams.rope_freq_base_train` en lugar del valor efectivo
  derivado de `cparams`, rompiendo el scoring en modelos con YaRN/NTK-aware scaling.
- La crítica encontró un issue **P3 MEDIUM** en `pg_alloc_for_sinfo()`
  (`src/llama-kv-cache.cpp:1219`, gate de eviction en el bloque que ahora empieza en
  `src/llama-kv-cache.cpp:1246`): el enforcement del presupuesto no debía quedar restringido a
  decode de un solo token. La guardia `sinfo.size() == 1` fue removida para que el eviction opere
  también durante prefill.
- La corrección añadió `rope_freq_base_eff` y `rope_freq_scale_eff` a `llama_kv_cache`
  (`src/llama-kv-cache.h:111-112`, estado almacenado en `src/llama-kv-cache.h:330-331`).
- `llama_model.cpp` ahora pasa explícitamente los parámetros efectivos de RoPE al constructor de
  `llama_kv_cache` (`src/llama-model.cpp:8266-8282`).
- Archivos adicionales tocados por esta fix pass: `src/llama-kv-cache.h`,
  `src/llama-kv-cache.cpp`, `src/llama-model.cpp`.
- Build después de las correcciones: pasa.

## Pendiente

- [ ] Validación numérica GPU
- [ ] Calibración de scores TriAttention
- [ ] Verificación de hipótesis H6.1 frente al baseline
