# Milestone 006: Conclusions

*Estado: IMPLEMENTADO — correcciones P1/P3 aplicadas, build OK, pendiente de validación numérica.*

## Resumen

Se implementó TriAttention KV eviction sobre el pool paginado de V con presupuesto configurable
de páginas físicas y scoring por claves K con RoPE inverso. La revisión posterior encontró dos
problemas relevantes en la primera pasada de M006 y ambos fueron corregidos antes de dar el hito
por listo.

## Resultado Final

1. **La crítica encontró dos issues reales**: un blocker de correctitud en
   `get_unrotated_key()` (P1, uso incorrecto de `rope_freq_base`) y un issue de enforcement del
   presupuesto durante prefill en `pg_alloc_for_sinfo()` (P3, eviction sólo en decode).
2. **Ambos issues quedaron corregidos antes de shipping**: `llama_kv_cache` ahora almacena
   `rope_freq_base_eff` y `rope_freq_scale_eff`, y la lógica de eviction ya no depende de que el
   batch sea de un solo token.
3. **Estado de correctitud actual**: la implementación queda limpia de problemas conocidos de
   correctitud para escenarios de una sola secuencia.
4. **Compatibilidad de RoPE ampliada**: modelos con escalado YaRN/NTK-aware ahora quedan
   soportados correctamente en el scoring por un-rotation de K.

## Logros

1. **Nuevo control de runtime**: `--triattention-page-budget N` activa eviction cuando el
   presupuesto es mayor que cero.
2. **Bloque dummy reservado**: el bloque físico 0 queda reservado como zero block para páginas
   desalojadas.
3. **Eviction por score**: `pg_score_and_evict()` puntúa páginas residentes y expulsa la de menor
   relevancia al alcanzarse el presupuesto, ahora tanto en prefill como en decode.
4. **Integración end-to-end**: la configuración y el comportamiento quedaron cableados en CLI,
   runtime, modelo y KV cache, incluyendo el transporte explícito de los parámetros efectivos de
   RoPE hacia `llama_kv_cache`.

## Limitaciones Conocidas

1. **P2**: scoring cruzado entre secuencias sigue siendo una limitación conocida, pero de bajo
   impacto para el caso objetivo actual.
2. **P4**: las lecturas host/GPU síncronas en el scoring siguen siendo un problema de rendimiento,
   no de correctitud.
3. **P5**: persiste el desajuste entre el naming de algunas funciones y la especificación.

## Estado de la Hipótesis H6.1

Pendiente. La implementación y la fix pass existen, pero la validación GPU/numerical de
calidad/perplexity todavía no se ha ejecutado. H6.1 sigue abierta hasta completar esa verificación.

## Métricas de Cambio

- Implementación inicial: 9 archivos
- Fix pass adicional: `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-model.cpp`
- Estado de build: pasa
