# Milestone 004: Evidence

*Estado: COMPLETADO — verificado.*

## Resultados de Tests

```
=== TurboQuant C Round-Trip Test ===
Test 1 (turbo3): e0 = [1, 0, ...]
Test 2 (turbo3): sin*10
Test 3 (turbo4): cos*5
=== Extreme-Input Test Cases (Robustness) ===
Running test: Zero Vector            -> OK
Running test: NaN Input              -> OK
Running test: Inf Input              -> OK
Running test: Single Mass Outlier (1e6) -> OK
Running test: Laplace Distribution with Outliers -> OK
Running test: Combined Extreme Cases -> OK
=== Done ===
```

Todos los asserts `isfinite(output[i])` pasan para turbo3 y turbo4 en los 6 casos extremos.

## Cambios por Archivo

### `ggml/src/ggml-cuda/turbo-quant.cuh` (+9 líneas)

```cuda
// turbo4_dequant_element
if (isnan(norm) || isinf(norm)) { norm = 0.0f; }
// turbo3_dequant_element
if (isnan(norm) || isinf(norm)) { norm = 0.0f; }
// turbo2_dequant_element
if (isnan(norm) || isinf(norm)) { norm = 0.0f; }
```

### `ggml/src/ggml-turbo-quant.c` (+25 líneas)

- `quantize_row_turbo4_0_ref`: reemplaza NaN/Inf en `src[i]` con `0.0f` antes de calcular `norm_sq`; guarda `isfinite(norm)` y `isfinite(corrected_norm)`.
- `dequantize_row_turbo4_0`: guarda `isfinite(norm)` en ambos codepaths (directo y QJL); guarda `isfinite(rnorm)` en QJL.

### `tests/test-turbo-quant.c` (+98 líneas)

- Función `run_roundtrip_test`: cuantiza → de-cuantiza → verifica `isfinite` en toda la salida para turbo3 y turbo4.
- 6 casos: Zero Vector, NaN Input, Inf Input, Single Mass Outlier (1e6), Laplace Distribution with Outliers, Combined Extreme Cases.

### `tests/CMakeLists.txt` (+1 línea)

- `llama_build_and_test(test-turbo-quant.c)`

### `research/state.md` (+7 líneas)

- Actualizado hito activo de `milestone-003` a `milestone-004`.
