# Milestone 004: Explicit turbo4 NaN/Inf Validation

## Objetivo

Garantizar que ningún camino de cuantización/de-cuantización de `turbo4` retorne NaN o Inf bajo ninguna entrada, incluyendo vectores de norma cero, entradas con NaN/Inf, y vectores con outliers extremos.

## Hipótesis

**H4.1**: La validación explícita de NaN/Inf en el cuantizador y de-cuantizador de `turbo4` evitará desbordamientos aritméticos catastróficos durante inferencias prolongadas con rangos dinámicos extremos.

## Criterios de Éxito

1. `test-turbo-quant` pasa con todas las nuevas entradas extremas sin retornar NaN/Inf.
2. `cmake --build build` exits 0.
3. Los casos de prueba cubren: vector cero, vector con NaN, vector con Inf, vector con un único outlier masivo (1e6), distribución Laplace con outliers.

## Archivos a Modificar

- `tests/test-turbo-quant.c` — agregar 6+ casos extremos
- `ggml/src/ggml-turbo-quant.c` — guardar NaN/Inf en quantize y dequantize
- `ggml/src/ggml-cuda/turbo-quant.cuh` — guardar NaN/Inf en `turbo4_dequant_element`
- `research/milestone-004/conclusions.md` — rellenar al finalizar
- `research/milestone-004/evidence.md` — rellenar al finalizar
- `research/state.md` — actualizar hito activo a milestone-004
