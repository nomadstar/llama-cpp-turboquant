// Compile: cd /home/ignatus/GitHub/llama-cpp-turboquant
//   nvcc -O2 -arch=sm_86 -std=c++17 \
//        -I ggml/include -I ggml/src \
//        tests/test-turbo3-dequant.cu -o /tmp/test-turbo3-dequant
// Run: /tmp/test-turbo3-dequant

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ggml-common.h"
#include "ggml-cuda/turbo-quant.cuh"

static inline void cuda_check(cudaError_t err, const char * what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static inline uint8_t nearest_centroid_3bit(float val) {
    if      (val < -0.154259f) return 0;
    else if (val < -0.091775f) return 1;
    else if (val < -0.043589f) return 2;
    else if (val <  0.0f)      return 3;
    else if (val <  0.043589f) return 4;
    else if (val <  0.091775f) return 5;
    else if (val <  0.154259f) return 6;
    else                       return 7;
}

static constexpr float HOST_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static void fwht_128(float * x, const float * signs1, const float * signs2) {
    for (int i = 0; i < 128; ++i) x[i] *= signs1[i];
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) x[i] = x[i] * inv_sqrt_128 * signs2[i];
}

static void print_idx_block(const uint8_t * idx) {
    std::printf("CPU-quantized idx values:\n");
    for (int i = 0; i < 128; ++i) {
        std::printf("%u%s", (unsigned) idx[i], (i + 1 == 128) ? "\n" : ((i % 32 == 31) ? "\n" : " "));
    }
}

__global__ void dequant_turbo3_single(const block_turbo3_0 * blk, float * out) {
    const int j = threadIdx.x;
    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
    const uint8_t idx   = low2 | (hi1 << 2);
    out[j] = TURBO_CENTROIDS_3BIT[idx] * norm;
}

int main() {
    float h_signs1[128];
    float h_signs2[128];
    cuda_check(cudaMemcpyFromSymbol(h_signs1, TURBO_WHT_SIGNS1, sizeof(h_signs1)), "cudaMemcpyFromSymbol signs1");
    cuda_check(cudaMemcpyFromSymbol(h_signs2, TURBO_WHT_SIGNS2, sizeof(h_signs2)), "cudaMemcpyFromSymbol signs2");

    srand(42);
    float x[128];
    for (int i = 0; i < 128; ++i) {
        x[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }

    float qx[128];
    for (int i = 0; i < 128; ++i) qx[i] = x[i];
    const float grp_norm = std::sqrt([&]() { float s = 0.0f; for (float v : x) s += v * v; return s; }());
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    for (int i = 0; i < 128; ++i) qx[i] *= inv_norm;
    fwht_128(qx, h_signs1, h_signs2);

    block_turbo3_0 blk{};
    uint8_t idx[128];
    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 32; ++j) blk.qs[j] = 0;
    for (int j = 0; j < 16; ++j) blk.signs[j] = 0;
    for (int j = 0; j < 128; ++j) {
        idx[j] = nearest_centroid_3bit(qx[j]);
        blk.qs[j / 4] |= (idx[j] & 0x3) << ((j % 4) * 2);
        if (idx[j] & 0x4) blk.signs[j / 8] |= (1u << (j % 8));
        const float c = HOST_CENTROIDS_3BIT[idx[j]];
        recon_norm_sq += c * c;
    }
    const float corrected_norm = (recon_norm_sq > 1e-20f) ? (grp_norm / std::sqrt(recon_norm_sq)) : 0.0f;
    blk.norm = __float2half(corrected_norm);

    print_idx_block(idx);

    float cpu[128];
    for (int j = 0; j < 128; ++j) cpu[j] = HOST_CENTROIDS_3BIT[idx[j]] * corrected_norm;

    block_turbo3_0 * d_blk = nullptr;
    float * d_out = nullptr;
    cuda_check(cudaMalloc(&d_blk, sizeof(block_turbo3_0)), "cudaMalloc d_blk");
    cuda_check(cudaMalloc(&d_out, 128 * sizeof(float)), "cudaMalloc d_out");
    cuda_check(cudaMemcpy(d_blk, &blk, sizeof(block_turbo3_0), cudaMemcpyHostToDevice), "cudaMemcpy blk");
    dequant_turbo3_single<<<1, 128>>>(d_blk, d_out);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    float gpu[128];
    cuda_check(cudaMemcpy(gpu, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy gpu");

    std::printf("CPU dequant first 8: ");
    for (int i = 0; i < 8; ++i) std::printf("%.6f%s", cpu[i], (i == 7) ? "\n" : " ");
    std::printf("GPU dequant first 8: ");
    for (int i = 0; i < 8; ++i) std::printf("%.6f%s", gpu[i], (i == 7) ? "\n" : " ");

    float max_abs = 0.0f;
    float rms = 0.0f;
    int first_mismatch = -1;
    for (int i = 0; i < 128; ++i) {
        const float err = std::fabs(cpu[i] - gpu[i]);
        rms += err * err;
        if (err > max_abs) max_abs = err;
        if (first_mismatch < 0 && err > 1e-4f) first_mismatch = i;
    }
    rms = std::sqrt(rms / 128.0f);

    std::printf("Max abs error: %.9g\n", max_abs);
    std::printf("RMS error: %.9g\n", rms);
    if (first_mismatch >= 0) {
        const uint8_t low2 = (blk.qs[first_mismatch / 4] >> ((first_mismatch % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk.signs[first_mismatch / 8] >> (first_mismatch % 8)) & 0x1;
        const uint8_t packed_idx = low2 | (hi1 << 2);
        std::printf("First mismatch: j=%d cpu=%.8f gpu=%.8f packed_idx=%u expected_centroid=%.6f\n",
                    first_mismatch, cpu[first_mismatch], gpu[first_mismatch], (unsigned) packed_idx,
                    HOST_CENTROIDS_3BIT[packed_idx]);
    }

    const bool pass = (max_abs < 1e-4f);
    std::printf(pass ? "PASS\n" : "FAIL\n");

    cudaFree(d_blk);
    cudaFree(d_out);
    return pass ? 0 : 1;
}
