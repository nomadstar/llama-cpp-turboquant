import os
import re

def patch_mma():
    file_path = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # 1. Update flash_attn_ext_f16_load_tile signature
    content = content.replace(
        """static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup) {""",
        """static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV_base, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup,
        const char * block_table, const int k_VKQ_0, const int block_size, const int sequence, const int ne11) {"""
    )

    # 2. Update cp_async loop
    cp_async_old = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                }"""
    cp_async_new = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);
                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV_base + physical_i*stride_KV + k*h2_per_chunk);
                }"""
    content = content.replace(cp_async_old, cp_async_new)

    # 3. Update non-cp_async loop
    memcpy_old = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4,
                        !oob_check || i < i_sup ? KV + i*stride_KV + k*h2_per_chunk : zero);
                }"""
    memcpy_new = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);
                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4,
                        !oob_check || i < i_sup ? KV_base + physical_i*stride_KV + k*h2_per_chunk : zero);
                }"""
    content = content.replace(memcpy_old, memcpy_new)

    # 4. Update flash_attn_ext_f16_iter signature
    iter_old = """        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup) {"""
    iter_new = """        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const char * block_table,
        const int block_size,
        const int sequence,
        const int ne11) {"""
    content = content.replace(iter_old, iter_new)

    # 5. Update flash_attn_ext_f16_load_tile calls inside iter
    content = content.replace(
        """(V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);""",
        """(V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_sup, block_table, k_VKQ_0, block_size, sequence, ne11);"""
    )
    content = content.replace(
        """(K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, k0_diff, stride_K, k_VKQ_sup);""",
        """(K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_sup, block_table, k_VKQ_0, block_size, sequence, ne11);"""
    )
    content = content.replace(
        """(K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);""",
        """(K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_sup, block_table, k_VKQ_0 + nbatch_fa, block_size, sequence, ne11);"""
    )
    content = content.replace(
        """(V_h2 + int64_t(k_VKQ_0_next)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup_next);""",
        """(V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup_next, block_table, k_VKQ_0_next, block_size, sequence, ne11);"""
    )
    content = content.replace(
        """(K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);""",
        """(K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_sup, block_table, kb0*nbatch_fa, block_size, sequence, ne11);"""
    )

    # 6. Update flash_attn_ext_f16_iter calls
    content = content.replace(
        """Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_max - kb0 * nbatch_fa);""",
        """Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_max - kb0 * nbatch_fa, block_table, block_size, sequence, ne01.x);"""
    )

    # Remove GGML_UNUSED(block_table)
    content = content.replace("GGML_UNUSED(block_table);\n", "")

    with open(file_path, "w") as f:
        f.write(content)

def patch_wmma():
    file_path = "ggml/src/ggml-cuda/fattn-wmma-f16.cu"
    with open(file_path, "r") as f:
        content = f.read()

    # Remove GGML_UNUSED(block_table)
    content = content.replace("GGML_UNUSED(block_table);\n", "")

    # K load
    k_load_old = "wmma::load_matrix_sync(K_a, K_h_f16 + int64_t(k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);"
    k_load_new = """int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y, block_size, sequence, ne11);
                wmma::load_matrix_sync(K_a, K_h_f16 + physical_i*stride_KV + k_KQ_0, stride_KV);"""
    content = content.replace(k_load_old, k_load_new)

    # V load
    v_load_old = "wmma::load_matrix_sync(v_a, V_h_f16 + int64_t(k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);"
    v_load_new = """int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                wmma::load_matrix_sync(v_a, V_h_f16 + physical_i*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);"""
    content = content.replace(v_load_old, v_load_new)

    with open(file_path, "w") as f:
        f.write(content)

def patch_vec():
    file_path = "ggml/src/ggml-cuda/fattn-vec.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # Remove GGML_UNUSED(block_table)
    content = content.replace("GGML_UNUSED(block_table);\n", "")

    # K load
    k_load_old = "float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);"
    k_load_new = """int64_t physical_k = get_physical_token_idx(block_table, k_VKQ_0 + i_KQ, block_size, sequence, ne11);
                float sum = vec_dot_KQ(K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + physical_k*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);"""
    content = content.replace(k_load_old, k_load_new)

    # V load 1
    v_load_old1 = "dequantize_V(V + k*nb21, tmp_f,"
    v_load_new1 = """int64_t physical_v = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                    dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + physical_v*nb21, tmp_f,"""
    content = content.replace(v_load_old1, v_load_new1)

    # V load 2
    v_load_old2 = "dequantize_V(V + k*nb21, tmp,"
    v_load_new2 = """int64_t physical_v = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                    dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + physical_v*nb21, tmp,"""
    content = content.replace(v_load_old2, v_load_new2)

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_mma()
    patch_wmma()
    patch_vec()
    print("Done")
