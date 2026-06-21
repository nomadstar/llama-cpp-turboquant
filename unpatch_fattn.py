import os

def unpatch_mma():
    file_path = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    cp_async_old = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);
                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV_base + physical_i*stride_KV + k*h2_per_chunk);
                }"""
    cp_async_new = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV_base + (k_VKQ_0 + i)*stride_KV + k*h2_per_chunk);
                }"""
    content = content.replace(cp_async_old, cp_async_new)

    memcpy_old = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);
                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4,
                        !oob_check || i < i_sup ? KV_base + physical_i*stride_KV + k*h2_per_chunk : zero);
                }"""
    memcpy_new = """#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);
                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4,
                        !oob_check || i < i_sup ? KV_base + (k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero);
                }"""
    content = content.replace(memcpy_old, memcpy_new)

    with open(file_path, "w") as f:
        f.write(content)

def unpatch_wmma():
    file_path = "ggml/src/ggml-cuda/fattn-wmma-f16.cu"
    with open(file_path, "r") as f:
        content = f.read()

    k_load_old = """int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y, block_size, sequence, ne11);
                wmma::load_matrix_sync(K_a, K_h_f16 + physical_i*stride_KV + k_KQ_0, stride_KV);"""
    k_load_new = """wmma::load_matrix_sync(K_a, K_h_f16 + int64_t(k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);"""
    content = content.replace(k_load_old, k_load_new)

    v_load_old = """int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                wmma::load_matrix_sync(v_a, V_h_f16 + physical_i*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);"""
    v_load_new = """wmma::load_matrix_sync(v_a, V_h_f16 + int64_t(k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);"""
    content = content.replace(v_load_old, v_load_new)

    with open(file_path, "w") as f:
        f.write(content)

def unpatch_vec():
    file_path = "ggml/src/ggml-cuda/fattn-vec.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    k_load_old = """int64_t physical_k = get_physical_token_idx(block_table, k_VKQ_0 + i_KQ, block_size, sequence, ne11);
                float sum = vec_dot_KQ(K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + physical_k*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);"""
    k_load_new = """float sum = vec_dot_KQ(K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + (k_VKQ_0 + i_KQ)*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);"""
    content = content.replace(k_load_old, k_load_new)

    v_load_old1 = """int64_t physical_v = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                    dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + physical_v*nb21, tmp_f,"""
    v_load_new1 = """dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + (k_VKQ_0 + k)*nb21, tmp_f,"""
    content = content.replace(v_load_old1, v_load_new1)

    v_load_old2 = """int64_t physical_v = get_physical_token_idx(block_table, k_VKQ_0 + k, block_size, sequence, ne11);
                    dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + physical_v*nb21, tmp,"""
    v_load_new2 = """dequantize_V(V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + (k_VKQ_0 + k)*nb21, tmp,"""
    content = content.replace(v_load_old2, v_load_new2)

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    unpatch_mma()
    unpatch_wmma()
    unpatch_vec()
    print("Done")
