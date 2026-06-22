#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

void test_foo(ggml_context * ctx) {
    auto * turbo_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (turbo_buf) {
        ggml_backend_buffer_set_usage(turbo_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
}
