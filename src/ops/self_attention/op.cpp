#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(out, q, k, v);
    CHECK_SAME_DTYPE(out->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(out->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-Attention: all tensors must be contiguous");

    // shape: [seq_len, num_heads, head_dim]
    auto &q_shape = q->shape();
    auto &k_shape = k->shape();
    auto &v_shape = v->shape();
    auto &out_shape = out->shape();

    ASSERT(q_shape.size() == 3 && k_shape.size() == 3 && v_shape.size() == 3,
           "Self-Attention: all inputs must be 3D tensor");

    size_t q_len = q_shape[0];
    size_t kv_len = k_shape[0];
    size_t num_heads = q_shape[1];
    size_t num_kv_heads = k_shape[1];
    size_t head_dim = q_shape[2];

    ASSERT(k_shape[0] == v_shape[0] && k_shape[1] == v_shape[1] && k_shape[2] == v_shape[2],
           "Self-Attention: k and v must have same shape");
    ASSERT(k_shape[2] == head_dim && v_shape[2] == head_dim,
           "Self-Attention: head_dim must match");
    ASSERT(out_shape[0] == q_len && out_shape[1] == num_heads && out_shape[2] == head_dim,
           "Self-Attention: output shape mismatch");
    ASSERT(num_heads % num_kv_heads == 0,
           "Self-Attention: num_heads must be divisible by num_kv_heads");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(out->data(), q->data(), k->data(), v->data(),
                                   out->dtype(), q_len, kv_len, num_heads,
                                   num_kv_heads, head_dim, scale);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(out->data(), q->data(), k->data(), v->data(),
                                   out->dtype(), q_len, kv_len, num_heads,
                                   num_kv_heads, head_dim, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops