#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t input, tensor_t pos, float theta) {
    CHECK_SAME_DEVICE(out, input, pos);
    CHECK_SAME_DTYPE(out->dtype(), input->dtype());
    CHECK_SAME_SHAPE(out->shape(), input->shape());
    ASSERT(pos->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos must be int64");
    ASSERT(out->isContiguous() && input->isContiguous() && pos->isContiguous(),
           "RoPE: all tensors must be contiguous");

    auto &shape = input->shape();
    ASSERT(shape.size() == 3, "RoPE: input must be 3D tensor [seq_len, num_heads, head_dim]");

    size_t seq_len = shape[0];
    size_t num_heads = shape[1];
    size_t head_dim = shape[2];

    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even");
    ASSERT(pos->numel() == seq_len, "RoPE: pos size must match seq_len");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), input->data(), pos->data(),
                         out->dtype(), seq_len, num_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), input->data(), pos->data(),
                         out->dtype(), seq_len, num_heads, head_dim, theta);
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