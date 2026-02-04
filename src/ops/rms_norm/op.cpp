#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t input, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, input, weight);
    CHECK_SAME_DTYPE(out->dtype(), input->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && input->isContiguous() && weight->isContiguous(),
           "RMS Norm: all tensors must be contiguous");

    // input 和 out 形状应该相同
    CHECK_SAME_SHAPE(out->shape(), input->shape());

    // weight 应该是 1D 张量，长度等于 hidden_size
    ASSERT(weight->shape().size() == 1, "RMS Norm: weight must be 1D tensor");

    size_t hidden_size = weight->shape()[0];
    size_t batch_size = input->numel() / hidden_size;

    ASSERT(input->numel() % hidden_size == 0,
           "RMS Norm: input size must be divisible by hidden_size");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), input->data(), weight->data(),
                             out->dtype(), eps, batch_size, hidden_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), input->data(), weight->data(),
                             out->dtype(), eps, batch_size, hidden_size);
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