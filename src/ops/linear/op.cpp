#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, input, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), input->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, input, weight);
        CHECK_SAME_DTYPE(out->dtype(), input->dtype(), weight->dtype());
    }

    ASSERT(out->isContiguous() && input->isContiguous() && weight->isContiguous(),
           "Linear: out, input, weight must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    }

    // input: [m, k], weight: [n, k], out: [m, n]
    auto &in_shape = input->shape();
    auto &w_shape = weight->shape();
    auto &out_shape = out->shape();

    ASSERT(in_shape.size() == 2 && w_shape.size() == 2 && out_shape.size() == 2,
           "Linear: all tensors must be 2D");

    size_t m = in_shape[0];
    size_t k = in_shape[1];
    size_t n = w_shape[0];

    ASSERT(w_shape[1] == k, "Linear: weight shape mismatch");
    ASSERT(out_shape[0] == m && out_shape[1] == n, "Linear: output shape mismatch");

    if (bias) {
        ASSERT(bias->shape().size() == 1 && bias->shape()[0] == n,
               "Linear: bias shape must be [n]");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), input->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), input->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), m, n, k);
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