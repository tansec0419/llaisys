#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *input, const T *weight, float eps,
               size_t batch_size, size_t hidden_size) {
    for (size_t i = 0; i < batch_size; i++) {
        const T *input_row = input + i * hidden_size;
        T *out_row = out + i * hidden_size;

        // 计算均方根
        float sum_squares = 0.0f;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < hidden_size; j++) {
                float val = llaisys::utils::cast<float>(input_row[j]);
                sum_squares += val * val;
            }
        } else {
            for (size_t j = 0; j < hidden_size; j++) {
                float val = static_cast<float>(input_row[j]);
                sum_squares += val * val;
            }
        }

        float rms = std::sqrt(sum_squares / hidden_size + eps);

        // 归一化并应用权重
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < hidden_size; j++) {
                float val = llaisys::utils::cast<float>(input_row[j]);
                float w = llaisys::utils::cast<float>(weight[j]);
                float result = (val / rms) * w;
                out_row[j] = llaisys::utils::cast<T>(result);
            }
        } else {
            for (size_t j = 0; j < hidden_size; j++) {
                float val = static_cast<float>(input_row[j]);
                float w = static_cast<float>(weight[j]);
                out_row[j] = static_cast<T>((val / rms) * w);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight,
              llaisysDataType_t type, float eps,
              size_t batch_size, size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(input),
                         reinterpret_cast<const float *>(weight),
                         eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(input),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(input),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         eps, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu