#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *input, const int64_t *pos,
           size_t seq_len, size_t num_heads, size_t head_dim, float theta) {

    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        int64_t position = pos[s];
        float pos_float = static_cast<float>(position);

        for (size_t h = 0; h < num_heads; h++) {
            for (size_t d = 0; d < half_dim; d++) {
                // 计算频率: freq = pos / (theta^(2*d / head_dim))
                float exponent = (2.0f * d) / static_cast<float>(head_dim);
                float freq = pos_float / std::pow(theta, exponent);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                // 索引计算: [seq_len, num_heads, head_dim]
                size_t base_idx = s * num_heads * head_dim + h * head_dim;

                size_t idx_a = base_idx + d;
                size_t idx_b = base_idx + half_dim + d;

                float x_a, x_b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x_a = llaisys::utils::cast<float>(input[idx_a]);
                    x_b = llaisys::utils::cast<float>(input[idx_b]);
                } else {
                    x_a = static_cast<float>(input[idx_a]);
                    x_b = static_cast<float>(input[idx_b]);
                }

                // 旋转变换: 注意这里的公式!
                float y_a = x_a * cos_val - x_b * sin_val;
                float y_b = x_b * cos_val + x_a * sin_val; // 注意顺序!

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(y_a);
                    out[idx_b] = llaisys::utils::cast<T>(y_b);
                } else {
                    out[idx_a] = static_cast<T>(y_a);
                    out[idx_b] = static_cast<T>(y_b);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *input, const std::byte *pos,
          llaisysDataType_t type, size_t seq_len, size_t num_heads,
          size_t head_dim, float theta) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(input),
                     pos_ptr, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(input),
                     pos_ptr, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(input),
                     pos_ptr, seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu