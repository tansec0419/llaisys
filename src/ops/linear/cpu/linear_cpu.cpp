#include "linear_cpu.hpp"
#include "../../../utils.hpp"

// 朴素矩阵乘法实现: Y = X @ W^T + b
// X: [m, k], W: [n, k], Y: [m, n]
template <typename T>
void linear_(T *out, const T *input, const T *weight, const T *bias,
             size_t m, size_t n, size_t k) {

    // 遍历输出矩阵的每个元素
    for (size_t i = 0; i < m; i++) {     // 行
        for (size_t j = 0; j < n; j++) { // 列
            float sum = 0.0f;

            // 计算点积: out[i,j] = input[i,:] @ weight[j,:]
            for (size_t p = 0; p < k; p++) {
                float x_val, w_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x_val = llaisys::utils::cast<float>(input[i * k + p]);
                    w_val = llaisys::utils::cast<float>(weight[j * k + p]);
                } else {
                    x_val = static_cast<float>(input[i * k + p]);
                    w_val = static_cast<float>(weight[j * k + p]);
                }

                sum += x_val * w_val;
            }

            // 加上偏置(如果有)
            if (bias != nullptr) {
                float b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    b_val = llaisys::utils::cast<float>(bias[j]);
                } else {
                    b_val = static_cast<float>(bias[j]);
                }
                sum += b_val;
            }

            // 写回结果
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * n + j] = llaisys::utils::cast<T>(sum);
            } else {
                out[i * n + j] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *input, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t m, size_t n, size_t k) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(input),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr,
                       m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(input),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                       m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(input),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                       m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu