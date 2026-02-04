#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t q_len, size_t kv_len, size_t num_heads,
                     size_t num_kv_heads, size_t head_dim, float scale) {

    size_t head_ratio = num_heads / num_kv_heads; // GQA 比例

    // 为每个 query head 计算 attention
    for (size_t h = 0; h < num_heads; h++) {
        size_t kv_h = h / head_ratio; // 对应的 kv head

        // 计算注意力分数矩阵 [q_len, kv_len]
        std::vector<float> scores(q_len * kv_len);

        for (size_t i = 0; i < q_len; i++) {
            for (size_t j = 0; j < kv_len; j++) {
                float score = 0.0f;

                // 计算 q[i, h] @ k[j, kv_h]^T
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = i * num_heads * head_dim + h * head_dim + d;
                    size_t k_idx = j * num_kv_heads * head_dim + kv_h * head_dim + d;

                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = static_cast<float>(q[q_idx]);
                        k_val = static_cast<float>(k[k_idx]);
                    }

                    score += q_val * k_val;
                }

                // 缩放
                score *= scale;

                // Causal mask: 当前 query 位置只能看到 <= 当前位置的 key
                // 计算实际的 query 和 key 位置
                size_t q_pos = kv_len - q_len + i; // query 在整个序列中的绝对位置
                if (j > q_pos) {
                    score = -std::numeric_limits<float>::infinity();
                }

                scores[i * kv_len + j] = score;
            }

            // Softmax 归一化当前行
            float max_score = -std::numeric_limits<float>::infinity();
            size_t q_pos = kv_len - q_len + i;
            for (size_t j = 0; j <= q_pos && j < kv_len; j++) {
                max_score = std::max(max_score, scores[i * kv_len + j]);
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j <= q_pos && j < kv_len; j++) {
                float exp_val = std::exp(scores[i * kv_len + j] - max_score);
                scores[i * kv_len + j] = exp_val;
                sum_exp += exp_val;
            }

            for (size_t j = 0; j <= q_pos && j < kv_len; j++) {
                scores[i * kv_len + j] /= sum_exp;
            }

            for (size_t j = q_pos + 1; j < kv_len; j++) {
                scores[i * kv_len + j] = 0.0f;
            }
        }

        // 计算输出: out = scores @ V
        for (size_t i = 0; i < q_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;

                for (size_t j = 0; j < kv_len; j++) {
                    size_t v_idx = j * num_kv_heads * head_dim + kv_h * head_dim + d;

                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = static_cast<float>(v[v_idx]);
                    }

                    sum += scores[i * kv_len + j] * v_val;
                }

                size_t out_idx = i * num_heads * head_dim + h * head_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[out_idx] = llaisys::utils::cast<T>(sum);
                } else {
                    out[out_idx] = static_cast<T>(sum);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t q_len, size_t kv_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, float scale) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               q_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               q_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               q_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu