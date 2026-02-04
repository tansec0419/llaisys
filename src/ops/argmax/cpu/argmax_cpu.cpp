#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    T max_value = vals[0];
    int64_t max_index = 0;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float max_float = llaisys::utils::cast<float>(max_value);
        for (size_t i = 1; i < size; i++) {
            float current = llaisys::utils::cast<float>(vals[i]);
            if (current > max_float) {
                max_float = current;
                max_value = vals[i];
                max_index = i;
            }
        }
    } else {
        for (size_t i = 1; i < size; i++) {
            if (vals[i] > max_value) {
                max_value = vals[i];
                max_index = i;
            }
        }
    }

    max_idx[0] = max_index;
    max_val[0] = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t size) {
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), size);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu