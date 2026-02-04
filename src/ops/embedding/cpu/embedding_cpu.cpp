#include "embedding_cpu.hpp"
#include "../../../utils.hpp" // 添加这一行!
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_size, size_t hidden_size) {
    size_t elem_size = 0;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        elem_size = 4;
        break;
    case LLAISYS_DTYPE_F16:
        elem_size = 2;
        break;
    case LLAISYS_DTYPE_BF16:
        elem_size = 2;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    size_t row_bytes = hidden_size * elem_size;

    for (size_t i = 0; i < index_size; i++) {
        int64_t idx = index[i];
        const std::byte *src = weight + idx * row_bytes;
        std::byte *dst = out + i * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
}
} // namespace llaisys::ops::cpu