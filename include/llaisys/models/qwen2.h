#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct LlaisysQwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

typedef struct LlaisysQwen2Model *llaisysQwen2Model_t;

__export llaisysQwen2Model_t llaisysQwen2ModelCreate(const struct LlaisysQwen2Meta *meta);
__export void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model);
__export void llaisysQwen2ModelLoadWeight(llaisysQwen2Model_t model, const char *name,
                                          const void *data, size_t *shape, size_t ndim,
                                          llaisysDataType_t dtype);
__export void llaisysQwen2ModelResetCache(llaisysQwen2Model_t model);
__export int64_t llaisysQwen2ModelInfer(llaisysQwen2Model_t model, int64_t *token_ids, size_t ntoken);

#ifdef __cplusplus
}
#endif

#endif // LLAISYS_MODELS_QWEN2_H