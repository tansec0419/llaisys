#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"
#include <cstring>

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *model;
};

#ifdef __cplusplus
extern "C" {
#endif

llaisysQwen2Model_t llaisysQwen2ModelCreate(const struct LlaisysQwen2Meta *meta) {
    llaisys::models::Qwen2Config config;
    config.dtype = meta->dtype;
    config.num_layers = meta->nlayer;
    config.hidden_size = meta->hs;
    config.num_heads = meta->nh;
    config.num_kv_heads = meta->nkvh;
    config.head_dim = meta->dh;
    config.intermediate_size = meta->di;
    config.max_seq_len = meta->maxseq;
    config.vocab_size = meta->voc;
    config.epsilon = meta->epsilon;
    config.theta = meta->theta;
    config.eos_token_id = meta->end_token;

    auto *wrapper = new LlaisysQwen2Model;
    wrapper->model = new llaisys::models::Qwen2Model(config);
    return wrapper;
}

void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model) {
    if (model) {
        delete model->model;
        delete model;
    }
}

void llaisysQwen2ModelLoadWeight(llaisysQwen2Model_t model, const char *name,
                                 const void *data, size_t *shape, size_t ndim,
                                 llaisysDataType_t dtype) {
    std::vector<size_t> shape_vec(shape, shape + ndim);
    model->model->loadWeight(name, data, shape_vec, dtype);
}

void llaisysQwen2ModelResetCache(llaisysQwen2Model_t model) {
    model->model->resetCache();
}

int64_t llaisysQwen2ModelInfer(llaisysQwen2Model_t model, int64_t *token_ids, size_t ntoken) {
    return model->model->forward(token_ids, ntoken);
}

#ifdef __cplusplus
}
#endif