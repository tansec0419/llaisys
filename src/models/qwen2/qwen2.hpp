#pragma once

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"
#include "llaisys.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaisys::models {

struct Qwen2Config {
    llaisysDataType_t dtype;
    size_t num_layers;        // nlayer
    size_t hidden_size;       // hs
    size_t num_heads;         // nh
    size_t num_kv_heads;      // nkvh
    size_t head_dim;          // dh
    size_t intermediate_size; // di
    size_t max_seq_len;       // maxseq
    size_t vocab_size;        // voc
    float epsilon;
    float theta;
    int64_t eos_token_id;
};

struct Qwen2Weights {
    tensor_t embed_tokens; // input embedding
    tensor_t lm_head;      // output embedding (may share with embed_tokens)
    tensor_t norm_weight;  // final layer norm

    // Per-layer weights
    std::vector<tensor_t> input_layernorm_weight;
    std::vector<tensor_t> post_attention_layernorm_weight;

    // Attention weights
    std::vector<tensor_t> q_proj_weight;
    std::vector<tensor_t> q_proj_bias;
    std::vector<tensor_t> k_proj_weight;
    std::vector<tensor_t> k_proj_bias;
    std::vector<tensor_t> v_proj_weight;
    std::vector<tensor_t> v_proj_bias;
    std::vector<tensor_t> o_proj_weight;

    // MLP weights
    std::vector<tensor_t> gate_proj_weight;
    std::vector<tensor_t> up_proj_weight;
    std::vector<tensor_t> down_proj_weight;
};

struct KVCache {
    std::vector<tensor_t> k_cache; // [num_layers][max_seq, nkvh, head_dim]
    std::vector<tensor_t> v_cache; // [num_layers][max_seq, nkvh, head_dim]
    size_t seq_len;                // current cached sequence length
};

class Qwen2Model {
public:
    explicit Qwen2Model(const Qwen2Config &config);
    ~Qwen2Model();

    // Load weights from tensor data
    void loadWeight(const std::string &name, const void *data,
                    const std::vector<size_t> &shape, llaisysDataType_t dtype);

    // Forward pass for tokens (with KV cache)
    int64_t forward(const int64_t *token_ids, size_t num_tokens);

    // Reset KV cache for new conversation
    void resetCache();

    const Qwen2Config &config() const { return _config; }

private:
    tensor_t attention(tensor_t hidden_states, size_t layer_idx, size_t pos);
    tensor_t mlp(tensor_t hidden_states, size_t layer_idx);

    Qwen2Config _config;
    Qwen2Weights _weights;
    KVCache _kv_cache;
};

} // namespace llaisys::models