#include "qwen2.hpp"
#include "../../core/context/context.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config &config) : _config(config) {
    // Initialize weight vectors
    _weights.input_layernorm_weight.resize(config.num_layers);
    _weights.post_attention_layernorm_weight.resize(config.num_layers);
    _weights.q_proj_weight.resize(config.num_layers);
    _weights.q_proj_bias.resize(config.num_layers);
    _weights.k_proj_weight.resize(config.num_layers);
    _weights.k_proj_bias.resize(config.num_layers);
    _weights.v_proj_weight.resize(config.num_layers);
    _weights.v_proj_bias.resize(config.num_layers);
    _weights.o_proj_weight.resize(config.num_layers);
    _weights.gate_proj_weight.resize(config.num_layers);
    _weights.up_proj_weight.resize(config.num_layers);
    _weights.down_proj_weight.resize(config.num_layers);

    // Initialize KV cache
    _kv_cache.k_cache.resize(config.num_layers);
    _kv_cache.v_cache.resize(config.num_layers);
    _kv_cache.seq_len = 0;

    auto device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    for (size_t i = 0; i < config.num_layers; ++i) {
        _kv_cache.k_cache[i] = Tensor::create(
            {config.max_seq_len, config.num_kv_heads, config.head_dim},
            config.dtype, device_type, device_id);
        _kv_cache.v_cache[i] = Tensor::create(
            {config.max_seq_len, config.num_kv_heads, config.head_dim},
            config.dtype, device_type, device_id);
    }
}

Qwen2Model::~Qwen2Model() = default;

void Qwen2Model::loadWeight(const std::string &name, const void *data,
                            const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    auto device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    auto tensor = Tensor::create(shape, dtype, device_type, device_id);
    tensor->load(data);

    // Parse weight name and assign to appropriate field
    if (name == "model.embed_tokens.weight") {
        _weights.embed_tokens = tensor;
    } else if (name == "model.norm.weight") {
        _weights.norm_weight = tensor;
    } else if (name == "lm_head.weight") {
        _weights.lm_head = tensor;
    } else if (name.find("model.layers.") == 0) {
        // Parse layer index
        size_t layer_start = std::string("model.layers.").length();
        size_t dot_pos = name.find('.', layer_start);
        size_t layer_idx = std::stoi(name.substr(layer_start, dot_pos - layer_start));

        std::string suffix = name.substr(dot_pos + 1);

        if (suffix == "input_layernorm.weight") {
            _weights.input_layernorm_weight[layer_idx] = tensor;
        } else if (suffix == "post_attention_layernorm.weight") {
            _weights.post_attention_layernorm_weight[layer_idx] = tensor;
        } else if (suffix == "self_attn.q_proj.weight") {
            _weights.q_proj_weight[layer_idx] = tensor;
        } else if (suffix == "self_attn.q_proj.bias") {
            _weights.q_proj_bias[layer_idx] = tensor;
        } else if (suffix == "self_attn.k_proj.weight") {
            _weights.k_proj_weight[layer_idx] = tensor;
        } else if (suffix == "self_attn.k_proj.bias") {
            _weights.k_proj_bias[layer_idx] = tensor;
        } else if (suffix == "self_attn.v_proj.weight") {
            _weights.v_proj_weight[layer_idx] = tensor;
        } else if (suffix == "self_attn.v_proj.bias") {
            _weights.v_proj_bias[layer_idx] = tensor;
        } else if (suffix == "self_attn.o_proj.weight") {
            _weights.o_proj_weight[layer_idx] = tensor;
        } else if (suffix == "mlp.gate_proj.weight") {
            _weights.gate_proj_weight[layer_idx] = tensor;
        } else if (suffix == "mlp.up_proj.weight") {
            _weights.up_proj_weight[layer_idx] = tensor;
        } else if (suffix == "mlp.down_proj.weight") {
            _weights.down_proj_weight[layer_idx] = tensor;
        }
    }
}

void Qwen2Model::resetCache() {
    _kv_cache.seq_len = 0;
}

tensor_t Qwen2Model::attention(tensor_t hidden_states, size_t layer_idx, size_t pos) {
    auto device_type = hidden_states->deviceType();
    int device_id = hidden_states->deviceId();
    size_t seq_len = hidden_states->shape()[0];

    // Linear projections: Q, K, V
    auto q_proj = Tensor::create({seq_len, _config.num_heads * _config.head_dim},
                                 _config.dtype, device_type, device_id);
    auto k_proj = Tensor::create({seq_len, _config.num_kv_heads * _config.head_dim},
                                 _config.dtype, device_type, device_id);
    auto v_proj = Tensor::create({seq_len, _config.num_kv_heads * _config.head_dim},
                                 _config.dtype, device_type, device_id);

    ops::linear(q_proj, hidden_states, _weights.q_proj_weight[layer_idx],
                _weights.q_proj_bias[layer_idx]);
    ops::linear(k_proj, hidden_states, _weights.k_proj_weight[layer_idx],
                _weights.k_proj_bias[layer_idx]);
    ops::linear(v_proj, hidden_states, _weights.v_proj_weight[layer_idx],
                _weights.v_proj_bias[layer_idx]);

    // Reshape to [seq_len, num_heads, head_dim]
    auto q = q_proj->view({seq_len, _config.num_heads, _config.head_dim});
    auto k = k_proj->view({seq_len, _config.num_kv_heads, _config.head_dim});
    auto v = v_proj->view({seq_len, _config.num_kv_heads, _config.head_dim});

    // Apply RoPE
    auto q_rope = Tensor::create({seq_len, _config.num_heads, _config.head_dim},
                                 _config.dtype, device_type, device_id);
    auto k_rope = Tensor::create({seq_len, _config.num_kv_heads, _config.head_dim},
                                 _config.dtype, device_type, device_id);

    // Create position ids
    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type, device_id);
    std::vector<int64_t> positions(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        positions[i] = static_cast<int64_t>(pos + i);
    }
    pos_ids->load(positions.data());

    ops::rope(q_rope, q, pos_ids, _config.theta);
    ops::rope(k_rope, k, pos_ids, _config.theta);

    // Update KV cache
    size_t cache_start = _kv_cache.seq_len;
    auto k_cache_slice = _kv_cache.k_cache[layer_idx]->slice(0, cache_start, cache_start + seq_len);
    auto v_cache_slice = _kv_cache.v_cache[layer_idx]->slice(0, cache_start, cache_start + seq_len);

    // Copy k_rope and v to cache
    size_t k_bytes = seq_len * _config.num_kv_heads * _config.head_dim * k_rope->elementSize();
    core::context().runtime().api()->memcpy_sync(
        k_cache_slice->data(), k_rope->data(), k_bytes, LLAISYS_MEMCPY_D2D);

    size_t v_bytes = seq_len * _config.num_kv_heads * _config.head_dim * v->elementSize();
    core::context().runtime().api()->memcpy_sync(
        v_cache_slice->data(), v->data(), v_bytes, LLAISYS_MEMCPY_D2D);

    // Get full K, V from cache
    size_t total_len = cache_start + seq_len;
    auto k_full = _kv_cache.k_cache[layer_idx]->slice(0, 0, total_len);
    auto v_full = _kv_cache.v_cache[layer_idx]->slice(0, 0, total_len);

    // Self attention
    auto attn_output = Tensor::create({seq_len, _config.num_heads, _config.head_dim},
                                      _config.dtype, device_type, device_id);
    float scale = 1.0f / std::sqrt(static_cast<float>(_config.head_dim));
    ops::self_attention(attn_output, q_rope, k_full, v_full, scale);

    // Reshape and output projection
    auto attn_flat = attn_output->view({seq_len, _config.num_heads * _config.head_dim});
    auto output = Tensor::create({seq_len, _config.hidden_size}, _config.dtype, device_type, device_id);
    ops::linear(output, attn_flat, _weights.o_proj_weight[layer_idx], nullptr);

    return output;
}

tensor_t Qwen2Model::mlp(tensor_t hidden_states, size_t layer_idx) {
    auto device_type = hidden_states->deviceType();
    int device_id = hidden_states->deviceId();
    size_t seq_len = hidden_states->shape()[0];

    auto gate = Tensor::create({seq_len, _config.intermediate_size}, _config.dtype, device_type, device_id);
    auto up = Tensor::create({seq_len, _config.intermediate_size}, _config.dtype, device_type, device_id);

    ops::linear(gate, hidden_states, _weights.gate_proj_weight[layer_idx], nullptr);
    ops::linear(up, hidden_states, _weights.up_proj_weight[layer_idx], nullptr);

    auto swiglu_out = Tensor::create({seq_len, _config.intermediate_size}, _config.dtype, device_type, device_id);
    ops::swiglu(swiglu_out, gate, up);

    auto output = Tensor::create({seq_len, _config.hidden_size}, _config.dtype, device_type, device_id);
    ops::linear(output, swiglu_out, _weights.down_proj_weight[layer_idx], nullptr);

    return output;
}

int64_t Qwen2Model::forward(const int64_t *token_ids, size_t num_tokens) {
    auto device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    size_t pos = _kv_cache.seq_len;

    // Embedding lookup
    auto input_ids = Tensor::create({num_tokens}, LLAISYS_DTYPE_I64, device_type, device_id);
    input_ids->load(token_ids);

    auto hidden_states = Tensor::create({num_tokens, _config.hidden_size}, _config.dtype, device_type, device_id);
    ops::embedding(hidden_states, input_ids, _weights.embed_tokens);

    // Process each layer
    for (size_t layer = 0; layer < _config.num_layers; ++layer) {
        // Input layernorm
        auto normed = Tensor::create({num_tokens, _config.hidden_size}, _config.dtype, device_type, device_id);
        ops::rms_norm(normed, hidden_states, _weights.input_layernorm_weight[layer], _config.epsilon);

        // Attention
        auto attn_out = attention(normed, layer, pos);

        // Residual connection
        ops::add(hidden_states, hidden_states, attn_out);

        // Post attention layernorm
        ops::rms_norm(normed, hidden_states, _weights.post_attention_layernorm_weight[layer], _config.epsilon);

        // MLP
        auto mlp_out = mlp(normed, layer);

        // Residual connection
        ops::add(hidden_states, hidden_states, mlp_out);
    }

    // Update cache length
    _kv_cache.seq_len += num_tokens;

    // Final layer norm
    auto final_normed = Tensor::create({num_tokens, _config.hidden_size}, _config.dtype, device_type, device_id);
    ops::rms_norm(final_normed, hidden_states, _weights.norm_weight, _config.epsilon);

    // Get last token's hidden state
    auto last_hidden = final_normed->slice(0, num_tokens - 1, num_tokens);

    // LM head
    auto logits = Tensor::create({1, _config.vocab_size}, _config.dtype, device_type, device_id);

    // Check if lm_head is loaded, if not, use embed_tokens (weight tying)
    tensor_t lm_head_weight = _weights.lm_head ? _weights.lm_head : _weights.embed_tokens;
    ops::linear(logits, last_hidden, lm_head_weight, nullptr);

    // Argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
    auto max_val = Tensor::create({1}, _config.dtype, device_type, device_id);
    auto logits_1d = logits->view({_config.vocab_size});
    ops::argmax(max_idx, max_val, logits_1d);

    int64_t result;
    core::context().runtime().api()->memcpy_sync(&result, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    return result;
}

} // namespace llaisys::models