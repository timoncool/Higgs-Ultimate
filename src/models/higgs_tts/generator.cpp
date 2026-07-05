#include "engine/models/higgs_tts/generator.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

using Clock = std::chrono::steady_clock;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TextLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct HiggsTTSGeneratorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue modality_embedding;
    std::vector<TextLayerWeights> layers;
    core::TensorValue norm;
};

struct DecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

struct SamplingCandidate {
    int32_t token = 0;
    float score = 0.0F;
};

struct SamplingScratch {
    std::vector<SamplingCandidate> candidates;
    std::vector<double> probabilities;
};

struct HiggsSamplerState {
    int64_t num_codebooks = 0;
    int64_t delay_count = 0;
    std::optional<int64_t> eoc_countdown;
    bool generation_done = false;
    std::vector<int32_t> last_codes;
};

struct PromptReferenceInfo {
    int64_t start = 0;
    int64_t tokens = 0;
};

int64_t head_dim(const HiggsTTSConfig & config) {
    if (config.text.num_attention_heads <= 0 || config.text.num_key_value_heads <= 0 || config.text.head_dim <= 0) {
        throw std::runtime_error("Higgs TTS attention config is invalid");
    }
    return config.text.head_dim;
}

int64_t modality_vocab_size(const HiggsTTSConfig & config) {
    return config.audio_encoder.num_codebooks * config.audio_encoder.vocab_size;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k_contiguous = core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v_contiguous = core::ensure_backend_addressable_layout(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_contiguous.tensor,
        k_contiguous.tensor,
        v_contiguous.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

DecoderLayerOutputs decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const HiggsTTSConfig & config,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.text.num_attention_heads / config.text.num_key_value_heads;
    const modules::LinearModule q_proj({config.text.hidden_size, config.text.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.text.hidden_size, config.text.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.text.hidden_size, config.text.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.text.num_attention_heads * dim, config.text.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.text.hidden_size, config.text.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.text.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.text.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.text.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.text.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.text.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.text.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.text.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.text.hidden_size, config.text.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.text.hidden_size, config.text.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.text.intermediate_size, config.text.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

DecoderLayerOutputs decoder_layer_with_static_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const HiggsTTSConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t dim = head_dim(config);
    const modules::LinearModule q_proj({config.text.hidden_size, config.text.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.text.hidden_size, config.text.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.text.hidden_size, config.text.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.text.num_attention_heads * dim, config.text.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.text.hidden_size, config.text.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = modules::RMSNormModule({dim, config.text.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, q, config.text.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({dim, config.text.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, k, config.text.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.text.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.text.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.text.rope_theta}).build(ctx, k, positions);

    const modules::FastKVSetRowsModule set_rows;
    auto updated_cache_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_cache_value = set_rows.build(ctx, cache_value, v, cache_slot);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_key.shape.rank}).build(ctx, updated_cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_value.shape.rank}).build(ctx, updated_cache_value);
    auto context = flash_attention_from_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, 1, config.text.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.text.hidden_size, config.text.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.text.hidden_size, config.text.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.text.intermediate_size, config.text.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

core::TensorValue text_prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config,
    ggml_tensor * token_ids,
    int64_t prompt_steps) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({config.text.vocab_size, config.text.hidden_size}).build(ctx, ids, weights.text_embedding);
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.text.hidden_size}));
}

core::TensorValue text_prompt_slice_embeddings(
    core::ModuleBuildContext & ctx,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config,
    const core::TensorValue & token_ids,
    int64_t start,
    int64_t steps) {
    auto ids = modules::SliceModule({0, start, steps}).build(ctx, token_ids);
    auto x = modules::EmbeddingModule({config.text.vocab_size, config.text.hidden_size}).build(ctx, ids, weights.text_embedding);
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, steps, config.text.hidden_size}));
}

core::TensorValue reference_prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config,
    const core::TensorValue & reference_code_ids,
    int64_t reference_tokens) {
    auto embeddings = modules::EmbeddingModule({modality_vocab_size(config), config.text.hidden_size})
                          .build(ctx, reference_code_ids, weights.modality_embedding);
    auto summed = modules::ReduceSumModule({1}).build(ctx, embeddings);
    return core::reshape_tensor(ctx, summed, core::TensorShape::from_dims({1, reference_tokens, config.text.hidden_size}));
}

core::TensorValue prompt_embeddings_with_optional_reference(
    core::ModuleBuildContext & ctx,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * reference_code_ids,
    int64_t prompt_steps,
    const PromptReferenceInfo & reference) {
    if (reference.tokens == 0) {
        return text_prompt_embeddings(ctx, weights, config, token_ids, prompt_steps);
    }
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto prefix = text_prompt_slice_embeddings(ctx, weights, config, ids, 0, reference.start);
    auto reference_ids = core::wrap_tensor(
        reference_code_ids,
        core::TensorShape::from_dims({reference.tokens, config.audio_encoder.num_codebooks}),
        GGML_TYPE_I32);
    auto reference_embed = reference_prompt_embeddings(ctx, weights, config, reference_ids, reference.tokens);
    auto x = modules::ConcatModule({1}).build(ctx, prefix, reference_embed);
    const int64_t suffix_start = reference.start + reference.tokens;
    const int64_t suffix_steps = prompt_steps - suffix_start;
    if (suffix_steps > 0) {
        auto suffix = text_prompt_slice_embeddings(ctx, weights, config, ids, suffix_start, suffix_steps);
        x = modules::ConcatModule({1}).build(ctx, x, suffix);
    }
    return x;
}

core::TensorValue modality_code_embedding(
    core::ModuleBuildContext & ctx,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config,
    ggml_tensor * offset_code_ids) {
    auto ids = core::wrap_tensor(
        offset_code_ids,
        core::TensorShape::from_dims({config.audio_encoder.num_codebooks}),
        GGML_TYPE_I32);
    auto embeddings = modules::EmbeddingModule({modality_vocab_size(config), config.text.hidden_size})
                          .build(ctx, ids, weights.modality_embedding);
    auto summed = modules::ReduceSumModule({0}).build(ctx, embeddings);
    return core::reshape_tensor(ctx, summed, core::TensorShape::from_dims({1, 1, config.text.hidden_size}));
}

core::TensorValue modality_logits(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const HiggsTTSGeneratorWeights & weights,
    const HiggsTTSConfig & config) {
    return modules::LinearModule({config.text.hidden_size, modality_vocab_size(config), false})
        .build(ctx, hidden, {weights.modality_embedding, std::nullopt});
}

HiggsTTSGeneratorWeights load_weights(
    const HiggsTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    HiggsTTSGeneratorWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "higgs_tts.generator.weights",
        weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "tied.embedding.text_embedding.weight",
        storage_type,
        {config.text.vocab_size, config.text.hidden_size});
    weights.modality_embedding = weights.store->load_tensor(
        source,
        "tied.embedding.modality_embeddings.0.embedding.weight",
        storage_type,
        {modality_vocab_size(config), config.text.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.text.num_hidden_layers));
    const int64_t dim = head_dim(config);
    for (int64_t layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        const std::string prefix = "body.layers." + std::to_string(layer);
        TextLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.text.hidden_size});
        w.q_proj = weights.store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.text.num_attention_heads * dim, config.text.hidden_size});
        w.k_proj = weights.store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.text.num_key_value_heads * dim, config.text.hidden_size});
        w.v_proj = weights.store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.text.num_key_value_heads * dim, config.text.hidden_size});
        w.o_proj = weights.store->load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.text.hidden_size, config.text.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.text.hidden_size});
        w.gate_proj = weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.text.intermediate_size, config.text.hidden_size});
        w.up_proj = weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.text.intermediate_size, config.text.hidden_size});
        w.down_proj = weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.text.hidden_size, config.text.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "body.norm.weight", {config.text.hidden_size});
    weights.store->upload();
    return weights;
}

PromptReferenceInfo resolve_prompt_reference(
    const HiggsTTSPrompt & prompt,
    const HiggsAudioCodeMatrix * reference_delayed_codes,
    const HiggsTTSConfig & config) {
    PromptReferenceInfo info;
    int64_t placeholders = 0;
    int64_t first_placeholder = -1;
    int64_t last_placeholder = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(prompt.input_ids.size()); ++i) {
        if (prompt.input_ids[static_cast<size_t>(i)] == kHiggsAudioPlaceholderId) {
            if (first_placeholder < 0) {
                first_placeholder = i;
            }
            last_placeholder = i;
            ++placeholders;
        }
    }
    if (placeholders == 0) {
        if (reference_delayed_codes != nullptr && reference_delayed_codes->frames > 0) {
            throw std::runtime_error("Higgs TTS reference audio codes were provided, but the prompt has no reference slots");
        }
        return info;
    }
    if (last_placeholder - first_placeholder + 1 != placeholders) {
        throw std::runtime_error("Higgs TTS reference placeholders must be contiguous");
    }
    if (reference_delayed_codes == nullptr) {
        throw std::runtime_error("Higgs TTS reference prompt requires delayed reference audio codes");
    }
    if (reference_delayed_codes->frames != placeholders) {
        throw std::runtime_error("Higgs TTS reference code count does not match the prompt placeholder count");
    }
    if (reference_delayed_codes->codebooks != config.audio_encoder.num_codebooks) {
        throw std::runtime_error("Higgs TTS reference codebook count mismatch");
    }
    if (static_cast<int64_t>(reference_delayed_codes->token_ids.size()) !=
        reference_delayed_codes->frames * reference_delayed_codes->codebooks) {
        throw std::runtime_error("Higgs TTS reference code matrix shape is invalid");
    }
    for (const int32_t code : reference_delayed_codes->token_ids) {
        if (code < 0 || static_cast<int64_t>(code) >= config.audio_encoder.vocab_size) {
            throw std::runtime_error("Higgs TTS reference code is outside the modality vocabulary");
        }
    }
    info.start = first_placeholder;
    info.tokens = placeholders;
    return info;
}

std::vector<int32_t> offset_modality_code_matrix(
    const HiggsTTSConfig & config,
    const HiggsAudioCodeMatrix & codes) {
    if (codes.codebooks != config.audio_encoder.num_codebooks) {
        throw std::runtime_error("Higgs TTS reference codebook count mismatch");
    }
    if (static_cast<int64_t>(codes.token_ids.size()) != codes.frames * codes.codebooks) {
        throw std::runtime_error("Higgs TTS reference code matrix shape is invalid");
    }
    std::vector<int32_t> out(codes.token_ids.size(), 0);
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        for (int64_t codebook = 0; codebook < codes.codebooks; ++codebook) {
            const size_t index = static_cast<size_t>(frame * codes.codebooks + codebook);
            const int32_t code = codes.token_ids[index];
            if (code < 0 || static_cast<int64_t>(code) >= config.audio_encoder.vocab_size) {
                throw std::runtime_error("Higgs TTS reference code is outside the modality vocabulary");
            }
            out[index] = static_cast<int32_t>(codebook * config.audio_encoder.vocab_size + static_cast<int64_t>(code));
        }
    }
    return out;
}

int32_t argmax_row(const std::vector<float> & logits, int64_t row, int64_t vocab_size) {
    const int64_t offset = row * vocab_size;
    int64_t best = 0;
    for (int64_t token = 1; token < vocab_size; ++token) {
        if (logits[static_cast<size_t>(offset + token)] > logits[static_cast<size_t>(offset + best)]) {
            best = token;
        }
    }
    return static_cast<int32_t>(best);
}

int32_t sample_row(
    const std::vector<float> & logits,
    int64_t row,
    int64_t vocab_size,
    const HiggsTTSGenerationOptions & options,
    std::mt19937 & rng,
    SamplingScratch & scratch) {
    if (!options.do_sample || options.top_k == 1 || options.temperature <= 1.0e-5F) {
        return argmax_row(logits, row, vocab_size);
    }
    if (!(options.temperature > 0.0F)) {
        throw std::runtime_error("Higgs TTS temperature must be positive");
    }
    if (!(options.top_p >= 0.0F && options.top_p <= 1.0F)) {
        throw std::runtime_error("Higgs TTS top_p must be in [0, 1]");
    }
    auto & candidates = scratch.candidates;
    candidates.clear();
    candidates.reserve(static_cast<size_t>(vocab_size));
    const int64_t offset = row * vocab_size;
    for (int64_t token = 0; token < vocab_size; ++token) {
        candidates.push_back({
            static_cast<int32_t>(token),
            logits[static_cast<size_t>(offset + token)] / options.temperature,
        });
    }
    auto candidate_order = [](const SamplingCandidate & lhs, const SamplingCandidate & rhs) {
        if (lhs.score == rhs.score) {
            return lhs.token < rhs.token;
        }
        return lhs.score > rhs.score;
    };
    if (options.top_k > 0 && static_cast<size_t>(options.top_k) < candidates.size()) {
        const auto top_end = candidates.begin() + static_cast<std::ptrdiff_t>(options.top_k);
        std::nth_element(candidates.begin(), top_end, candidates.end(), candidate_order);
        std::sort(candidates.begin(), top_end, candidate_order);
        candidates.erase(top_end, candidates.end());
    } else {
        std::sort(candidates.begin(), candidates.end(), candidate_order);
    }
    const float max_score = candidates.front().score;
    double total = 0.0;
    auto & probabilities = scratch.probabilities;
    probabilities.assign(candidates.size(), 0.0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(static_cast<double>(candidates[i].score - max_score));
        total += probabilities[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("Higgs TTS sampler produced invalid probability mass");
    }
    for (double & probability : probabilities) {
        probability /= total;
    }
    if (options.top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = probabilities.size();
        for (size_t i = 0; i < probabilities.size(); ++i) {
            cumulative += probabilities[i];
            if (cumulative >= static_cast<double>(options.top_p)) {
                keep = i + 1;
                break;
            }
        }
        candidates.resize(keep);
        probabilities.resize(keep);
        double kept_total = 0.0;
        for (const double probability : probabilities) {
            kept_total += probability;
        }
        if (!(kept_total > 0.0)) {
            throw std::runtime_error("Higgs TTS top-p removed all probability mass");
        }
        for (double & probability : probabilities) {
            probability /= kept_total;
        }
    }
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return candidates[distribution(rng)].token;
}

std::vector<int32_t> sample_higgs_step(
    const std::vector<float> & logits,
    HiggsSamplerState & state,
    const HiggsTTSConfig & config,
    const HiggsTTSGenerationOptions & options,
    std::mt19937 & rng,
    SamplingScratch & scratch) {
    const int64_t codebooks = config.audio_encoder.num_codebooks;
    const int64_t vocab_size = config.audio_encoder.vocab_size;
    if (state.generation_done) {
        return std::vector<int32_t>(static_cast<size_t>(codebooks), kHiggsAudioStopCode);
    }
    if (static_cast<int64_t>(logits.size()) != modality_vocab_size(config)) {
        throw std::runtime_error("Higgs TTS modality logits shape mismatch");
    }
    std::vector<int32_t> codes(static_cast<size_t>(codebooks), 0);
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        codes[static_cast<size_t>(codebook)] = sample_row(logits, codebook, vocab_size, options, rng, scratch);
    }
    if (state.delay_count < codebooks) {
        const int64_t next_codebook = state.delay_count + 1;
        if (next_codebook < codebooks) {
            for (int64_t codebook = next_codebook; codebook < codebooks; ++codebook) {
                codes[static_cast<size_t>(codebook)] = kHiggsAudioBocId;
            }
        }
        ++state.delay_count;
    } else if (state.eoc_countdown.has_value()) {
        --(*state.eoc_countdown);
        if (*state.eoc_countdown <= 0) {
            state.generation_done = true;
        }
    } else if (codes.front() == kHiggsAudioEocId) {
        state.eoc_countdown = codebooks > 2 ? codebooks - 2 : 0;
        if (codebooks <= 2) {
            state.generation_done = true;
        }
    }
    if (!state.generation_done) {
        state.last_codes = codes;
    }
    return codes;
}

std::vector<int32_t> offset_modality_codes(const HiggsTTSConfig & config, const std::vector<int32_t> & codes) {
    if (static_cast<int64_t>(codes.size()) != config.audio_encoder.num_codebooks) {
        throw std::runtime_error("Higgs TTS modality codebook count mismatch");
    }
    std::vector<int32_t> out(codes.size(), 0);
    for (int64_t codebook = 0; codebook < config.audio_encoder.num_codebooks; ++codebook) {
        const int32_t code = codes[static_cast<size_t>(codebook)];
        if (code < 0 || static_cast<int64_t>(code) >= config.audio_encoder.vocab_size) {
            throw std::runtime_error("Higgs TTS generated code is outside the modality vocabulary");
        }
        out[static_cast<size_t>(codebook)] = static_cast<int32_t>(
            codebook * config.audio_encoder.vocab_size + static_cast<int64_t>(code));
    }
    return out;
}

class HiggsTTSGeneratorWeightsRuntime {
public:
    HiggsTTSGeneratorWeightsRuntime(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<HiggsTTSGeneratorWeights>(
              load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Higgs TTS generator weights runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Higgs TTS generator backend is not initialized");
        }
    }

    const HiggsTTSAssets & assets() const noexcept {
        return *assets_;
    }

    const HiggsTTSGeneratorWeights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const HiggsTTSAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const HiggsTTSGeneratorWeights> weights_;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<HiggsTTSGeneratorWeightsRuntime> runtime,
        int64_t prompt_steps,
        PromptReferenceInfo reference,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          reference_(reference) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Higgs TTS generator prefill requires positive prompt length");
        }
        if (reference_.tokens < 0 || reference_.start < 0 || reference_.start + reference_.tokens > prompt_steps_) {
            throw std::runtime_error("Higgs TTS generator prefill reference span is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS generator prefill graph context");
        }
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.generator.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        if (reference_.tokens > 0) {
            auto reference_codes = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({reference_.tokens, config.audio_encoder.num_codebooks}));
            reference_code_ids_ = reference_codes.tensor;
        }
        auto x = prompt_embeddings_with_optional_reference(
            ctx,
            weights,
            config,
            token_ids_,
            reference_code_ids_,
            prompt_steps_,
            reference_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        for (const auto & layer : weights.layers) {
            auto out = decoder_layer(ctx, x, positions, layer, config);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.text.hidden_size, config.text.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modality_logits(ctx, x, weights, config);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS generator prefill graph");
        }
        std::vector<int32_t> pos(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        debug::timing_log_scalar("higgs_tts.generator.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("higgs_tts.generator.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(
        const HiggsTTSGeneratorWeightsRuntime & runtime,
        int64_t prompt_steps,
        const PromptReferenceInfo & reference) const {
        return runtime_.get() == &runtime &&
            prompt_steps_ == prompt_steps &&
            reference_.start == reference.start &&
            reference_.tokens == reference.tokens;
    }

    PrefillOutput run(
        const std::vector<int32_t> & token_ids,
        const HiggsAudioCodeMatrix * reference_delayed_codes) {
        const auto & config = runtime_->assets().config;
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Higgs TTS generator prefill token id count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (reference_.tokens > 0) {
            if (reference_delayed_codes == nullptr || reference_delayed_codes->frames != reference_.tokens) {
                throw std::runtime_error("Higgs TTS generator prefill reference code count mismatch");
            }
            const auto offset_reference_codes = offset_modality_code_matrix(config, *reference_delayed_codes);
            ggml_backend_tensor_set(
                reference_code_ids_,
                offset_reference_codes.data(),
                0,
                offset_reference_codes.size() * sizeof(int32_t));
        }
        debug::timing_log_scalar("higgs_tts.generator.prefill_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("higgs_tts.generator.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS generator prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(modality_vocab_size(config)));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            logits_,
            out.logits.data(),
            0,
            out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config.text.num_key_value_heads * head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        debug::timing_log_scalar("higgs_tts.generator.prefill_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<HiggsTTSGeneratorWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    PromptReferenceInfo reference_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * reference_code_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class DecodeGraph {
public:
    DecodeGraph(
        std::shared_ptr<HiggsTTSGeneratorWeightsRuntime> runtime,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("Higgs TTS generator decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS generator decode graph context");
        }
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        const int64_t dim = head_dim(config);
        core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.generator.decode", runtime_->backend_type()};
        offset_code_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, config.audio_encoder.num_codebooks);
        auto x = modality_code_embedding(ctx, weights, config, offset_code_ids_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        for (const auto & layer : weights.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.text.num_key_value_heads, dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.text.num_key_value_heads, dim})));
            auto out = decoder_layer_with_static_cache(
                ctx,
                x,
                positions,
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                cache_slot,
                attention_mask);
            x = out.output;
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_,
            config.text.num_key_value_heads * dim,
            std::move(cache_keys),
            std::move(cache_values));
        x = modules::RMSNormModule({config.text.hidden_size, config.text.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modality_logits(ctx, x, weights, config);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS generator decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("higgs_tts.generator.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("higgs_tts.generator.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const HiggsTTSGeneratorWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    void reset_timing() noexcept {
        input_upload_ms_ = 0.0;
        mask_upload_ms_ = 0.0;
        graph_compute_ms_ = 0.0;
        logits_read_ms_ = 0.0;
    }

    void run_step_into(const std::vector<int32_t> & codes, std::vector<float> & logits) {
        const auto & config = runtime_->assets().config;
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("Higgs TTS generator decode cache exhausted");
        }
        auto timing_start = Clock::now();
        const auto offset_codes = offset_modality_codes(config, codes);
        ggml_backend_tensor_set(offset_code_ids_, offset_codes.data(), 0, offset_codes.size() * sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        input_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_values_[static_cast<size_t>(i)] = visible;
        }
        attention_mask_values_[static_cast<size_t>(cache_slot)] = visible;
        timing_start = Clock::now();
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        mask_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        graph_compute_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS generator decode graph compute failed");
        }
        logits.resize(static_cast<size_t>(modality_vocab_size(config)));
        timing_start = Clock::now();
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        logits_read_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        step_cache_.advance_after_direct_append(1);
    }

    void log_timing() const {
        debug::timing_log_scalar("higgs_tts.generator.decode.input_upload_ms", input_upload_ms_);
        debug::timing_log_scalar("higgs_tts.generator.decode.mask_upload_ms", mask_upload_ms_);
        debug::timing_log_scalar("higgs_tts.generator.decode.graph.compute_ms", graph_compute_ms_);
        debug::timing_log_scalar("higgs_tts.generator.decode.logits_read_ms", logits_read_ms_);
    }

private:
    std::shared_ptr<HiggsTTSGeneratorWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * offset_code_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    double input_upload_ms_ = 0.0;
    double mask_upload_ms_ = 0.0;
    double graph_compute_ms_ = 0.0;
    double logits_read_ms_ = 0.0;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct HiggsTTSGeneratorRuntime::Impl {
    Impl(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<HiggsTTSGeneratorWeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes) {}

    HiggsTTSGeneratedCodes generate(
        const HiggsTTSPrompt & prompt,
        const HiggsTTSGenerationOptions & options,
        const HiggsAudioCodeMatrix * reference_delayed_codes,
        const HiggsTTSCodeStreamCallback * code_stream) {
        const auto & config = weights->assets().config;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("Higgs TTS generator prompt is empty");
        }
        if (options.max_tokens <= 0) {
            throw std::runtime_error("Higgs TTS max_tokens must be positive");
        }
        if (!(options.temperature >= 0.0F && options.temperature <= 2.0F)) {
            throw std::runtime_error("Higgs TTS temperature must be in [0, 2]");
        }
        if (!(options.top_p >= 0.0F && options.top_p <= 1.0F)) {
            throw std::runtime_error("Higgs TTS top_p must be in [0, 1]");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps + options.max_tokens > config.text.max_position_embeddings) {
            throw std::runtime_error("Higgs TTS request exceeds max_position_embeddings");
        }
        const auto reference = resolve_prompt_reference(prompt, reference_delayed_codes, config);
        if (prefill_graph == nullptr || !prefill_graph->matches(*weights, prompt_steps, reference)) {
            prefill_graph = std::make_unique<PrefillGraph>(
                weights,
                prompt_steps,
                reference,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("higgs_tts.generator.prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar("higgs_tts.generator.prefill_prompt_steps", prompt_steps);
        }
        auto timing_start = Clock::now();
        auto prefill = prefill_graph->run(prompt.input_ids, reference_delayed_codes);
        debug::timing_log_scalar("higgs_tts.generator.prefill_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        prefill_graph.reset();
        const int64_t required_cache_steps = prompt_steps + options.max_tokens;
        if (decode_graph == nullptr || !decode_graph->can_run(*weights, required_cache_steps)) {
            decode_graph = std::make_unique<DecodeGraph>(weights, required_cache_steps, decode_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("higgs_tts.generator.decode.graph.build_ms", 0.0);
            debug::trace_log_scalar("higgs_tts.generator.decode_cache_steps", required_cache_steps);
        }
        decode_graph->import_state(prefill.kv_state);
        decode_graph->reset_timing();

        HiggsSamplerState sampler;
        sampler.num_codebooks = config.audio_encoder.num_codebooks;
        std::mt19937 rng(options.seed);
        SamplingScratch scratch;
        HiggsAudioCodeMatrix delayed;
        delayed.codebooks = config.audio_encoder.num_codebooks;
        delayed.token_ids.reserve(static_cast<size_t>(options.max_tokens * delayed.codebooks));
        std::vector<float> logits = std::move(prefill.logits);
        double sampling_ms = 0.0;
        double decode_step_ms = 0.0;
        timing_start = Clock::now();
        for (int64_t step = 0; step < options.max_tokens; ++step) {
            const auto sampling_start = Clock::now();
            const auto codes = sample_higgs_step(logits, sampler, config, options, rng, scratch);
            sampling_ms += engine::debug::elapsed_ms(sampling_start, Clock::now());
            if (!codes.empty() && codes.front() != kHiggsAudioStopCode) {
                delayed.token_ids.insert(delayed.token_ids.end(), codes.begin(), codes.end());
                ++delayed.frames;
                if (code_stream != nullptr && delayed.frames >= delayed.codebooks) {
                    if (!(*code_stream)(delayed, false)) {
                        throw std::runtime_error("generation cancelled");
                    }
                }
            }
            if (sampler.generation_done) {
                break;
            }
            if (sampler.last_codes.empty()) {
                break;
            }
            const auto step_start = Clock::now();
            decode_graph->run_step_into(sampler.last_codes, logits);
            decode_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
        }
        debug::trace_log_scalar("higgs_tts.generator.stopped_by_eoc", sampler.generation_done);
        debug::trace_log_scalar("higgs_tts.generator.hit_max_tokens", !sampler.generation_done);
        debug::timing_log_scalar("higgs_tts.generator.sampling_ms", sampling_ms);
        debug::timing_log_scalar("higgs_tts.generator.decode_step_ms", decode_step_ms);
        decode_graph->log_timing();
        debug::timing_log_scalar("higgs_tts.generator.decode_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        debug::trace_log_scalar("higgs_tts.generator.delayed_rows", delayed.frames);
        if (delayed.frames < delayed.codebooks) {
            throw std::runtime_error("Higgs TTS generated too few audio-code rows");
        }
        if (code_stream != nullptr) {
            if (!(*code_stream)(delayed, true)) {
                throw std::runtime_error("generation cancelled");
            }
        }
        HiggsTTSGeneratedCodes out;
        out.delayed_codes = std::move(delayed);
        out.raw_codes = reverse_delay_pattern(out.delayed_codes);
        debug::trace_log_scalar("higgs_tts.generator.raw_frames", out.raw_codes.frames);
        return out;
    }

    std::shared_ptr<HiggsTTSGeneratorWeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
};

HiggsTTSGeneratorRuntime::HiggsTTSGeneratorRuntime(
    std::shared_ptr<const HiggsTTSAssets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

HiggsTTSGeneratorRuntime::~HiggsTTSGeneratorRuntime() = default;

HiggsTTSGeneratedCodes HiggsTTSGeneratorRuntime::generate(
    const HiggsTTSPrompt & prompt,
    const HiggsTTSGenerationOptions & options,
    const HiggsAudioCodeMatrix * reference_delayed_codes,
    const HiggsTTSCodeStreamCallback * code_stream) {
    return impl_->generate(prompt, options, reference_delayed_codes, code_stream);
}


}  // namespace engine::models::higgs_tts
