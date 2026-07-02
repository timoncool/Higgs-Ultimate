#include "engine/models/higgs_tts/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

namespace engine::models::higgs_tts {
namespace json = engine::io::json;
namespace {

std::filesystem::path canonical_existing_file(const std::filesystem::path & path) {
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path canonical_existing_directory(const std::filesystem::path & path) {
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return canonical_existing_directory(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return canonical_existing_directory(model_path.parent_path());
    }
    throw std::runtime_error("Higgs TTS model path does not exist: " + model_path.string());
}

bool is_model_weight_file(const std::filesystem::path & path) {
    const auto extension = path.extension().string();
    return extension == ".safetensors" || extension == ".gguf";
}

std::filesystem::path resolve_model_weights_path(
    const std::filesystem::path & model_path,
    const std::filesystem::path & model_root) {
    if (engine::io::is_existing_file(model_path) && is_model_weight_file(model_path)) {
        return canonical_existing_file(model_path);
    }
    const auto gguf = model_root / "model.gguf";
    if (engine::io::is_existing_file(gguf)) {
        return canonical_existing_file(gguf);
    }
    const auto safetensors = model_root / "model.safetensors";
    if (engine::io::is_existing_file(safetensors)) {
        return canonical_existing_file(safetensors);
    }
    throw std::runtime_error(
        "Higgs TTS missing model weights; searched:\n  " + gguf.string() + "\n  " + safetensors.string());
}

std::optional<std::filesystem::path> env_path(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path(value);
}

void append_unique_existing_dir(std::vector<std::filesystem::path> & dirs, const std::filesystem::path & path) {
    if (path.empty() || !engine::io::is_existing_directory(path)) {
        return;
    }
    const auto canonical = canonical_existing_directory(path);
    for (const auto & dir : dirs) {
        if (dir == canonical) {
            return;
        }
    }
    dirs.push_back(canonical);
}

std::vector<std::filesystem::path> small_asset_roots(const std::filesystem::path & model_root) {
    std::vector<std::filesystem::path> roots;
    if (const auto env = env_path("HIGGS_TTS_SMALL_ASSETS_ROOT")) {
        append_unique_existing_dir(roots, *env);
    }
    if (const auto env = env_path("HIGGS_TTS_ASSETS_ROOT")) {
        append_unique_existing_dir(roots, *env);
    }
    append_unique_existing_dir(roots, model_root);
    append_unique_existing_dir(roots, model_root / "higgs-audio-v3-tts-4b");
    return roots;
}

std::filesystem::path require_first_file(
    const std::vector<std::filesystem::path> & roots,
    const std::filesystem::path & relative_path,
    const char * role) {
    for (const auto & root : roots) {
        const auto path = root / relative_path;
        if (engine::io::is_existing_file(path)) {
            return canonical_existing_file(path);
        }
    }
    std::string searched;
    for (const auto & root : roots) {
        searched += "\n  " + (root / relative_path).string();
    }
    throw std::runtime_error(std::string("Higgs TTS missing ") + role + "; searched:" + searched);
}

std::filesystem::path optional_first_file(
    const std::vector<std::filesystem::path> & roots,
    const std::filesystem::path & relative_path) {
    for (const auto & root : roots) {
        const auto path = root / relative_path;
        if (engine::io::is_existing_file(path)) {
            return canonical_existing_file(path);
        }
    }
    return {};
}

std::filesystem::path optional_codec_config(
    const std::filesystem::path & model_root,
    const std::vector<std::filesystem::path> & small_roots) {
    if (const auto env = env_path("HIGGS_TTS_CODEC_CONFIG")) {
        if (engine::io::is_existing_file(*env)) {
            return canonical_existing_file(*env);
        }
        throw std::runtime_error("HIGGS_TTS_CODEC_CONFIG does not point to a file: " + env->string());
    }
    std::vector<std::filesystem::path> roots;
    append_unique_existing_dir(roots, model_root);
    for (const auto & root : small_roots) {
        append_unique_existing_dir(roots, root);
        append_unique_existing_dir(roots, root.parent_path());
    }
    return optional_first_file(roots, "higgs_audio_v2_tokenizer_config.json");
}

bool architecture_contains(const engine::io::json::Value & root, const std::string & needle) {
    const auto * architectures = root.find("architectures");
    if (architectures == nullptr || !architectures->is_array()) {
        return false;
    }
    for (const auto & item : architectures->as_array()) {
        if (item.is_string() && item.as_string().find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

float optional_rope_theta(const engine::io::json::Value & config, float default_value) {
    if (const auto * rope = config.find("rope_parameters"); rope != nullptr && rope->is_object()) {
        return json::optional_f32(*rope, "rope_theta", default_value);
    }
    if (const auto * rope = config.find("rope_scaling"); rope != nullptr && rope->is_object()) {
        return json::optional_f32(*rope, "rope_theta", default_value);
    }
    return json::optional_f32(config, "rope_theta", default_value);
}

HiggsTextConfig parse_text_config(const engine::io::json::Value & value) {
    HiggsTextConfig config;
    config.model_type = value.require("model_type").as_string();
    config.vocab_size = value.require("vocab_size").as_i64();
    config.hidden_size = value.require("hidden_size").as_i64();
    config.intermediate_size = value.require("intermediate_size").as_i64();
    config.num_hidden_layers = value.require("num_hidden_layers").as_i64();
    config.num_attention_heads = value.require("num_attention_heads").as_i64();
    config.num_key_value_heads = value.require("num_key_value_heads").as_i64();
    config.head_dim = json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = value.require("max_position_embeddings").as_i64();
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.pad_token_id = json::optional_i64(value, "pad_token_id", config.eos_token_id);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = optional_rope_theta(value, config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);

    if (config.model_type != "qwen3") {
        throw std::runtime_error("Higgs TTS expects text_config.model_type=qwen3");
    }
    if (config.vocab_size <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_hidden_layers <= 0 || config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
        config.head_dim <= 0 || config.max_position_embeddings <= 0) {
        throw std::runtime_error("Higgs TTS text_config contains non-positive dimensions");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Higgs TTS text hidden_size must be divisible by num_attention_heads");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Higgs TTS text attention heads must be divisible by key-value heads");
    }
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("Higgs TTS currently expects tied text embeddings");
    }
    return config;
}

HiggsAudioEncoderConfig parse_audio_encoder_config(const engine::io::json::Value & value) {
    HiggsAudioEncoderConfig config;
    config.model_type = value.require("model_type").as_string();
    config.encoder_type = value.require("encoder_type").as_string();
    config.num_codebooks = value.require("num_codebooks").as_i64();
    config.vocab_size = value.require("vocab_size").as_i64();
    config.out_dim = value.require("out_dim").as_i64();
    config.max_chunk_size = json::optional_i64(value, "max_chunk_size", config.max_chunk_size);
    config.mel_per_sample = json::optional_i64(value, "mel_per_sample", config.mel_per_sample);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.use_delay_pattern = json::optional_bool(value, "use_delay_pattern", config.use_delay_pattern);

    if (config.model_type != "higgs_audio_encoder") {
        throw std::runtime_error("Higgs TTS expects audio_encoder_config.model_type=higgs_audio_encoder");
    }
    if (config.encoder_type != "discrete") {
        throw std::runtime_error("Higgs TTS currently supports only the discrete audio encoder");
    }
    if (config.num_codebooks <= 0 || config.vocab_size <= 0 || config.out_dim <= 0 ||
        config.max_chunk_size <= 0 || config.mel_per_sample <= 0) {
        throw std::runtime_error("Higgs TTS audio_encoder_config contains non-positive dimensions");
    }
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("Higgs TTS currently expects tied audio embeddings");
    }
    if (!config.use_delay_pattern) {
        throw std::runtime_error("Higgs TTS currently expects delay-pattern audio codes");
    }
    return config;
}

HiggsTTSConfig parse_config(const std::filesystem::path & config_path) {
    const auto root = engine::io::json::parse_file(config_path);
    HiggsTTSConfig config;
    config.model_type = root.require("model_type").as_string();
    config.audio_token_id = json::optional_i64(root, "audio_token_id", config.audio_token_id);
    config.ignore_index = json::optional_i64(root, "ignore_index", config.ignore_index);
    config.text = parse_text_config(root.require("text_config"));
    config.audio_encoder = parse_audio_encoder_config(root.require("audio_encoder_config"));

    if (config.model_type != "higgs_multimodal_qwen3") {
        throw std::runtime_error("Higgs TTS expects model_type=higgs_multimodal_qwen3");
    }
    if (!architecture_contains(root, "HiggsMultimodalQwen3")) {
        throw std::runtime_error("Higgs TTS config is missing HiggsMultimodalQwen3 architecture marker");
    }
    if (config.text.hidden_size != config.audio_encoder.out_dim) {
        throw std::runtime_error("Higgs TTS text hidden_size must match audio encoder out_dim");
    }
    return config;
}

HiggsTTSAssetPaths resolve_paths(const std::filesystem::path & model_path) {
    const auto model_root = resolve_model_root(model_path);
    const auto small_roots = small_asset_roots(model_root);
    HiggsTTSAssetPaths paths;
    paths.model_root = model_root;
    paths.model_weights_path = resolve_model_weights_path(model_path, model_root);
    paths.config_path = require_first_file(small_roots, "config.json", "config");
    paths.tokenizer_config_path = require_first_file(small_roots, "tokenizer_config.json", "tokenizer config");
    paths.tokenizer_json_path = require_first_file(small_roots, "tokenizer.json", "tokenizer json");
    paths.model_index_path = optional_first_file(small_roots, "model.safetensors.index.json");
    paths.chat_template_path = optional_first_file(small_roots, "chat_template.jinja");
    paths.codec_config_path = optional_codec_config(model_root, small_roots);
    paths.small_assets_root = paths.config_path.parent_path();
    return paths;
}

}  // namespace

HiggsTTSAssetPaths resolve_higgs_tts_assets(const std::filesystem::path & model_path) {
    return resolve_paths(model_path);
}

std::shared_ptr<const HiggsTTSAssets> load_higgs_tts_assets(const std::filesystem::path & model_path) {
    HiggsTTSAssets assets;
    assets.paths = resolve_paths(model_path);
    assets.config = parse_config(assets.paths.config_path);
    assets.model_weights = engine::assets::open_tensor_source(assets.paths.model_weights_path);
    return std::make_shared<HiggsTTSAssets>(std::move(assets));
}

}  // namespace engine::models::higgs_tts
