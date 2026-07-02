#include "engine/models/higgs_tts/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/higgs_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Higgs TTS model path does not exist: " + model_path.string());
}

bool has_higgs_tts_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "model.safetensors");
}

void push_asset(std::vector<runtime::NamedAsset> & assets, std::string id, const std::filesystem::path & path) {
    if (!path.empty() && engine::io::is_existing_file(path)) {
        assets.push_back({std::move(id), std::filesystem::weakly_canonical(path)});
    }
}

std::vector<runtime::NamedAsset> discovered_config_assets(const HiggsTTSAssetPaths & paths) {
    std::vector<runtime::NamedAsset> assets;
    push_asset(assets, "config", paths.config_path);
    push_asset(assets, "tokenizer_config", paths.tokenizer_config_path);
    push_asset(assets, "tokenizer", paths.tokenizer_json_path);
    push_asset(assets, "chat_template", paths.chat_template_path);
    push_asset(assets, "codec_config", paths.codec_config_path);
    return assets;
}

std::vector<runtime::NamedAsset> discovered_weight_assets(const HiggsTTSAssetPaths & paths) {
    std::vector<runtime::NamedAsset> assets;
    push_asset(assets, "model", paths.model_weights_path);
    push_asset(assets, "model.safetensors.index", paths.model_index_path);
    return assets;
}

runtime::CapabilitySet make_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    capabilities.supports_timestamps = false;
    return capabilities;
}

runtime::ModelMetadata make_metadata(const HiggsTTSAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "higgs_tts";
    metadata.variant = assets.config.text.model_type + "-" +
        std::to_string(assets.config.text.num_hidden_layers) + "l-" +
        std::to_string(assets.config.audio_encoder.num_codebooks) + "codebook";
    metadata.description = "Higgs Audio v3 TTS loaded from local Hugging Face safetensors assets.";
    metadata.config_candidates = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "higgs_audio_v2_tokenizer_config.json",
    };
    metadata.weight_candidates = {"model.safetensors", "model.safetensors.index.json"};
    return metadata;
}

    runtime::ModelCliInterface make_cli_interface() {
    runtime::ModelCliInterface cli;
    cli.request_options = {
        {"--voice-ref", "wav", "Speaker reference audio for zero-shot voice cloning"},
        {"--reference-text", "text", "Transcript for the speaker reference audio"},
        {"--temperature", "value", "Sampling temperature"},
        {"--top-p", "value", "Nucleus sampling threshold"},
        {"--top-k", "n", "Top-k sampling limit"},
        {"--repetition-penalty", "value", "Repetition penalty"},
        {"--max-tokens", "n", "Maximum generated audio-code tokens"},
        {"--seed", "n", "Request seed; omitted means random"},
        {"--text-chunk-size", "n", "Maximum text codepoints per generated chunk"},
    };
    cli.session_options = {
        {"--session-option higgs_tts.weight_type", "native|f32|f16|bf16|q8_0", "Language-model weight storage type"},
        {"--session-option higgs_tts.codec_weight_type", "native|f32|f16|bf16|q8_0", "Audio codec weight storage type"},
    };
    return cli;
}

class HiggsTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_tts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            if (!has_higgs_tts_assets(root) ||
                (request.family_hint.has_value() && *request.family_hint != family())) {
                return false;
            }
            (void) load_higgs_tts_assets(root);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_higgs_tts_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = make_metadata(*assets);
        inspection.capabilities = make_capabilities();
        inspection.cli = make_cli_interface();
        inspection.discovered_configs = discovered_config_assets(assets->paths);
        inspection.discovered_weights = discovered_weight_assets(assets->paths);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_higgs_tts_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

HiggsTTSLoadedModel::HiggsTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HiggsTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HiggsTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HiggsTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HiggsTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS only supports the Tts task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS currently supports offline sessions");
    }
    return std::make_unique<HiggsTTSSession>(task, options, assets_);
}

std::unique_ptr<HiggsTTSLoadedModel> load_higgs_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_higgs_tts_assets(model_path);
    return std::make_unique<HiggsTTSLoadedModel>(
        make_metadata(*assets),
        make_capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_tts_loader() {
    return std::make_shared<HiggsTTSLoader>();
}

}  // namespace engine::models::higgs_tts
