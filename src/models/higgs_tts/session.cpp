#include "engine/models/higgs_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultGeneratorWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorPrefillGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorDecodeGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecWeightContextBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecGraphArenaBytes = 1024ull * 1024ull * 1024ull;

std::shared_ptr<const HiggsTTSAssets> require_assets(std::shared_ptr<const HiggsTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs TTS session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

HiggsTTSGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) {
    HiggsTTSGenerationOptions out;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.max_tokens = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    out.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        out.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (out.max_tokens <= 0) {
        throw std::runtime_error("Higgs TTS max_tokens must be positive");
    }
    if (out.top_k < 0) {
        throw std::runtime_error("Higgs TTS top_k must be non-negative");
    }
    if (out.temperature < 0.0F || out.temperature > 2.0F) {
        throw std::runtime_error("Higgs TTS temperature must be in [0, 2]");
    }
    if (out.top_p < 0.0F || out.top_p > 1.0F) {
        throw std::runtime_error("Higgs TTS top_p must be in [0, 1]");
    }
    return out;
}

}  // namespace

HiggsTTSSession::HiggsTTSSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const HiggsTTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS only supports the Tts task");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS currently supports offline sessions");
    }
    const auto generator_weight_type = option_weight_type(
        options,
        "higgs_tts.weight_type",
        engine::assets::TensorStorageType::Native);
    validate_matmul_weight_storage(generator_weight_type, "higgs_tts.weight_type");
    const auto codec_weight_type = option_weight_type(
        options,
        "higgs_tts.codec_weight_type",
        engine::assets::TensorStorageType::Native);
    validate_matmul_weight_storage(codec_weight_type, "higgs_tts.codec_weight_type");
    generator_ = std::make_unique<HiggsTTSGeneratorRuntime>(
        assets_,
        execution_context(),
        runtime::parse_size_mb_option(
            options.options,
            {"higgs_tts.prefill_graph_arena_mb"},
            kDefaultGeneratorPrefillGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options,
            {"higgs_tts.decode_graph_arena_mb"},
            kDefaultGeneratorDecodeGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options,
            {"higgs_tts.weight_context_mb"},
            kDefaultGeneratorWeightContextBytes),
        generator_weight_type);
    codec_decoder_ = std::make_unique<HiggsAudioCodecDecoderRuntime>(
        assets_,
        execution_context(),
        runtime::parse_size_mb_option(
            options.options,
            {"higgs_tts.codec_graph_arena_mb"},
            kDefaultCodecGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options,
            {"higgs_tts.codec_weight_context_mb"},
            kDefaultCodecWeightContextBytes),
        codec_weight_type);
    assets_->model_weights->release_storage();
}

std::string HiggsTTSSession::family() const {
    return "higgs_tts";
}

runtime::VoiceTaskKind HiggsTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HiggsTTSSession::run_mode() const {
    return task_.mode;
}

void HiggsTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult HiggsTTSSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("Higgs TTS run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Higgs TTS requires text input");
    }
    const std::string reference_text =
        request.options.count("reference_text") != 0 ? request.options.at("reference_text") : std::string{};
    const auto generation = generation_options_from_request(request);
    engine::debug::trace_log_scalar("higgs_tts.sampling.max_tokens", generation.max_tokens);
    engine::debug::trace_log_scalar("higgs_tts.sampling.temperature", generation.temperature);
    engine::debug::trace_log_scalar("higgs_tts.sampling.top_k", generation.top_k);
    engine::debug::trace_log_scalar("higgs_tts.sampling.top_p", generation.top_p);
    engine::debug::trace_log_scalar("higgs_tts.sampling.seed", generation.seed);
    std::optional<HiggsAudioCodeMatrix> reference_delayed_codes;
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        auto reference_raw_codes = codec_decoder_->encode_reference_audio(*request.voice->speaker->audio);
        reference_delayed_codes = apply_delay_pattern(reference_raw_codes);
        engine::debug::trace_log_scalar("higgs_tts.reference.raw_code_frames", reference_raw_codes.frames);
        engine::debug::trace_log_scalar("higgs_tts.reference.delayed_code_rows", reference_delayed_codes->frames);
    }
    const auto prompt = tokenizer_.build_prompt(
        request.text_input->text,
        reference_delayed_codes.has_value() ? reference_delayed_codes->frames : 0,
        reference_text);
    const auto codes = generator_->generate(
        prompt,
        generation,
        reference_delayed_codes.has_value() ? &*reference_delayed_codes : nullptr);
    engine::debug::trace_log_scalar("higgs_tts.delayed_code_rows", codes.delayed_codes.frames);
    engine::debug::trace_log_scalar("higgs_tts.raw_code_frames", codes.raw_codes.frames);
    auto audio = codec_decoder_->decode(codes.raw_codes);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    return result;
}

}  // namespace engine::models::higgs_tts
