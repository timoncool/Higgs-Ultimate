#include "engine/models/higgs_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultGeneratorWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorPrefillGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorDecodeGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecWeightContextBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecGraphArenaBytes = 4ull * 1024ull * 1024ull * 1024ull;
constexpr int64_t kDefaultTextChunkSize = 300;
constexpr int64_t kDefaultFirstStreamFrames = 20;
constexpr int64_t kDefaultStreamFrames = 40;

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

int64_t positive_int_option(
    const runtime::TaskRequest & request,
    const char * key,
    int64_t fallback) {
    const auto value = runtime::parse_int_option(request.options, {key}).value_or(fallback);
    return value > 0 ? value : fallback;
}

bool bool_option(
    const runtime::TaskRequest & request,
    const char * key,
    bool fallback) {
    const auto value = runtime::find_option(request.options, {key});
    if (!value.has_value()) {
        return fallback;
    }
    return runtime::parse_bool_option(*value, key);
}

class RuntimeCacheReleaseGuard {
public:
    RuntimeCacheReleaseGuard(
        HiggsTTSGeneratorRuntime * generator,
        HiggsAudioCodecDecoderRuntime * codec)
        : generator_(generator), codec_(codec) {}

    ~RuntimeCacheReleaseGuard() noexcept {
        if (!active_) {
            return;
        }
        if (codec_ != nullptr) {
            codec_->release_runtime_cache();
        }
        if (generator_ != nullptr) {
            generator_->release_runtime_cache();
        }
    }

    RuntimeCacheReleaseGuard(const RuntimeCacheReleaseGuard &) = delete;
    RuntimeCacheReleaseGuard & operator=(const RuntimeCacheReleaseGuard &) = delete;

    void dismiss() noexcept {
        active_ = false;
    }

private:
    HiggsTTSGeneratorRuntime * generator_ = nullptr;
    HiggsAudioCodecDecoderRuntime * codec_ = nullptr;
    bool active_ = true;
};

struct ReferenceAudioKey {
    int sample_rate = 0;
    int channels = 0;
    std::size_t sample_count = 0;
    std::uint64_t fingerprint = 0;
};

ReferenceAudioKey reference_audio_key(const runtime::AudioBuffer & audio) {
    ReferenceAudioKey key;
    key.sample_rate = audio.sample_rate;
    key.channels = audio.channels;
    key.sample_count = audio.samples.size();
    std::uint64_t fp = 14695981039346656037ull;
    for (const float sample : audio.samples) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        fp ^= static_cast<std::uint64_t>(bits);
        fp *= 1099511628211ull;
    }
    key.fingerprint = fp;
    return key;
}

bool reference_key_matches(
    const ReferenceAudioKey & key,
    std::uint64_t cached_fingerprint,
    std::size_t cached_sample_count,
    int cached_sample_rate,
    int cached_channels) {
    return key.sample_rate == cached_sample_rate &&
           key.channels == cached_channels &&
           key.sample_count == cached_sample_count &&
           key.fingerprint == cached_fingerprint;
}

std::optional<std::string> reference_cache_path(const runtime::TaskRequest & request) {
    const auto found = runtime::find_option(request.options, {"reference_cache_path"});
    if (!found.has_value() || found->empty()) {
        return std::nullopt;
    }
    return *found;
}

template <typename T>
void write_binary(std::ostream & out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
T read_binary(std::istream & in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("speaker cache file is truncated");
    }
    return value;
}

void write_reference_cache(
    const std::string & path,
    const ReferenceAudioKey & key,
    const HiggsAudioCodeMatrix & delayed_codes) {
    std::filesystem::path cache_path(path);
    if (cache_path.has_parent_path()) {
        std::filesystem::create_directories(cache_path.parent_path());
    }
    std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create speaker cache: " + path);
    }
    const char magic[12] = {'H', 'S', 'P', 'K', 'C', 'A', 'C', 'H', 'E', '0', '1', '\0'};
    out.write(magic, sizeof(magic));
    write_binary<std::uint32_t>(out, 1);
    write_binary<std::int32_t>(out, key.sample_rate);
    write_binary<std::int32_t>(out, key.channels);
    write_binary<std::uint64_t>(out, static_cast<std::uint64_t>(key.sample_count));
    write_binary<std::uint64_t>(out, key.fingerprint);
    write_binary<std::int64_t>(out, delayed_codes.frames);
    write_binary<std::int64_t>(out, delayed_codes.codebooks);
    write_binary<std::uint64_t>(out, static_cast<std::uint64_t>(delayed_codes.token_ids.size()));
    out.write(
        reinterpret_cast<const char *>(delayed_codes.token_ids.data()),
        static_cast<std::streamsize>(delayed_codes.token_ids.size() * sizeof(std::int32_t)));
    if (!out) {
        throw std::runtime_error("failed to write speaker cache: " + path);
    }
}

std::optional<HiggsAudioCodeMatrix> read_reference_cache(
    const std::string & path,
    const ReferenceAudioKey & key) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    char magic[12] = {};
    in.read(magic, sizeof(magic));
    const char expected[12] = {'H', 'S', 'P', 'K', 'C', 'A', 'C', 'H', 'E', '0', '1', '\0'};
    if (std::memcmp(magic, expected, sizeof(expected)) != 0) {
        return std::nullopt;
    }
    const auto schema = read_binary<std::uint32_t>(in);
    if (schema != 1) {
        return std::nullopt;
    }
    const auto sample_rate = read_binary<std::int32_t>(in);
    const auto channels = read_binary<std::int32_t>(in);
    const auto sample_count = read_binary<std::uint64_t>(in);
    const auto fingerprint = read_binary<std::uint64_t>(in);
    if (sample_rate != key.sample_rate ||
        channels != key.channels ||
        sample_count != static_cast<std::uint64_t>(key.sample_count) ||
        fingerprint != key.fingerprint) {
        return std::nullopt;
    }

    HiggsAudioCodeMatrix delayed;
    delayed.frames = read_binary<std::int64_t>(in);
    delayed.codebooks = read_binary<std::int64_t>(in);
    const auto count = read_binary<std::uint64_t>(in);
    if (delayed.frames <= 0 || delayed.codebooks <= 0 ||
        count != static_cast<std::uint64_t>(delayed.frames * delayed.codebooks)) {
        return std::nullopt;
    }
    delayed.token_ids.resize(static_cast<std::size_t>(count));
    in.read(
        reinterpret_cast<char *>(delayed.token_ids.data()),
        static_cast<std::streamsize>(delayed.token_ids.size() * sizeof(std::int32_t)));
    if (!in) {
        return std::nullopt;
    }
    return delayed;
}

void set_in_memory_reference_cache(
    HiggsAudioCodeMatrix delayed_codes,
    const ReferenceAudioKey & key,
    std::optional<HiggsAudioCodeMatrix> & cached_reference_codes,
    std::uint64_t & cached_reference_fingerprint,
    std::size_t & cached_reference_sample_count,
    int & cached_reference_sample_rate,
    int & cached_reference_channels,
    bool & has_cached_reference) {
    cached_reference_codes = std::move(delayed_codes);
    cached_reference_fingerprint = key.fingerprint;
    cached_reference_sample_count = key.sample_count;
    cached_reference_sample_rate = key.sample_rate;
    cached_reference_channels = key.channels;
    has_cached_reference = true;
}

int64_t bytes_to_mib(int64_t bytes) {
    return bytes / (1024 * 1024);
}

bool should_continue_generation(const HiggsTTSSession::CancelCallback & should_continue) {
    return !should_continue || should_continue();
}

void require_generation_alive(const HiggsTTSSession::CancelCallback & should_continue) {
    if (!should_continue_generation(should_continue)) {
        throw std::runtime_error("generation cancelled");
    }
}

bool emit_memory_diagnostic(
    const core::ExecutionContext & execution,
    const HiggsTTSSession::StreamProgressCallback & on_progress,
    const char * stage) {
    const auto snapshot = execution.memory_snapshot();
    if (!snapshot.available) {
        return true;
    }
    const auto used_mib = bytes_to_mib(snapshot.used_bytes);
    const auto free_mib = bytes_to_mib(snapshot.free_bytes);
    const auto total_mib = bytes_to_mib(snapshot.total_bytes);
    const std::string metric_prefix = std::string("higgs_tts.vram.") + stage;
    engine::debug::trace_log_scalar(metric_prefix + ".used_mib", used_mib);
    engine::debug::trace_log_scalar(metric_prefix + ".free_mib", free_mib);
    engine::debug::trace_log_scalar(metric_prefix + ".total_mib", total_mib);
    if (!on_progress) {
        return true;
    }
    std::ostringstream phase;
    phase << "diag:vram stage=" << stage
          << " used=" << used_mib << "MiB"
          << " free=" << free_mib << "MiB"
          << " total=" << total_mib << "MiB";
    const auto phase_text = phase.str();
    return on_progress(0, 0, phase_text.c_str());
}

void require_memory_diagnostic_sent(
    const core::ExecutionContext & execution,
    const HiggsTTSSession::StreamProgressCallback & on_progress,
    const char * stage) {
    if (!emit_memory_diagnostic(execution, on_progress, stage)) {
        throw std::runtime_error("generation cancelled");
    }
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
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    engine::debug::trace_log_scalar("higgs_tts.text_chunks", static_cast<int64_t>(chunk_requests.size()));
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        if (!chunk_request.text_input.has_value() || chunk_request.text_input->text.empty()) {
            continue;
        }
        runtime::append_audio_buffer(merged_audio, run_chunk(chunk_request));
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    return result;
}

runtime::TaskResult HiggsTTSSession::run_streaming(
    const runtime::TaskRequest & request,
    const AudioStreamCallback & on_audio,
    const StreamProgressCallback & on_progress,
    const CancelCallback & should_continue) {
    const auto wall_start = Clock::now();
    require_prepared("Higgs TTS streaming run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Higgs TTS requires text input");
    }
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    engine::debug::trace_log_scalar("higgs_tts.text_chunks", static_cast<int64_t>(chunk_requests.size()));
    runtime::AudioBuffer merged_audio;
    int64_t output_start_sample = 0;
    for (const auto & chunk_request : chunk_requests) {
        require_generation_alive(should_continue);
        if (!chunk_request.text_input.has_value() || chunk_request.text_input->text.empty()) {
            continue;
        }
        runtime::append_audio_buffer(
            merged_audio,
            run_chunk_streaming(chunk_request, on_audio, on_progress, output_start_sample, should_continue));
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    return result;
}

runtime::AudioBuffer HiggsTTSSession::run_chunk(const runtime::TaskRequest & request) {
    RuntimeCacheReleaseGuard runtime_cache_guard(generator_.get(), codec_decoder_.get());
    const bool keep_runtime_cache = bool_option(request, "keep_runtime_cache", false);
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
        const auto & reference_audio = *request.voice->speaker->audio;
        const auto key = reference_audio_key(reference_audio);
        const auto cache_path = reference_cache_path(request);
        if (cache_path.has_value()) {
            try {
                reference_delayed_codes = read_reference_cache(*cache_path, key);
                if (reference_delayed_codes.has_value()) {
                    set_in_memory_reference_cache(
                        *reference_delayed_codes,
                        key,
                        cached_reference_codes_,
                        cached_reference_fingerprint_,
                        cached_reference_sample_count_,
                        cached_reference_sample_rate_,
                        cached_reference_channels_,
                        has_cached_reference_);
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_hit", 1);
                }
            } catch (...) {
                reference_delayed_codes.reset();
            }
        }
        if (!reference_delayed_codes.has_value() &&
            has_cached_reference_ &&
            cached_reference_codes_.has_value() &&
            reference_key_matches(
                key,
                cached_reference_fingerprint_,
                cached_reference_sample_count_,
                cached_reference_sample_rate_,
                cached_reference_channels_)) {
            reference_delayed_codes = cached_reference_codes_;
            engine::debug::trace_log_scalar("higgs_tts.reference.cache_hit", 1);
        }
        if (!reference_delayed_codes.has_value()) {
            auto reference_raw_codes = codec_decoder_->encode_reference_audio(reference_audio);
            if (!keep_runtime_cache) {
                codec_decoder_->release_runtime_cache();
            }
            reference_delayed_codes = apply_delay_pattern(reference_raw_codes);
            engine::debug::trace_log_scalar("higgs_tts.reference.raw_code_frames", reference_raw_codes.frames);
            engine::debug::trace_log_scalar("higgs_tts.reference.delayed_code_rows", reference_delayed_codes->frames);
            set_in_memory_reference_cache(
                *reference_delayed_codes,
                key,
                cached_reference_codes_,
                cached_reference_fingerprint_,
                cached_reference_sample_count_,
                cached_reference_sample_rate_,
                cached_reference_channels_,
                has_cached_reference_);
            if (cache_path.has_value()) {
                try {
                    write_reference_cache(*cache_path, key, *reference_delayed_codes);
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_write", 1);
                } catch (...) {
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_write_failed", 1);
                }
            }
        }
    }
    const auto prompt = tokenizer_.build_prompt(
        request.text_input->text,
        reference_delayed_codes.has_value() ? reference_delayed_codes->frames : 0,
        reference_text);
    const auto codes = generator_->generate(
        prompt,
        generation,
        reference_delayed_codes.has_value() ? &*reference_delayed_codes : nullptr);
    if (!keep_runtime_cache) {
        generator_->release_runtime_cache();
    }
    engine::debug::trace_log_scalar("higgs_tts.delayed_code_rows", codes.delayed_codes.frames);
    engine::debug::trace_log_scalar("higgs_tts.raw_code_frames", codes.raw_codes.frames);
    auto audio = codec_decoder_->decode(codes.raw_codes);
    if (!keep_runtime_cache) {
        codec_decoder_->release_runtime_cache();
    }
    if (keep_runtime_cache) {
        runtime_cache_guard.dismiss();
    }
    return audio;
}

runtime::AudioBuffer HiggsTTSSession::run_chunk_streaming(
    const runtime::TaskRequest & request,
    const AudioStreamCallback & on_audio,
    const StreamProgressCallback & on_progress,
    int64_t & output_start_sample,
    const CancelCallback & should_continue) {
    RuntimeCacheReleaseGuard runtime_cache_guard(generator_.get(), codec_decoder_.get());
    const bool keep_runtime_cache = bool_option(request, "keep_runtime_cache", false);
    const std::string reference_text =
        request.options.count("reference_text") != 0 ? request.options.at("reference_text") : std::string{};
    const auto generation = generation_options_from_request(request);
    const int64_t first_stream_frames = positive_int_option(
        request,
        "first_stream_frames",
        kDefaultFirstStreamFrames);
    const int64_t stream_frames = positive_int_option(
        request,
        "stream_frames",
        kDefaultStreamFrames);
    const bool emit_stream_audio = bool_option(request, "emit_stream_audio_chunks", true);
    engine::debug::trace_log_scalar("higgs_tts.sampling.max_tokens", generation.max_tokens);
    engine::debug::trace_log_scalar("higgs_tts.sampling.temperature", generation.temperature);
    engine::debug::trace_log_scalar("higgs_tts.sampling.top_k", generation.top_k);
    engine::debug::trace_log_scalar("higgs_tts.sampling.top_p", generation.top_p);
    engine::debug::trace_log_scalar("higgs_tts.sampling.seed", generation.seed);
    engine::debug::trace_log_scalar("higgs_tts.streaming.first_frames", first_stream_frames);
    engine::debug::trace_log_scalar("higgs_tts.streaming.frames", stream_frames);
    engine::debug::trace_log_scalar("higgs_tts.streaming.emit_audio_chunks", emit_stream_audio ? 1 : 0);
    require_memory_diagnostic_sent(execution_context(), on_progress, "chunk_start");
    require_generation_alive(should_continue);
    std::optional<HiggsAudioCodeMatrix> reference_delayed_codes;
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        const auto & reference_audio = *request.voice->speaker->audio;
        const auto key = reference_audio_key(reference_audio);
        const auto cache_path = reference_cache_path(request);
        if (cache_path.has_value()) {
            try {
                reference_delayed_codes = read_reference_cache(*cache_path, key);
                if (reference_delayed_codes.has_value()) {
                    set_in_memory_reference_cache(
                        *reference_delayed_codes,
                        key,
                        cached_reference_codes_,
                        cached_reference_fingerprint_,
                        cached_reference_sample_count_,
                        cached_reference_sample_rate_,
                        cached_reference_channels_,
                        has_cached_reference_);
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_hit", 1);
                }
            } catch (...) {
                reference_delayed_codes.reset();
            }
        }
        if (!reference_delayed_codes.has_value() &&
            has_cached_reference_ &&
            cached_reference_codes_.has_value() &&
            reference_key_matches(
                key,
                cached_reference_fingerprint_,
                cached_reference_sample_count_,
                cached_reference_sample_rate_,
                cached_reference_channels_)) {
            reference_delayed_codes = cached_reference_codes_;
            engine::debug::trace_log_scalar("higgs_tts.reference.cache_hit", 1);
        }
        if (!reference_delayed_codes.has_value()) {
            require_memory_diagnostic_sent(execution_context(), on_progress, "before_reference_encode");
            require_generation_alive(should_continue);
            auto reference_raw_codes = codec_decoder_->encode_reference_audio(reference_audio);
            require_memory_diagnostic_sent(execution_context(), on_progress, "after_reference_encode");
            if (!keep_runtime_cache) {
                codec_decoder_->release_runtime_cache();
                require_memory_diagnostic_sent(execution_context(), on_progress, "after_reference_codec_release");
            }
            reference_delayed_codes = apply_delay_pattern(reference_raw_codes);
            engine::debug::trace_log_scalar("higgs_tts.reference.raw_code_frames", reference_raw_codes.frames);
            engine::debug::trace_log_scalar("higgs_tts.reference.delayed_code_rows", reference_delayed_codes->frames);
            set_in_memory_reference_cache(
                *reference_delayed_codes,
                key,
                cached_reference_codes_,
                cached_reference_fingerprint_,
                cached_reference_sample_count_,
                cached_reference_sample_rate_,
                cached_reference_channels_,
                has_cached_reference_);
            if (cache_path.has_value()) {
                try {
                    write_reference_cache(*cache_path, key, *reference_delayed_codes);
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_write", 1);
                } catch (...) {
                    engine::debug::trace_log_scalar("higgs_tts.reference.disk_cache_write_failed", 1);
                }
            }
        }
    }
    const auto prompt = tokenizer_.build_prompt(
        request.text_input->text,
        reference_delayed_codes.has_value() ? reference_delayed_codes->frames : 0,
        reference_text);
    require_memory_diagnostic_sent(execution_context(), on_progress, "after_prompt_build");
    require_generation_alive(should_continue);

    int64_t emitted_raw_frames = 0;
    int64_t emitted_samples = 0;
    int64_t last_progress_frame = -1;
    const HiggsTTSCodeStreamCallback code_stream = [&](const HiggsAudioCodeMatrix & delayed_codes, bool is_final) {
        if (delayed_codes.frames < delayed_codes.codebooks) {
            return true;
        }
        const int64_t available_raw_frames = delayed_codes.frames - (delayed_codes.codebooks - 1);
        if (available_raw_frames <= 0) {
            return true;
        }
        if (is_final ||
            last_progress_frame < 0 ||
            (available_raw_frames - last_progress_frame) >= 4 ||
            available_raw_frames >= generation.max_tokens) {
            last_progress_frame = available_raw_frames;
            if (!on_progress(
                    std::min<int64_t>(available_raw_frames, generation.max_tokens),
                    generation.max_tokens,
                    is_final ? "decoding" : "generating")) {
                return false;
            }
        }
        if (!emit_stream_audio) {
            return true;
        }
        const int64_t wanted_frames = emitted_raw_frames == 0 ? first_stream_frames : stream_frames;
        if (!is_final && (available_raw_frames - emitted_raw_frames) < wanted_frames) {
            return true;
        }

        if (!should_continue_generation(should_continue)) {
            return false;
        }
        if (emitted_raw_frames == 0 || is_final) {
            if (!emit_memory_diagnostic(execution_context(), on_progress, "before_stream_codec_decode")) {
                return false;
            }
        }
        const auto raw_prefix = reverse_delay_pattern(delayed_codes);
        auto audio_prefix = codec_decoder_->decode(raw_prefix);
        if (emitted_raw_frames == 0 || is_final) {
            if (!emit_memory_diagnostic(execution_context(), on_progress, "after_stream_codec_decode")) {
                return false;
            }
        }
        if (!keep_runtime_cache) {
            codec_decoder_->release_runtime_cache();
            if (emitted_raw_frames == 0 || is_final) {
                if (!emit_memory_diagnostic(execution_context(), on_progress, "after_stream_codec_release")) {
                    return false;
                }
            }
        }
        if (static_cast<int64_t>(audio_prefix.samples.size()) <= emitted_samples) {
            emitted_raw_frames = raw_prefix.frames;
            return true;
        }

        runtime::AudioBuffer chunk;
        chunk.sample_rate = audio_prefix.sample_rate;
        chunk.channels = audio_prefix.channels;
        chunk.samples.assign(
            audio_prefix.samples.begin() + emitted_samples,
            audio_prefix.samples.end());
        const bool keep_going = on_audio(chunk, output_start_sample + emitted_samples, is_final);
        emitted_samples = static_cast<int64_t>(audio_prefix.samples.size());
        emitted_raw_frames = raw_prefix.frames;
        return keep_going;
    };

    require_memory_diagnostic_sent(execution_context(), on_progress, "before_generator");
    const auto codes = generator_->generate(
        prompt,
        generation,
        reference_delayed_codes.has_value() ? &*reference_delayed_codes : nullptr,
        &code_stream,
        &should_continue);
    require_memory_diagnostic_sent(execution_context(), on_progress, "after_generator");
    if (!keep_runtime_cache) {
        generator_->release_runtime_cache();
        require_memory_diagnostic_sent(execution_context(), on_progress, "after_generator_release");
    }
    engine::debug::trace_log_scalar("higgs_tts.delayed_code_rows", codes.delayed_codes.frames);
    engine::debug::trace_log_scalar("higgs_tts.raw_code_frames", codes.raw_codes.frames);
    require_memory_diagnostic_sent(execution_context(), on_progress, "before_final_codec_decode");
    auto audio = codec_decoder_->decode(codes.raw_codes);
    require_memory_diagnostic_sent(execution_context(), on_progress, "after_final_codec_decode");
    if (!keep_runtime_cache) {
        codec_decoder_->release_runtime_cache();
        require_memory_diagnostic_sent(execution_context(), on_progress, "after_final_codec_release");
    }
    if (emit_stream_audio && static_cast<int64_t>(audio.samples.size()) > emitted_samples) {
        runtime::AudioBuffer chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.channels = audio.channels;
        chunk.samples.assign(audio.samples.begin() + emitted_samples, audio.samples.end());
        if (!on_audio(chunk, output_start_sample + emitted_samples, true)) {
            throw std::runtime_error("generation cancelled");
        }
    }
    output_start_sample += static_cast<int64_t>(audio.samples.size());
    if (keep_runtime_cache) {
        runtime_cache_guard.dismiss();
    }
    require_memory_diagnostic_sent(execution_context(), on_progress, "chunk_end");
    return audio;
}

}  // namespace engine::models::higgs_tts
