#include "audiocpp_api.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/higgs_tts/session.h"

#include "whisper.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace rt = engine::runtime;
namespace core = engine::core;
namespace higgs = engine::models::higgs_tts;

// ─── helpers ──────────────────────────────────────────────────────────────

std::string capture_exception_message() {
    try {
        throw;
    } catch (const std::exception & e) {
        return e.what();
    } catch (...) {
        return "unknown error";
    }
}

char * dup_cstr(const std::string & s) {
    if (s.empty()) return nullptr;
    const auto n = s.size();
    auto * p = static_cast<char *>(std::malloc(n + 1));
    if (p == nullptr) throw std::bad_alloc();
    std::memcpy(p, s.c_str(), n + 1);
    return p;
}

float * dup_samples(const std::vector<float> & v) {
    if (v.empty()) return nullptr;
    const auto bytes = v.size() * sizeof(float);
    auto * p = static_cast<float *>(std::malloc(bytes));
    if (p == nullptr) throw std::bad_alloc();
    std::memcpy(p, v.data(), bytes);
    return p;
}

void zero_result(audiocpp_audio_result * r) {
    r->sample_rate = 0;
    r->channels = 0;
    r->sample_count = 0;
    r->samples = nullptr;
    r->error = nullptr;
}

void fill_result_ok(audiocpp_audio_result * out, int sr, int ch, std::vector<float> & samples) {
    out->sample_rate = sr;
    out->channels = ch;
    out->sample_count = samples.size();
    out->samples = dup_samples(samples);
    out->error = nullptr;
}

void fill_result_err(audiocpp_audio_result * out, const std::string & msg) {
    out->sample_rate = 0;
    out->channels = 0;
    out->sample_count = 0;
    out->samples = nullptr;
    out->error = dup_cstr(msg);
}

// ─── option parsing ───────────────────────────────────────────────────────

std::string value_to_option_string(const engine::io::json::Value & v) {
    if (v.is_string()) return v.as_string();
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    if (v.is_number()) {
        std::ostringstream ss;
        ss.precision(std::numeric_limits<double>::max_digits10);
        ss << v.as_number();
        return ss.str();
    }
    return engine::io::json::stringify(v);
}

std::unordered_map<std::string, std::string> parse_options_json(const char * json) {
    std::unordered_map<std::string, std::string> result;
    if (json == nullptr || json[0] == '\0') return result;
    try {
        auto value = engine::io::json::parse(std::string(json));
        if (!value.is_object()) return result;
        for (const auto & [key, child] : value.as_object()) {
            result[key] = value_to_option_string(child);
        }
    } catch (...) {
    }
    return result;
}

// ─── backend config ───────────────────────────────────────────────────────

core::BackendType backend_enum_to_type(audiocpp_backend backend) {
    switch (backend) {
        case AUDIOCPP_BACKEND_CPU:    return core::BackendType::Cpu;
        case AUDIOCPP_BACKEND_CUDA:   return core::BackendType::Cuda;
        case AUDIOCPP_BACKEND_VULKAN: return core::BackendType::Vulkan;
        case AUDIOCPP_BACKEND_METAL:  return core::BackendType::Metal;
        default:                      return core::BackendType::BestAvailable;
    }
}

core::BackendConfig make_backend_config(audiocpp_backend backend, int32_t device, int32_t threads) {
    core::BackendConfig config = {};
    config.type = backend_enum_to_type(backend);
    config.device = device;
    config.threads = threads > 0 ? threads : 1;
    return config;
}

// ─── engine state ─────────────────────────────────────────────────────────

struct EngineState {
    std::mutex mutex;

    rt::ModelRegistry registry;
    std::unique_ptr<rt::ILoadedVoiceModel> loaded_model;
    std::unique_ptr<rt::IVoiceTaskSession> session;
    rt::IOfflineVoiceTaskSession * offline = nullptr;

    std::atomic<bool> model_loaded{false};
    std::atomic<bool> generating{false};
    std::atomic<bool> cancel_requested{false};

    std::string model_root;
    std::string weight_type;
    std::string family_name;
    std::string display_name;
    std::string last_error;

    audiocpp_progress_fn progress_fn = nullptr;
    audiocpp_audio_chunk_fn audio_chunk_fn = nullptr;
    void * progress_user_data = nullptr;
};

void emit_progress(EngineState & state, int32_t current, int32_t total, const char * phase) {
    if (state.progress_fn != nullptr) {
        state.progress_fn(current, total, phase, state.progress_user_data);
    }
}

void emit_audio_chunk(
    EngineState & state,
    const rt::AudioBuffer & audio,
    int64_t start_sample,
    bool is_final) {
    if (state.audio_chunk_fn != nullptr && !audio.samples.empty()) {
        state.audio_chunk_fn(
            audio.sample_rate,
            audio.channels,
            start_sample,
            audio.samples.data(),
            audio.samples.size(),
            is_final,
            state.progress_user_data);
    }
}

bool is_cancelled(EngineState & state) {
    return state.cancel_requested.load(std::memory_order_relaxed);
}

// ─── WAV reading ──────────────────────────────────────────────────────────

rt::AudioBuffer read_wav_buffer(const std::string & path) {
    auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    return rt::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
}

// ─── voice reference builder ──────────────────────────────────────────────

void fill_voice_reference(rt::TaskRequest & req, const char * ref_path, const char * ref_text) {
    if (ref_path == nullptr || ref_path[0] == '\0') return;

    rt::VoiceCondition voice;
    rt::VoiceReference ref;
    ref.audio = read_wav_buffer(ref_path);
    voice.speaker = std::move(ref);

    if (ref_text != nullptr && ref_text[0] != '\0') {
        rt::StyleCondition style;
        style.tags["ref_text"] = std::string(ref_text);
        voice.style = std::move(style);
    }
    req.voice = std::move(voice);
}

// ─── core generation runner ───────────────────────────────────────────────

audiocpp_status run_generation(
    EngineState & state,
    rt::TaskRequest & base_request,
    audiocpp_audio_result * out_result) {

    // Don't pre-chunk here — the session handles its own chunking internally
    // with voice consistency (same seed, reference reuse across chunks).
    // Pre-chunking here would break that by giving each chunk a new seed.
    emit_progress(state, 0, 1, "generating");

    rt::TaskResult result;
    try {
        result = state.offline->run(base_request);
    } catch (...) {
        state.last_error = capture_exception_message();
        fill_result_err(out_result, state.last_error);
        return AUDIOCPP_ERR_RUNTIME;
    }

    if (!result.audio_output.has_value()) {
        fill_result_err(out_result, "no audio output produced");
        return AUDIOCPP_ERR_RUNTIME;
    }

    emit_progress(state, 1, 1, "complete");

    fill_result_ok(out_result, result.audio_output->sample_rate, result.audio_output->channels,
                   result.audio_output->samples);
    return AUDIOCPP_OK;
}

audiocpp_status run_generation_streaming(
    EngineState & state,
    rt::TaskRequest & base_request,
    audiocpp_audio_result * out_result) {

    auto * higgs_session = dynamic_cast<higgs::HiggsTTSSession *>(state.offline);
    if (higgs_session == nullptr) {
        fill_result_err(out_result, "streaming is only supported by Higgs TTS sessions");
        return AUDIOCPP_ERR_UNSUPPORTED;
    }

    emit_progress(state, 0, 4, "preparing");
    bool emitted_first_audio = false;
    rt::TaskResult result;
    try {
        result = higgs_session->run_streaming(
            base_request,
            [&](const rt::AudioBuffer & chunk, int64_t start_sample, bool is_final) {
                if (is_cancelled(state)) {
                    return false;
                }
                if (!emitted_first_audio) {
                    emitted_first_audio = true;
                    emit_progress(state, 1, 4, "first audio");
                } else {
                    emit_progress(state, 2, 4, "streaming");
                }
                emit_audio_chunk(state, chunk, start_sample, is_final);
                return !is_cancelled(state);
            },
            [&](int64_t current, int64_t total, const char * phase) {
                if (is_cancelled(state)) {
                    return false;
                }
                emit_progress(
                    state,
                    static_cast<int32_t>(std::min<int64_t>(current, std::numeric_limits<int32_t>::max())),
                    static_cast<int32_t>(std::min<int64_t>(total, std::numeric_limits<int32_t>::max())),
                    phase);
                return !is_cancelled(state);
            });
    } catch (...) {
        state.last_error = capture_exception_message();
        if (is_cancelled(state) || state.last_error.find("cancelled") != std::string::npos) {
            fill_result_err(out_result, "generation cancelled");
            return AUDIOCPP_ERR_CANCELLED;
        }
        fill_result_err(out_result, state.last_error);
        return AUDIOCPP_ERR_RUNTIME;
    }

    if (!result.audio_output.has_value()) {
        fill_result_err(out_result, "no audio output produced");
        return AUDIOCPP_ERR_RUNTIME;
    }

    emit_progress(state, 3, 4, "finalizing");
    fill_result_ok(out_result, result.audio_output->sample_rate, result.audio_output->channels,
                   result.audio_output->samples);
    emit_progress(state, 4, 4, "complete");
    return AUDIOCPP_OK;
}

}  // namespace

struct audiocpp_engine {
    EngineState state;
};

extern "C" {

AUDIOCPP_API audiocpp_engine * audiocpp_create(void) {
    try {
        auto * engine = new audiocpp_engine();
        engine->state.registry = rt::make_default_registry();
        return engine;
    } catch (...) {
        return nullptr;
    }
}

AUDIOCPP_API void audiocpp_destroy(audiocpp_engine * engine) {
    try {
        delete engine;
    } catch (...) {
    }
}

AUDIOCPP_API audiocpp_status audiocpp_load_model(
    audiocpp_engine *    handle,
    const char *         model_root,
    audiocpp_backend     backend,
    int32_t              device,
    int32_t              threads,
    const char *         weight_type,
    const char *         session_options_json) {
    try {
        if (handle == nullptr || model_root == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        state.loaded_model.reset();
        state.session.reset();
        state.offline = nullptr;
        state.model_loaded.store(false, std::memory_order_relaxed);

        std::filesystem::path root(model_root);
        if (!std::filesystem::exists(root)) {
            state.last_error = "model directory does not exist: " + std::string(model_root);
            return AUDIOCPP_ERR_INVALID_PARAM;
        }

        rt::ModelLoadRequest load_req;
        load_req.model_path = root;

        state.loaded_model = state.registry.load(load_req);
        const auto & meta = state.loaded_model->metadata();
        state.family_name = meta.family;
        state.display_name = meta.family;
        if (!meta.variant.empty()) {
            state.display_name += " (" + meta.variant + ")";
        }

        rt::TaskSpec task;
        task.task = rt::VoiceTaskKind::Tts;
        task.mode = rt::RunMode::Offline;

        rt::SessionOptions opts;
        opts.backend = make_backend_config(backend, device, threads);
        if (weight_type != nullptr && weight_type[0] != '\0') {
            opts.options["weight_type"] = std::string(weight_type);
            state.weight_type = weight_type;
        }
        auto extra = parse_options_json(session_options_json);
        for (auto & [k, v] : extra) {
            opts.options[k] = std::move(v);
        }

        state.session = state.loaded_model->create_task_session(task, opts);
        state.offline = dynamic_cast<rt::IOfflineVoiceTaskSession *>(state.session.get());
        if (state.offline == nullptr) {
            state.last_error = "model does not support offline TTS sessions";
            return AUDIOCPP_ERR_UNSUPPORTED;
        }

        // The session requires prepare() before run() can be called.
        rt::SessionPreparationRequest prep_req;
        state.session->prepare(prep_req);

        state.model_root = model_root;
        state.model_loaded.store(true, std::memory_order_relaxed);

        return AUDIOCPP_OK;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.last_error = capture_exception_message();
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API void audiocpp_unload_model(audiocpp_engine * handle) {
    try {
        if (handle == nullptr) return;
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.offline = nullptr;
        state.session.reset();
        state.loaded_model.reset();
        state.model_loaded.store(false, std::memory_order_relaxed);
        state.model_root.clear();
        state.family_name.clear();
        state.display_name.clear();
        state.weight_type.clear();
    } catch (...) {
    }
}

AUDIOCPP_API bool audiocpp_is_model_loaded(const audiocpp_engine * handle) {
    if (handle == nullptr) return false;
    return handle->state.model_loaded.load(std::memory_order_relaxed);
}

AUDIOCPP_API bool audiocpp_is_generating(const audiocpp_engine * handle) {
    if (handle == nullptr) return false;
    return handle->state.generating.load(std::memory_order_relaxed);
}

AUDIOCPP_API void audiocpp_cancel(audiocpp_engine * handle) {
    if (handle == nullptr) return;
    handle->state.cancel_requested.store(true, std::memory_order_relaxed);
}

AUDIOCPP_API audiocpp_status audiocpp_get_model_info(
    const audiocpp_engine * handle,
    audiocpp_model_info *   out_info) {
    try {
        if (handle == nullptr || out_info == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        const auto & state = handle->state;
        out_info->family = state.family_name.c_str();
        out_info->display_name = state.display_name.c_str();
        out_info->weight_type = state.weight_type.c_str();
        out_info->model_root = state.model_root.c_str();
        return AUDIOCPP_OK;
    } catch (...) {
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_tts(
    audiocpp_engine *       handle,
    const char *            text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || text == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = nullptr;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        request.text_input = rt::Transcript{std::string(text), ""};
        request.options = parse_options_json(options_json);

        auto status = run_generation(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_voice_clone(
    audiocpp_engine *       handle,
    const char *            text,
    const char *            ref_audio_path,
    const char *            ref_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || text == nullptr || ref_audio_path == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = nullptr;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        request.text_input = rt::Transcript{std::string(text), ""};
        request.options = parse_options_json(options_json);
        fill_voice_reference(request, ref_audio_path, ref_text);

        auto status = run_generation(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_finish_sentence(
    audiocpp_engine *       handle,
    const char *            audio_path,
    const char *            continuation_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || audio_path == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = nullptr;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        if (continuation_text != nullptr && continuation_text[0] != '\0') {
            request.text_input = rt::Transcript{std::string(continuation_text), ""};
        }
        request.options = parse_options_json(options_json);
        request.options["route"] = "audio_continuation";

        rt::VoiceCondition voice;
        rt::VoiceReference ref;
        ref.audio = read_wav_buffer(audio_path);
        voice.speaker = std::move(ref);
        request.voice = std::move(voice);

        auto status = run_generation(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_tts_stream(
    audiocpp_engine *       handle,
    const char *            text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || text == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = audio_chunk;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        request.text_input = rt::Transcript{std::string(text), ""};
        request.options = parse_options_json(options_json);

        auto status = run_generation_streaming(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        state.audio_chunk_fn = nullptr;
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.audio_chunk_fn = nullptr;
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_voice_clone_stream(
    audiocpp_engine *       handle,
    const char *            text,
    const char *            ref_audio_path,
    const char *            ref_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || text == nullptr || ref_audio_path == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = audio_chunk;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        request.text_input = rt::Transcript{std::string(text), ""};
        request.options = parse_options_json(options_json);
        fill_voice_reference(request, ref_audio_path, ref_text);

        auto status = run_generation_streaming(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        state.audio_chunk_fn = nullptr;
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.audio_chunk_fn = nullptr;
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API audiocpp_status audiocpp_generate_finish_sentence_stream(
    audiocpp_engine *       handle,
    const char *            audio_path,
    const char *            continuation_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result) {
    zero_result(out_result);
    try {
        if (handle == nullptr || audio_path == nullptr || out_result == nullptr) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }
        auto & state = handle->state;
        std::lock_guard<std::mutex> lock(state.mutex);

        if (state.offline == nullptr) {
            fill_result_err(out_result, "model not loaded");
            return AUDIOCPP_ERR_NOT_LOADED;
        }

        state.progress_fn = progress;
        state.audio_chunk_fn = audio_chunk;
        state.progress_user_data = user_data;
        state.generating.store(true, std::memory_order_relaxed);
        state.cancel_requested.store(false, std::memory_order_relaxed);

        rt::TaskRequest request;
        if (continuation_text != nullptr && continuation_text[0] != '\0') {
            request.text_input = rt::Transcript{std::string(continuation_text), ""};
        }
        request.options = parse_options_json(options_json);
        request.options["route"] = "audio_continuation";

        rt::VoiceCondition voice;
        rt::VoiceReference ref;
        ref.audio = read_wav_buffer(audio_path);
        voice.speaker = std::move(ref);
        request.voice = std::move(voice);

        auto status = run_generation_streaming(state, request, out_result);

        state.generating.store(false, std::memory_order_relaxed);
        state.audio_chunk_fn = nullptr;
        return status;
    } catch (...) {
        if (handle != nullptr) {
            handle->state.generating.store(false, std::memory_order_relaxed);
            handle->state.audio_chunk_fn = nullptr;
            handle->state.last_error = capture_exception_message();
            fill_result_err(out_result, handle->state.last_error);
        }
        return AUDIOCPP_ERR_RUNTIME;
    }
}

AUDIOCPP_API void audiocpp_free_result(audiocpp_audio_result * result) {
    if (result == nullptr) return;
    if (result->samples != nullptr) {
        std::free(result->samples);
        result->samples = nullptr;
    }
    if (result->error != nullptr) {
        std::free(result->error);
        result->error = nullptr;
    }
    result->sample_count = 0;
    result->sample_rate = 0;
    result->channels = 0;
}

AUDIOCPP_API const char * audiocpp_last_error(const audiocpp_engine * handle) {
    if (handle == nullptr) return "null engine handle";
    return handle->state.last_error.c_str();
}

AUDIOCPP_API const char * audiocpp_version(void) {
    return "0.2.0";
}

AUDIOCPP_API audiocpp_status audiocpp_transcribe(
    audiocpp_engine *    handle,
    const char *         whisper_model_path,
    const char *         wav_path,
    const char *         language,
    char *               out_text,
    size_t               out_text_size) {
    try {
        if (handle == nullptr || whisper_model_path == nullptr || wav_path == nullptr
            || out_text == nullptr || out_text_size == 0) {
            return AUDIOCPP_ERR_INVALID_PARAM;
        }

        out_text[0] = '\0';

        // Read WAV file
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(wav_path));

        // Mix to mono f32
        std::vector<float> mono;
        mono.reserve(wav.samples.size() / std::max(1, wav.channels));
        for (size_t i = 0; i < wav.samples.size(); i += std::max(1, wav.channels)) {
            float sum = 0.0f;
            for (int c = 0; c < wav.channels && (i + c) < wav.samples.size(); ++c) {
                sum += wav.samples[i + c];
            }
            mono.push_back(sum / std::max(1, wav.channels));
        }

        // Resample to 16kHz (whisper requires exactly 16000 Hz)
        constexpr int kWhisperSampleRate = 16000;
        std::vector<float> pcmf32;
        if (wav.sample_rate == kWhisperSampleRate) {
            pcmf32 = std::move(mono);
        } else {
            const double ratio = static_cast<double>(kWhisperSampleRate) / wav.sample_rate;
            const size_t out_len = static_cast<size_t>(mono.size() * ratio);
            pcmf32.reserve(out_len);
            for (size_t i = 0; i < out_len; ++i) {
                const double src_pos = static_cast<double>(i) / ratio;
                const size_t src_idx = static_cast<size_t>(src_pos);
                const double frac = src_pos - src_idx;
                const size_t next_idx = std::min(src_idx + 1, mono.size() - 1);
                pcmf32.push_back(
                    static_cast<float>(mono[src_idx] * (1.0 - frac) + mono[next_idx] * frac));
            }
        }

        // Initialize whisper — load model (uses GPU via ggml)
        auto cparams = whisper_context_default_params();
        cparams.use_gpu = true;
        cparams.flash_attn = false;

        struct whisper_context * ctx = whisper_init_from_file_with_params(whisper_model_path, cparams);
        if (ctx == nullptr) {
            std::snprintf(out_text, out_text_size, "Failed to load whisper model: %s", whisper_model_path);
            return AUDIOCPP_ERR_RUNTIME;
        }

        // Transcribe
        auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.n_threads = static_cast<int>(std::min(8u, std::thread::hardware_concurrency()));
        wparams.language = (language != nullptr && language[0] != '\0') ? language : "auto";
        wparams.no_timestamps = true;
        wparams.print_progress = false;
        wparams.print_special = false;
        wparams.print_realtime = false;
        wparams.translate = false;

        int rc = whisper_full(ctx, wparams, pcmf32.data(), static_cast<int>(pcmf32.size()));
        if (rc != 0) {
            whisper_free(ctx);
            std::snprintf(out_text, out_text_size, "whisper_full failed (code %d)", rc);
            return AUDIOCPP_ERR_RUNTIME;
        }

        // Collect segments
        std::string text;
        const int n_seg = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_seg; ++i) {
            const char * seg = whisper_full_get_segment_text(ctx, i);
            if (seg != nullptr) {
                text += seg;
            }
        }

        // Free whisper immediately — releases all VRAM
        whisper_free(ctx);

        // Copy text to output buffer
        std::snprintf(out_text, out_text_size, "%s", text.c_str());
        return AUDIOCPP_OK;
    } catch (...) {
        return AUDIOCPP_ERR_RUNTIME;
    }
}

}  // extern "C"
