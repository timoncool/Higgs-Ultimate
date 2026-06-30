#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

std::vector<std::pair<std::string, std::string>> parse_session_options(int argc, char ** argv) {
    std::vector<std::pair<std::string, std::string>> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--session-option") {
            continue;
        }
        const std::string option = argv[i + 1];
        const size_t eq = option.find('=');
        if (eq == std::string::npos || eq == 0) {
            throw std::runtime_error("invalid ACE-Step --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported ACE-Step warmbench backend: " + value);
}

void set_env_required(const char * key, const std::string & value) {
#if defined(_WIN32)
    if (_putenv_s(key, value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#else
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#endif
}

std::string required_string_field(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        throw std::runtime_error("ACE-Step warmbench request missing required string field: " + key);
    }
    return value->as_string();
}

std::string option_text(const engine::io::json::Value & value) {
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    if (value.is_array()) {
        return engine::io::json::stringify(value);
    }
    return value.as_string();
}

void set_required_option_from_request(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & source,
    const std::string & target) {
    const auto * value = object.find(source);
    if (value == nullptr || value->is_null()) {
        throw std::runtime_error("ACE-Step warmbench request missing required field: " + source);
    }
    request.options[target] = option_text(*value);
}

void set_optional_option_from_request(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & source,
    const std::string & target) {
    const auto * value = object.find(source);
    if (value != nullptr && !value->is_null()) {
        request.options[target] = option_text(*value);
    }
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value & object) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{
        required_string_field(object, "caption"),
        required_string_field(object, "vocal_language"),
    };
    set_required_option_from_request(request, object, "lyrics", "lyrics");
    set_optional_option_from_request(request, object, "instruction", "instruction");
    set_optional_option_from_request(request, object, "negative_prompt", "negative_prompt");
    set_optional_option_from_request(request, object, "audio_codes", "audio_codes");
    set_required_option_from_request(request, object, "duration", "duration_seconds");
    set_required_option_from_request(request, object, "thinking", "thinking");
    set_required_option_from_request(request, object, "inference_steps", "num_inference_steps");
    set_required_option_from_request(request, object, "guidance_scale", "guidance_scale");
    set_optional_option_from_request(request, object, "use_adg", "use_adg");
    set_required_option_from_request(request, object, "seed", "seed");
    set_optional_option_from_request(request, object, "task_type", "task_type");
    set_optional_option_from_request(request, object, "track_name", "track_name");
    set_optional_option_from_request(request, object, "complete_track_classes", "complete_track_classes");
    set_optional_option_from_request(request, object, "bpm", "bpm");
    set_optional_option_from_request(request, object, "keyscale", "keyscale");
    set_optional_option_from_request(request, object, "timesignature", "timesignature");
    set_optional_option_from_request(request, object, "chunk_mask_mode", "chunk_mask_mode");
    set_optional_option_from_request(request, object, "repainting_start", "repainting_start");
    set_optional_option_from_request(request, object, "repainting_end", "repainting_end");
    set_optional_option_from_request(request, object, "use_cot_metas", "use_cot_metas");
    set_optional_option_from_request(request, object, "use_cot_caption", "use_cot_caption");
    set_optional_option_from_request(request, object, "use_cot_language", "use_cot_language");
    set_optional_option_from_request(request, object, "lm_temperature", "lm_temperature");
    set_optional_option_from_request(request, object, "lm_cfg_scale", "lm_cfg_scale");
    set_optional_option_from_request(request, object, "lm_top_k", "lm_top_k");
    set_optional_option_from_request(request, object, "lm_top_p", "lm_top_p");
    set_optional_option_from_request(request, object, "lm_repetition_penalty", "lm_repetition_penalty");
    set_optional_option_from_request(request, object, "shift", "shift");
    set_optional_option_from_request(request, object, "infer_method", "infer_method");
    set_optional_option_from_request(request, object, "timesteps", "timesteps");
    set_optional_option_from_request(request, object, "audio_cover_strength", "audio_cover_strength");
    set_optional_option_from_request(request, object, "cover_noise_strength", "cover_noise_strength");
    set_optional_option_from_request(request, object, "retake_seed", "retake_seed");
    set_optional_option_from_request(request, object, "retake_variance", "retake_variance");
    set_optional_option_from_request(request, object, "sampler_mode", "sampler_mode");
    set_optional_option_from_request(request, object, "velocity_norm_threshold", "velocity_norm_threshold");
    set_optional_option_from_request(request, object, "velocity_ema_factor", "velocity_ema_factor");
    set_optional_option_from_request(request, object, "cfg_interval_start", "cfg_interval_start");
    set_optional_option_from_request(request, object, "cfg_interval_end", "cfg_interval_end");
    set_optional_option_from_request(request, object, "dcw_enabled", "dcw_enabled");
    set_optional_option_from_request(request, object, "dcw_mode", "dcw_mode");
    set_optional_option_from_request(request, object, "dcw_scaler", "dcw_scaler");
    set_optional_option_from_request(request, object, "dcw_high_scaler", "dcw_high_scaler");
    set_optional_option_from_request(request, object, "dcw_wavelet", "dcw_wavelet");
    set_optional_option_from_request(request, object, "repaint_mode", "repaint_mode");
    set_optional_option_from_request(request, object, "repaint_strength", "repaint_strength");
    set_optional_option_from_request(request, object, "repaint_crossfade_frames", "repaint_crossfade_frames");
    set_optional_option_from_request(request, object, "repaint_injection_ratio", "repaint_injection_ratio");
    set_optional_option_from_request(request, object, "flow_edit_morph", "flow_edit_morph");
    if (const auto * src_audio = object.find("src_audio"); src_audio != nullptr && !src_audio->is_null()) {
        const auto wav = engine::audio::read_wav_f32(src_audio->as_string());
        request.audio_input = engine::runtime::AudioBuffer{
            wav.sample_rate,
            wav.channels,
            wav.samples,
        };
    }
    return request;
}

std::vector<engine::runtime::TaskRequest> parse_requests(
    const std::string & request_sequence_json,
    const std::string & noise_file) {
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> out;
    for (const auto & item : root.as_array()) {
        auto request = make_request(item);
        if (!noise_file.empty()) {
            request.options["noise_file"] = noise_file;
        }
        out.push_back(std::move(request));
    }
    return out;
}

engine::io::json::Value number(double value) {
    return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
    return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    const int64_t channels = std::max<int64_t>(1, audio.channels);
    return engine::io::json::Value::make_object({
        {"sample_rate", number(static_cast<double>(audio.sample_rate))},
        {"channels", number(static_cast<double>(audio.channels))},
        {"samples", number(static_cast<double>(audio.samples.size()))},
        {"frames", number(static_cast<double>(audio.samples.size() / static_cast<size_t>(channels)))},
    });
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("ACE-Step warmbench expected audio_output");
    }
    engine::io::json::Value::Object stem{
        {"name", string("mix")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", string(audio_path.string()));
    }
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/Ace-Step1.5");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 1);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::string noise_file = arg_value(argc, argv, "--noise-file", "");
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path =
            arg_value(argc, argv, "--timing-file", "/tmp/ace_step_warm_bench_timing.log");
        if (request_sequence_json.empty()) {
            throw std::runtime_error("ACE-Step warmbench requires --request-sequence-json");
        }
        const auto session_option_overrides = parse_session_options(argc, argv);

        set_env_required("ENGINE_TRACE_ENABLED", "0");
        set_env_required("ENGINE_TIMING_ENABLED", "1");
        set_env_required("ENGINE_TIMING_FILE", timing_path.string());
        for (const auto & [key, value] : session_option_overrides) {
            if (key == "ace_step.dit_tensor_dump_dir") {
                set_env_required("ENGINE_ACE_STEP_DIT_TENSOR_DUMP_DIR", value);
            } else if (key == "ace_step.dit_tensor_compare_dir") {
                set_env_required("ENGINE_ACE_STEP_DIT_TENSOR_COMPARE_DIR", value);
            } else if (key == "ace_step.dit_tensor_compare_result") {
                set_env_required("ENGINE_ACE_STEP_DIT_TENSOR_COMPARE_RESULT", value);
            } else if (key == "ace_step.dit_tensor_compare_threshold") {
                set_env_required("ENGINE_ACE_STEP_DIT_TENSOR_COMPARE_THRESHOLD", value);
            } else if (key == "ace_step.trace_enabled") {
                set_env_required("ENGINE_TRACE_ENABLED", value);
            } else if (key == "ace_step.trace_file") {
                set_env_required("ENGINE_TRACE_FILE", value);
            }
        }
        engine::runtime::TaskSpec task{engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline};
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "ace_step";
        for (const auto & [key, value] : session_option_overrides) {
            load_request.options.insert_or_assign(key, value);
        }
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions options;
        options.backend.type = parse_backend(backend_name);
        options.backend.device = device;
        options.backend.threads = threads;
        for (const auto & [key, value] : session_option_overrides) {
            options.options.insert_or_assign(key, value);
        }
        auto session_base = model->create_task_session(task, options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("loaded ACE-Step session is not an offline TTS session");
        }
        session->prepare({});

        const auto requests = parse_requests(request_sequence_json, noise_file);
        if (requests.empty()) {
            throw std::runtime_error("ACE-Step warmbench request sequence is empty");
        }
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(requests.front());
        }

        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        engine::io::json::Value::Array steps;
        steps.reserve(requests.size());
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            engine::runtime::TaskResult last_result;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                last_result = session->run(requests[request_index]);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            const double wall_ms = total_ms / static_cast<double>(std::max(1, iterations));
            std::filesystem::path audio_path;
            if (!output_dir.empty()) {
                audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
                const auto & audio = *last_result.audio_output;
                engine::audio::write_pcm16_wav(audio_path, audio.sample_rate, audio.channels, audio.samples);
            }
            std::cout << "ace_step.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(last_result, static_cast<int>(request_index), wall_ms, audio_path));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("ace_step")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "ace_step_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
