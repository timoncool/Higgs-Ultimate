#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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
            throw std::runtime_error("invalid Stable Audio --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    throw std::runtime_error("Stable Audio warmbench backend must be cuda or cpu");
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

std::string option_text(const engine::io::json::Value & value) {
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    if (value.is_array()) {
        std::string out;
        const auto & items = value.as_array();
        for (size_t i = 0; i < items.size(); ++i) {
            if (i != 0) {
                out += items[i].is_string() ? "|" : ",";
            }
            if (items[i].is_string()) {
                out += items[i].as_string();
            } else if (items[i].is_number()) {
                out += engine::io::json::stringify_number(items[i].as_number());
            } else if (items[i].is_bool()) {
                out += items[i].as_bool() ? "true" : "false";
            } else {
                throw std::runtime_error("Stable Audio warmbench option arrays must contain only strings, numbers, or bools");
            }
        }
        return out;
    }
    return value.as_string();
}

void set_optional_option(engine::runtime::TaskRequest & request, const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value != nullptr && !value->is_null()) {
        request.options[key] = option_text(*value);
    }
}

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

std::string required_prompt(const engine::io::json::Value & object) {
    const auto * value = object.find("prompt");
    if (value == nullptr || value->is_null()) {
        throw std::runtime_error("Stable Audio warmbench request missing prompt");
    }
    if (value->is_array()) {
        std::string out;
        const auto & items = value->as_array();
        for (size_t i = 0; i < items.size(); ++i) {
            if (!items[i].is_string()) {
                throw std::runtime_error("Stable Audio warmbench prompt arrays must contain only strings");
            }
            if (i != 0) {
                out += "|";
            }
            out += items[i].as_string();
        }
        return out;
    }
    return value->as_string();
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value & object) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{required_prompt(object), "en"};
    set_optional_option(request, object, "negative_prompt");
    set_optional_option(request, object, "duration");
    set_optional_option(request, object, "duration_padding_sec");
    set_optional_option(request, object, "steps");
    set_optional_option(request, object, "sampler_type");
    set_optional_option(request, object, "cfg_scale");
    set_optional_option(request, object, "apg_scale");
    set_optional_option(request, object, "seed");
    set_optional_option(request, object, "batch_size");
    set_optional_option(request, object, "chunked_decode");
    set_optional_option(request, object, "init_noise_level");
    set_optional_option(request, object, "inpaint_mask_start_seconds");
    set_optional_option(request, object, "inpaint_mask_end_seconds");
    set_optional_option(request, object, "init_audio_from_previous");
    set_optional_option(request, object, "inpaint_audio_from_previous");

    const std::string init_audio = optional_string(object, "init_audio");
    const std::string inpaint_audio = optional_string(object, "inpaint_audio");
    if (!init_audio.empty()) {
        request.audio_input = read_audio_buffer(init_audio);
        request.options["audio_input_kind"] = "init_audio";
    } else if (!inpaint_audio.empty()) {
        request.audio_input = read_audio_buffer(inpaint_audio);
        request.options["audio_input_kind"] = "inpaint_audio";
    }
    return request;
}

void attach_previous_audio(
    engine::runtime::TaskRequest & request,
    const std::vector<engine::runtime::AudioBuffer> & previous_audio) {
    auto attach = [&](const char * option, const char * kind) {
        const auto it = request.options.find(option);
        if (it == request.options.end()) {
            return false;
        }
        const size_t index = static_cast<size_t>(std::stoul(it->second));
        if (index >= previous_audio.size()) {
            throw std::runtime_error(std::string("Stable Audio warmbench previous audio index out of range: ") + option);
        }
        request.audio_input = previous_audio[index];
        request.options["audio_input_kind"] = kind;
        return true;
    };
    const bool init = attach("init_audio_from_previous", "init_audio");
    const bool inpaint = attach("inpaint_audio_from_previous", "inpaint_audio");
    if (init && inpaint) {
        throw std::runtime_error("Stable Audio warmbench request cannot use both init_audio_from_previous and inpaint_audio_from_previous");
    }
}

std::vector<engine::runtime::TaskRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("Stable Audio warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> requests;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item));
    }
    if (requests.empty()) {
        throw std::runtime_error("Stable Audio warmbench request sequence is empty");
    }
    return requests;
}

engine::io::json::Value number(double value) {
    return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
    return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Stable Audio warmbench received empty audio output");
    }
    double sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    float min_value = audio.samples.front();
    float max_value = audio.samples.front();
    for (const float sample : audio.samples) {
        sum += static_cast<double>(sample);
        abs_sum += std::abs(static_cast<double>(sample));
        sq_sum += static_cast<double>(sample) * static_cast<double>(sample);
        min_value = std::min(min_value, sample);
        max_value = std::max(max_value, sample);
    }
    const auto channels = std::max(1, audio.channels);
    const double frames = static_cast<double>(audio.samples.size() / static_cast<size_t>(channels));
    const double count = static_cast<double>(audio.samples.size());
    return engine::io::json::Value::make_object({
        {"sample_rate", number(static_cast<double>(audio.sample_rate))},
        {"channels", number(static_cast<double>(audio.channels))},
        {"samples", number(count)},
        {"frames", number(frames)},
        {"duration_sec", number(audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0)},
        {"sum", number(sum)},
        {"mean_abs", number(abs_sum / count)},
        {"rms", number(std::sqrt(sq_sum / count))},
        {"min", number(min_value)},
        {"max", number(max_value)},
    });
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    double wall_ms,
    const std::vector<std::filesystem::path> & audio_paths) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Stable Audio warmbench expected audio output");
    }
    std::vector<engine::io::json::Value> stems;
    if (!result.named_audio_outputs.empty()) {
        for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
            const auto & named = result.named_audio_outputs[i];
            engine::io::json::Value::Object stem{
                {"name", string(named.id)},
                {"summary", audio_summary_json(named.audio)},
            };
            if (i < audio_paths.size() && !audio_paths[i].empty()) {
                stem.emplace("audio", string(audio_paths[i].string()));
            }
            stems.push_back(engine::io::json::Value::make_object(std::move(stem)));
        }
    } else {
        engine::io::json::Value::Object stem{
            {"name", string("audio")},
            {"summary", audio_summary_json(*result.audio_output)},
        };
        if (!audio_paths.empty() && !audio_paths.front().empty()) {
            stem.emplace("audio", string(audio_paths.front().string()));
        }
        stems.push_back(engine::io::json::Value::make_object(std::move(stem)));
    }
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array(std::move(stems))},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/stable-audio-3-small-music");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path = arg_value(argc, argv, "--timing-file", "/tmp/stable_audio_warm_bench_timing.log");
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "stable_audio";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load_request);

        engine::runtime::TaskSpec task;
        task.task = engine::runtime::VoiceTaskKind::Tts;
        task.mode = engine::runtime::RunMode::Offline;
        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : parse_session_options(argc, argv)) {
            session_options.options[key] = value;
        }
        auto requests = parse_requests(request_sequence_json);
        auto session_base = model->create_task_session(task, session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("Stable Audio model did not create an offline voice task session");
        }
        engine::runtime::SessionPreparationRequest preparation;
        preparation.text = requests.front().text_input;
        preparation.options = requests.front().options;
        session->prepare(preparation);

        std::vector<engine::io::json::Value> steps;
        std::vector<std::string> timing_lines;
        std::vector<engine::runtime::AudioBuffer> previous_audio;
        timing_lines.push_back("stable_audio.backend " + backend_name);
        timing_lines.push_back("stable_audio.model_root " + model_path.string());

        for (int i = 0; i < warmup; ++i) {
            (void) session->run(requests.front());
        }

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            attach_previous_audio(requests[request_index], previous_audio);
            double total_ms = 0.0;
            engine::runtime::TaskResult last_result;
            for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                const auto start = Clock::now();
                last_result = session->run(requests[request_index]);
                const auto end = Clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
                total_ms += wall_ms;
                timing_lines.push_back("stable_audio.wall_ms " + engine::io::json::stringify_number(wall_ms));
            }
            const double avg_ms = total_ms / static_cast<double>(std::max(1, iterations));
            previous_audio.push_back(
                last_result.named_audio_outputs.empty()
                    ? *last_result.audio_output
                    : last_result.named_audio_outputs.front().audio);
            std::vector<std::filesystem::path> audio_paths;
            if (!output_dir.empty()) {
                std::filesystem::create_directories(output_dir);
                if (!last_result.named_audio_outputs.empty()) {
                    for (size_t i = 0; i < last_result.named_audio_outputs.size(); ++i) {
                        const auto & named = last_result.named_audio_outputs[i];
                        auto path = output_dir / (
                            "request_" + std::to_string(request_index) + "_" + named.id + ".wav");
                        engine::audio::write_pcm16_wav(
                            path,
                            named.audio.sample_rate,
                            named.audio.channels,
                            named.audio.samples);
                        audio_paths.push_back(std::move(path));
                    }
                } else {
                    auto path = output_dir / ("request_" + std::to_string(request_index) + ".wav");
                    engine::audio::write_pcm16_wav(
                        path,
                        last_result.audio_output->sample_rate,
                        last_result.audio_output->channels,
                        last_result.audio_output->samples);
                    audio_paths.push_back(std::move(path));
                }
            }
            steps.push_back(step_json(last_result, static_cast<int>(request_index), avg_ms, audio_paths));
            std::cout << "stable_audio.wall_ms=" << avg_ms << "\n";
        }

        if (!timing_path.empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
            std::ofstream timing(timing_path, std::ios::app);
            for (const auto & line : timing_lines) {
                timing << line << "\n";
            }
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("stable_audio")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "stable_audio_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
