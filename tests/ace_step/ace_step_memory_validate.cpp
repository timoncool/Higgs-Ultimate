#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/vae_decoder.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::runtime::TaskRequest make_request(
    const std::string & prompt,
    const std::string & lyrics,
    float duration_seconds,
    int64_t num_inference_steps,
    uint32_t seed) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{prompt, "en"};
    request.options["thinking"] = "true";
    request.options["lyrics"] = lyrics;
    request.options["duration_seconds"] = std::to_string(duration_seconds);
    request.options["num_inference_steps"] = std::to_string(num_inference_steps);
    request.options["seed"] = std::to_string(seed);
    return request;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: ace_step_memory_validate <model_dir> [cpu|cuda|vulkan|best] [threads] "
                     "[--decode-latents-file path --output-wav path]\n";
        return 2;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::string backend_name = argc >= 3 ? argv[2] : "cuda";
    const int threads = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 1;
    const std::string decode_latents_file = arg_value(argc, argv, "--decode-latents-file");
    const std::string output_wav = arg_value(argc, argv, "--output-wav");

    engine::core::BackendType backend_type = engine::core::BackendType::Cuda;
    if (backend_name == "cpu") {
        backend_type = engine::core::BackendType::Cpu;
    } else if (backend_name == "vulkan") {
        backend_type = engine::core::BackendType::Vulkan;
    } else if (backend_name == "best") {
        backend_type = engine::core::BackendType::BestAvailable;
    } else if (backend_name != "cuda") {
        throw std::runtime_error("backend must be cpu, cuda, vulkan, or best");
    }

    if (!decode_latents_file.empty()) {
        if (output_wav.empty()) {
            throw std::runtime_error("--output-wav is required with --decode-latents-file");
        }
        auto assets = engine::models::ace_step::load_ace_step_assets(model_dir);
        engine::core::BackendConfig backend_config;
        backend_config.type = backend_type;
        backend_config.threads = threads;
        engine::core::ExecutionContext execution(backend_config);
        engine::models::ace_step::AceStepVAEDecoderRuntime vae(
            assets,
            execution,
            engine::assets::TensorStorageType::F32);
        const auto values = engine::io::read_f32_file(decode_latents_file);
        const int64_t channels = assets->config.vae.decoder_input_channels;
        if (channels <= 0 || values.size() % static_cast<size_t>(channels) != 0) {
            throw std::runtime_error("invalid ACE-Step latent file shape for VAE decode");
        }
        engine::models::ace_step::AceStepLatents latents;
        latents.channels = channels;
        latents.frames = static_cast<int64_t>(values.size() / static_cast<size_t>(channels));
        latents.values = values;
        const auto audio = vae.decode(latents);
        engine::audio::write_pcm16_wav(output_wav, audio.sample_rate, audio.channels, audio.samples);
        std::cout << "decode.samples=" << audio.samples.size() << "\n";
        return 0;
    }

    engine::runtime::TaskSpec task{engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline};
    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_dir;
    load_request.family_hint = "ace_step";
    auto model = registry.load(load_request);

    engine::runtime::SessionOptions options;
    options.backend.type = backend_type;
    options.backend.threads = threads;

    auto session_base = model->create_task_session(task, options);
    auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
    if (session == nullptr) {
        throw std::runtime_error("loaded ACE-Step session is not an offline TTS session");
    }
    session->prepare({});

    const auto request1 = make_request(
        "warm analog synth pop with female vocals",
        "[verse]\nwe build the night from neon air\n[chorus]\nturn the city into sound",
        10.0F,
        2,
        1234);
    const auto result1 = session->run(request1);
    if (!result1.audio_output.has_value()) {
        throw std::runtime_error("ACE-Step validation request 1 produced no audio");
    }
    std::cout << "request1.samples=" << result1.audio_output->samples.size() << "\n";

    const auto request2 = make_request(
        "soft piano ballad with intimate vocals",
        "[verse]\nslow light on the window glass\n[chorus]\nhold the quiet while it lasts",
        5.0F,
        2,
        1234);
    const auto result2 = session->run(request2);
    if (!result2.audio_output.has_value()) {
        throw std::runtime_error("ACE-Step validation request 2 produced no audio");
    }
    std::cout << "request2.samples=" << result2.audio_output->samples.size() << "\n";
    return 0;
}
