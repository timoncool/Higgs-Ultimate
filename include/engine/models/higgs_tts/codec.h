#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/audio_codes.h"

#include <cstddef>
#include <memory>

namespace engine::models::higgs_tts {

class HiggsAudioCodecDecoderRuntime {
public:
    struct Impl;

    HiggsAudioCodecDecoderRuntime(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~HiggsAudioCodecDecoderRuntime();

    HiggsAudioCodeMatrix encode_reference_audio(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer decode(const HiggsAudioCodeMatrix & raw_codes);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
