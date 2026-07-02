#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/audio_codes.h"
#include "engine/models/higgs_tts/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace engine::models::higgs_tts {

struct HiggsTTSGenerationOptions {
    int64_t max_tokens = 1024;
    int top_k = 30;
    float top_p = 0.8F;
    float temperature = 0.8F;
    uint32_t seed = 0;
    bool do_sample = true;
};

struct HiggsTTSGeneratedCodes {
    HiggsAudioCodeMatrix delayed_codes;
    HiggsAudioCodeMatrix raw_codes;
};

class HiggsTTSGeneratorRuntime {
public:
    struct Impl;

    HiggsTTSGeneratorRuntime(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~HiggsTTSGeneratorRuntime();

    HiggsTTSGeneratedCodes generate(
        const HiggsTTSPrompt & prompt,
        const HiggsTTSGenerationOptions & options,
        const HiggsAudioCodeMatrix * reference_delayed_codes = nullptr);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_tts
