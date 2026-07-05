#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/codec.h"
#include "engine/models/higgs_tts/generator.h"
#include "engine/models/higgs_tts/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::higgs_tts {

class HiggsTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    using AudioStreamCallback = std::function<bool(const runtime::AudioBuffer &, int64_t, bool)>;
    using StreamProgressCallback = std::function<bool(int64_t, int64_t, const char *)>;

    HiggsTTSSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const HiggsTTSAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::TaskResult run_streaming(
        const runtime::TaskRequest & request,
        const AudioStreamCallback & on_audio,
        const StreamProgressCallback & on_progress);

private:
    runtime::AudioBuffer run_chunk(const runtime::TaskRequest & request);
    runtime::AudioBuffer run_chunk_streaming(
        const runtime::TaskRequest & request,
        const AudioStreamCallback & on_audio,
        const StreamProgressCallback & on_progress,
        int64_t & output_start_sample);

    runtime::TaskSpec task_;
    std::shared_ptr<const HiggsTTSAssets> assets_;
    HiggsTTSTokenizer tokenizer_;
    std::unique_ptr<HiggsTTSGeneratorRuntime> generator_;
    std::unique_ptr<HiggsAudioCodecDecoderRuntime> codec_decoder_;
    std::optional<HiggsAudioCodeMatrix> cached_reference_codes_;
    uint64_t cached_reference_fingerprint_ = 0;
    std::size_t cached_reference_sample_count_ = 0;
    int cached_reference_sample_rate_ = 0;
    int cached_reference_channels_ = 0;
    bool has_cached_reference_ = false;
};

}  // namespace engine::models::higgs_tts
