#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/higgs_tts/assets.h"
#include "engine/models/higgs_tts/codec.h"
#include "engine/models/higgs_tts/generator.h"
#include "engine/models/higgs_tts/tokenizer.h"

#include <memory>
#include <string>

namespace engine::models::higgs_tts {

class HiggsTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    HiggsTTSSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const HiggsTTSAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const HiggsTTSAssets> assets_;
    HiggsTTSTokenizer tokenizer_;
    std::unique_ptr<HiggsTTSGeneratorRuntime> generator_;
    std::unique_ptr<HiggsAudioCodecDecoderRuntime> codec_decoder_;
};

}  // namespace engine::models::higgs_tts
