#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/higgs_tts/assets.h"

#include <memory>

namespace engine::models::higgs_tts {

class HiggsTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    HiggsTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const HiggsTTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const HiggsTTSAssets> assets_;
};

std::unique_ptr<HiggsTTSLoadedModel> load_higgs_tts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_tts_loader();

}  // namespace engine::models::higgs_tts
