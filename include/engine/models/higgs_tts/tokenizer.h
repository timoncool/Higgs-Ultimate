#pragma once

#include "engine/models/higgs_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::higgs_tts {

struct HiggsTTSPrompt {
    std::string text;
    std::string reference_text;
    int64_t reference_audio_tokens = 0;
    std::vector<int32_t> input_ids;
};

class HiggsTTSTokenizer {
public:
    explicit HiggsTTSTokenizer(std::shared_ptr<const HiggsTTSAssets> assets);

    HiggsTTSPrompt build_prompt(
        const std::string & text,
        int64_t reference_audio_tokens = 0,
        const std::string & reference_text = "") const;
    std::vector<int32_t> encode_text(const std::string & text) const;

    int32_t tts_token_id() const noexcept;
    int32_t ref_audio_token_id() const noexcept;
    int32_t ref_text_token_id() const noexcept;
    int32_t text_token_id() const noexcept;
    int32_t audio_token_id() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::higgs_tts
