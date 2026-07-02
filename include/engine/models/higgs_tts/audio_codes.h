#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::higgs_tts {

constexpr int32_t kHiggsAudioBocId = 1024;
constexpr int32_t kHiggsAudioEocId = 1025;
constexpr int32_t kHiggsAudioStopCode = -1;
constexpr int32_t kHiggsAudioPlaceholderId = -100;
constexpr int kHiggsAudioSampleRate = 24000;

struct HiggsAudioCodeMatrix {
    int64_t frames = 0;
    int64_t codebooks = 0;
    std::vector<int32_t> token_ids;
};

HiggsAudioCodeMatrix apply_delay_pattern(const HiggsAudioCodeMatrix & codes);
HiggsAudioCodeMatrix reverse_delay_pattern(const HiggsAudioCodeMatrix & delayed_codes);

}  // namespace engine::models::higgs_tts
