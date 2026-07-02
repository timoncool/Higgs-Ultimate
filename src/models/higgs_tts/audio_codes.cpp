#include "engine/models/higgs_tts/audio_codes.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace engine::models::higgs_tts {
namespace {

void validate_matrix(const HiggsAudioCodeMatrix & codes, const char * role) {
    if (codes.frames <= 0 || codes.codebooks <= 0) {
        throw std::runtime_error(std::string("Higgs TTS ") + role + " code matrix must be non-empty");
    }
    if (static_cast<int64_t>(codes.token_ids.size()) != codes.frames * codes.codebooks) {
        throw std::runtime_error(std::string("Higgs TTS ") + role + " code matrix shape is invalid");
    }
}

int32_t at(const HiggsAudioCodeMatrix & codes, int64_t frame, int64_t codebook) {
    return codes.token_ids[static_cast<size_t>(frame * codes.codebooks + codebook)];
}

void set(HiggsAudioCodeMatrix & codes, int64_t frame, int64_t codebook, int32_t value) {
    codes.token_ids[static_cast<size_t>(frame * codes.codebooks + codebook)] = value;
}

}  // namespace

HiggsAudioCodeMatrix apply_delay_pattern(const HiggsAudioCodeMatrix & codes) {
    validate_matrix(codes, "raw");
    HiggsAudioCodeMatrix delayed;
    delayed.frames = codes.frames + codes.codebooks - 1;
    delayed.codebooks = codes.codebooks;
    delayed.token_ids.assign(
        static_cast<size_t>(delayed.frames * delayed.codebooks),
        kHiggsAudioEocId);
    for (int64_t codebook = 0; codebook < codes.codebooks; ++codebook) {
        for (int64_t frame = 0; frame < codebook; ++frame) {
            set(delayed, frame, codebook, kHiggsAudioBocId);
        }
        for (int64_t frame = 0; frame < codes.frames; ++frame) {
            set(delayed, codebook + frame, codebook, at(codes, frame, codebook));
        }
    }
    return delayed;
}

HiggsAudioCodeMatrix reverse_delay_pattern(const HiggsAudioCodeMatrix & delayed_codes) {
    validate_matrix(delayed_codes, "delayed");
    const int64_t raw_frames = delayed_codes.frames - (delayed_codes.codebooks - 1);
    if (raw_frames <= 0) {
        throw std::runtime_error("Higgs TTS delayed code matrix is shorter than its codebook delay");
    }
    HiggsAudioCodeMatrix raw;
    raw.frames = raw_frames;
    raw.codebooks = delayed_codes.codebooks;
    raw.token_ids.assign(static_cast<size_t>(raw.frames * raw.codebooks), 0);
    for (int64_t codebook = 0; codebook < raw.codebooks; ++codebook) {
        for (int64_t frame = 0; frame < raw.frames; ++frame) {
            set(raw, frame, codebook, at(delayed_codes, codebook + frame, codebook));
        }
    }
    return raw;
}

}  // namespace engine::models::higgs_tts
