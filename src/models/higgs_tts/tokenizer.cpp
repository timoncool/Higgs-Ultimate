#include "engine/models/higgs_tts/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/higgs_tts/audio_codes.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

int32_t require_token_id(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & token,
    const char * role) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("Higgs TTS tokenizer missing ") + role + " token: " + token);
    }
    return *id;
}

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_tokenizer(const HiggsTTSAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_json_path = assets.paths.tokenizer_json_path;
    spec.tokenizer_config_path = assets.paths.tokenizer_config_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    return engine::tokenizers::load_llama_bpe_tokenizer(spec);
}

}  // namespace

struct HiggsTTSTokenizer::Impl {
    explicit Impl(const HiggsTTSAssets & assets)
        : tokenizer(load_tokenizer(assets)),
          tts_id(require_token_id(*tokenizer, "<|tts|>", "tts")),
          ref_audio_id(require_token_id(*tokenizer, "<|ref_audio|>", "ref_audio")),
          text_id(require_token_id(*tokenizer, "<|text|>", "text")),
          audio_id(require_token_id(*tokenizer, "<|audio|>", "audio")) {
        if (const auto id = tokenizer->find_token_id("<|ref_text|>")) {
            ref_text_id = *id;
        }
    }

    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t tts_id = 0;
    int32_t ref_audio_id = 0;
    int32_t ref_text_id = -1;
    int32_t text_id = 0;
    int32_t audio_id = 0;
};

HiggsTTSTokenizer::HiggsTTSTokenizer(std::shared_ptr<const HiggsTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs TTS tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(*assets);
}

HiggsTTSPrompt HiggsTTSTokenizer::build_prompt(
    const std::string & text,
    int64_t reference_audio_tokens,
    const std::string & reference_text) const {
    if (text.empty()) {
        throw std::runtime_error("Higgs TTS prompt text must not be empty");
    }
    if (reference_audio_tokens < 0) {
        throw std::runtime_error("Higgs TTS reference_audio_tokens must be non-negative");
    }
    HiggsTTSPrompt prompt;
    prompt.text = text;
    prompt.reference_text = reference_text;
    prompt.reference_audio_tokens = reference_audio_tokens;
    prompt.input_ids.push_back(impl_->tts_id);
    if (!reference_text.empty() && reference_audio_tokens > 0 && impl_->ref_text_id >= 0) {
        prompt.input_ids.push_back(impl_->ref_text_id);
        auto ids = impl_->tokenizer->encode(reference_text);
        prompt.input_ids.insert(prompt.input_ids.end(), ids.begin(), ids.end());
    }
    if (reference_audio_tokens > 0) {
        prompt.input_ids.push_back(impl_->ref_audio_id);
        prompt.input_ids.insert(
            prompt.input_ids.end(),
            static_cast<size_t>(reference_audio_tokens),
            kHiggsAudioPlaceholderId);
    }
    prompt.input_ids.push_back(impl_->text_id);
    auto text_ids = impl_->tokenizer->encode(text);
    prompt.input_ids.insert(prompt.input_ids.end(), text_ids.begin(), text_ids.end());
    prompt.input_ids.push_back(impl_->audio_id);
    return prompt;
}

std::vector<int32_t> HiggsTTSTokenizer::encode_text(const std::string & text) const {
    return impl_->tokenizer->encode(text);
}

int32_t HiggsTTSTokenizer::tts_token_id() const noexcept {
    return impl_->tts_id;
}

int32_t HiggsTTSTokenizer::ref_audio_token_id() const noexcept {
    return impl_->ref_audio_id;
}

int32_t HiggsTTSTokenizer::ref_text_token_id() const noexcept {
    return impl_->ref_text_id;
}

int32_t HiggsTTSTokenizer::text_token_id() const noexcept {
    return impl_->text_id;
}

int32_t HiggsTTSTokenizer::audio_token_id() const noexcept {
    return impl_->audio_id;
}

}  // namespace engine::models::higgs_tts
