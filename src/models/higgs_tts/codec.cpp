#include "engine/models/higgs_tts/codec.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

using Clock = std::chrono::steady_clock;
namespace modules = engine::modules;
namespace json = engine::io::json;

constexpr const char * kCodecPrefix = "tied.embedding.modality_embeddings.0.model.";
constexpr int64_t kCodecHiddenSize = 1024;
constexpr int64_t kCodecCodebookDim = 64;
constexpr int64_t kCodecCodebookSize = 1024;
constexpr int64_t kCodecAcousticHiddenSize = 256;
constexpr int64_t kCodecDecoderHiddenSize = 1024;
constexpr int64_t kCodecAcousticEncoderHiddenSize = 64;
constexpr int64_t kCodecSemanticHiddenSize = 768;
constexpr int64_t kCodecSemanticIntermediateSize = 3072;
constexpr int64_t kCodecSemanticHeads = 12;
constexpr int64_t kCodecSemanticLayers = 12;
constexpr int64_t kCodecDecodeFallbackFrames = 256;
constexpr int64_t kCodecDecodeFallbackOverlapFrames = 8;
constexpr int64_t kCodecDecodeFallbackMinFrames = 1;
constexpr int64_t kCodecDecodeFallbackMaxFrames = 1024;
constexpr int64_t kCodecDecodeFallbackMaxOverlapFrames = 64;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct HiggsAudioCodecConfig {
    struct SemanticModelConfig {
        int64_t hidden_size = kCodecSemanticHiddenSize;
        int64_t intermediate_size = kCodecSemanticIntermediateSize;
        int64_t num_attention_heads = kCodecSemanticHeads;
        int64_t num_hidden_layers = kCodecSemanticLayers;
        int64_t num_conv_pos_embeddings = 128;
        int64_t num_conv_pos_embedding_groups = 16;
        float layer_norm_eps = 1.0e-5F;
        bool feat_proj_layer_norm = true;
        bool do_stable_layer_norm = false;
        std::vector<int64_t> conv_dim = {512, 512, 512, 512, 512, 512, 512};
        std::vector<int64_t> conv_kernel = {10, 3, 3, 3, 3, 2, 2};
        std::vector<int64_t> conv_stride = {5, 2, 2, 2, 2, 2, 2};
    };

    int sample_rate = kHiggsAudioSampleRate;
    int semantic_sample_rate = 16000;
    int64_t downsample_factor = 320;
    int64_t hop_length = 960;
    int64_t codebook_size = kCodecCodebookSize;
    int64_t codebook_dim = kCodecCodebookDim;
    int64_t hidden_size = kCodecHiddenSize;
    int64_t acoustic_hidden_size = kCodecAcousticHiddenSize;
    int64_t acoustic_encoder_hidden_size = kCodecAcousticEncoderHiddenSize;
    int64_t decoder_hidden_size = kCodecDecoderHiddenSize;
    int64_t kernel_size = 3;
    int64_t unit_kernel_size = 3;
    std::vector<int64_t> downsampling_ratios = {8, 5, 4, 2, 3};
    std::vector<int64_t> upsampling_ratios = {8, 5, 4, 2, 3};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> block_dilations = {1, 1};
    SemanticModelConfig semantic;
};

struct SnakeActivationWeights {
    core::TensorValue alpha;
};

struct FlatConv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel_size = 0;
};

struct FlatConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
};

struct ResidualUnitWeights {
    SnakeActivationWeights snake1;
    FlatConv1dWeights conv1;
    SnakeActivationWeights snake2;
    FlatConv1dWeights conv2;
};

struct SemanticEncoderBlockWeights {
    std::vector<FlatConv1dWeights> residual_conv1;
    std::vector<FlatConv1dWeights> residual_conv2;
    FlatConv1dWeights conv;
};

struct AcousticEncoderBlockWeights {
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
    ResidualUnitWeights res3;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv;
};

struct AcousticDecoderBlockWeights {
    SnakeActivationWeights snake;
    FlatConvTranspose1dWeights conv_t;
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
    ResidualUnitWeights res3;
};

struct AcousticDecoderWeights {
    FlatConv1dWeights conv1;
    std::vector<AcousticDecoderBlockWeights> blocks;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv2;
};

struct HubertFeatureExtractorWeights {
    std::vector<FlatConv1dWeights> convs;
    modules::NormWeights first_group_norm;
};

struct HubertFeatureProjectionWeights {
    modules::NormWeights layer_norm;
    modules::LinearWeights projection;
};

struct HubertPositionConvGroupWeights {
    FlatConv1dWeights conv;
};

struct HubertEncoderLayerWeights {
    modules::AttentionWeights attention;
    modules::NormWeights layer_norm;
    modules::FeedForwardWeights feed_forward;
    modules::NormWeights final_layer_norm;
};

struct SemanticEncoderWeights {
    FlatConv1dWeights conv;
    std::vector<SemanticEncoderBlockWeights> blocks;
};

struct AcousticEncoderWeights {
    FlatConv1dWeights conv1;
    std::vector<AcousticEncoderBlockWeights> blocks;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv2;
};

struct QuantizerWeights {
    modules::LinearWeights project_in;
    modules::LinearWeights score;
    core::TensorValue codebook;
    modules::LinearWeights project_out;
};

struct HiggsAudioCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    HubertFeatureExtractorWeights feature_extractor;
    HubertFeatureProjectionWeights feature_projection;
    std::vector<HubertPositionConvGroupWeights> positional_conv_groups;
    modules::NormWeights encoder_input_layer_norm;
    std::vector<HubertEncoderLayerWeights> hubert_layers;
    SemanticEncoderWeights semantic_encoder;
    AcousticEncoderWeights acoustic_encoder;
    modules::LinearWeights fc;
    std::vector<QuantizerWeights> quantizers;
    modules::LinearWeights fc2;
    AcousticDecoderWeights acoustic_decoder;
};

HiggsAudioCodecConfig codec_config_from_assets(const HiggsTTSAssets & assets) {
    HiggsAudioCodecConfig config;
    if (assets.paths.codec_config_path.empty() ||
        !engine::io::is_existing_file(assets.paths.codec_config_path)) {
        return config;
    }
    const auto root = json::parse_file(assets.paths.codec_config_path);
    config.sample_rate = static_cast<int>(json::optional_i64(root, "sample_rate", config.sample_rate));
    config.semantic_sample_rate = static_cast<int>(
        json::optional_i64(root, "semantic_sample_rate", config.semantic_sample_rate));
    config.downsample_factor = json::optional_i64(root, "downsample_factor", config.downsample_factor);
    config.codebook_size = json::optional_i64(root, "codebook_size", config.codebook_size);
    config.codebook_dim = json::optional_i64(root, "codebook_dim", config.codebook_dim);
    config.kernel_size = json::optional_i64(root, "kernel_size", config.kernel_size);
    config.unit_kernel_size = json::optional_i64(root, "unit_kernel_size", config.unit_kernel_size);
    config.strides = json::optional_i64_array(root, "strides", config.strides);
    config.block_dilations = json::optional_i64_array(root, "block_dilations", config.block_dilations);
    const auto & semantic = root.require("semantic_model_config");
    config.semantic.hidden_size = json::optional_i64(semantic, "hidden_size", config.semantic.hidden_size);
    config.semantic.intermediate_size = json::optional_i64(semantic, "intermediate_size", config.semantic.intermediate_size);
    config.semantic.num_attention_heads =
        json::optional_i64(semantic, "num_attention_heads", config.semantic.num_attention_heads);
    config.semantic.num_hidden_layers =
        json::optional_i64(semantic, "num_hidden_layers", config.semantic.num_hidden_layers);
    config.semantic.num_conv_pos_embeddings =
        json::optional_i64(semantic, "num_conv_pos_embeddings", config.semantic.num_conv_pos_embeddings);
    config.semantic.num_conv_pos_embedding_groups =
        json::optional_i64(semantic, "num_conv_pos_embedding_groups", config.semantic.num_conv_pos_embedding_groups);
    config.semantic.layer_norm_eps =
        json::optional_f32(semantic, "layer_norm_eps", config.semantic.layer_norm_eps);
    config.semantic.feat_proj_layer_norm =
        json::optional_bool(semantic, "feat_proj_layer_norm", config.semantic.feat_proj_layer_norm);
    config.semantic.do_stable_layer_norm =
        json::optional_bool(semantic, "do_stable_layer_norm", config.semantic.do_stable_layer_norm);
    config.semantic.conv_dim = json::optional_i64_array(semantic, "conv_dim", config.semantic.conv_dim);
    config.semantic.conv_kernel = json::optional_i64_array(semantic, "conv_kernel", config.semantic.conv_kernel);
    config.semantic.conv_stride = json::optional_i64_array(semantic, "conv_stride", config.semantic.conv_stride);
    const auto & acoustic = root.require("acoustic_model_config");
    config.hop_length = json::optional_i64(acoustic, "hop_length", config.hop_length);
    config.acoustic_encoder_hidden_size =
        json::optional_i64(acoustic, "encoder_hidden_size", config.acoustic_encoder_hidden_size);
    config.acoustic_hidden_size = json::optional_i64(acoustic, "hidden_size", config.acoustic_hidden_size);
    config.decoder_hidden_size = json::optional_i64(acoustic, "decoder_hidden_size", config.decoder_hidden_size);
    config.downsampling_ratios =
        json::optional_i64_array(acoustic, "downsampling_ratios", config.downsampling_ratios);
    config.upsampling_ratios =
        json::optional_i64_array(acoustic, "upsampling_ratios", config.upsampling_ratios);
    config.hidden_size = config.acoustic_hidden_size + config.semantic.hidden_size;
    return config;
}

void validate_codec_config(const HiggsAudioCodecConfig & config, const HiggsTTSConfig & tts_config) {
    if (config.sample_rate <= 0 || config.semantic_sample_rate <= 0 ||
        config.downsample_factor <= 0 || config.hop_length <= 0 ||
        config.codebook_size <= 0 || config.codebook_dim <= 0 ||
        config.hidden_size <= 0 || config.acoustic_hidden_size <= 0 ||
        config.acoustic_encoder_hidden_size <= 0 || config.decoder_hidden_size <= 0 ||
        config.downsampling_ratios.empty() || config.upsampling_ratios.empty() ||
        config.strides.empty() || config.block_dilations.empty() ||
        config.semantic.hidden_size <= 0 || config.semantic.intermediate_size <= 0 ||
        config.semantic.num_attention_heads <= 0 || config.semantic.num_hidden_layers <= 0 ||
        config.semantic.num_conv_pos_embeddings <= 0 || config.semantic.num_conv_pos_embedding_groups <= 0 ||
        config.semantic.conv_dim.empty() || config.semantic.conv_kernel.empty() || config.semantic.conv_stride.empty()) {
        throw std::runtime_error("Higgs TTS codec config contains non-positive dimensions");
    }
    if (config.codebook_size != kCodecCodebookSize) {
        throw std::runtime_error("Higgs TTS codec expects 1024-entry RVQ codebooks");
    }
    if (config.semantic.do_stable_layer_norm) {
        throw std::runtime_error("Higgs TTS HuBERT stable-layer-norm variant is not implemented");
    }
    if (config.semantic.hidden_size % config.semantic.num_attention_heads != 0 ||
        config.semantic.hidden_size % config.semantic.num_conv_pos_embedding_groups != 0) {
        throw std::runtime_error("Higgs TTS HuBERT semantic dimensions are invalid");
    }
    if (config.semantic.conv_dim.size() != config.semantic.conv_kernel.size() ||
        config.semantic.conv_dim.size() != config.semantic.conv_stride.size()) {
        throw std::runtime_error("Higgs TTS HuBERT feature extractor config is incomplete");
    }
    if (tts_config.audio_encoder.num_codebooks <= 0) {
        throw std::runtime_error("Higgs TTS audio codebook count is invalid");
    }
}

int64_t product(const std::vector<int64_t> & values) {
    int64_t out = 1;
    for (const int64_t value : values) {
        if (value <= 0) {
            throw std::runtime_error("Higgs TTS codec upsampling ratios must be positive");
        }
        out *= value;
    }
    return out;
}

std::string codec_name(const std::string & suffix) {
    return std::string(kCodecPrefix) + suffix;
}

FlatConv1dWeights load_flat_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    assets::TensorStorageType storage_type,
    bool use_bias) {
    FlatConv1dWeights weights;
    weights.out_channels = out_channels;
    weights.in_channels = in_channels;
    weights.kernel_size = kernel_size;
    weights.weight = store.load_tensor_as_shape(
        source,
        codec_name(prefix + ".weight"),
        storage_type,
        {out_channels, in_channels, kernel_size},
        core::TensorShape::from_dims({out_channels, in_channels * kernel_size}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(
            source,
            codec_name(prefix + ".bias"),
            {out_channels});
    }
    return weights;
}

FlatConvTranspose1dWeights load_flat_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    assets::TensorStorageType storage_type,
    bool use_bias) {
    FlatConvTranspose1dWeights weights;
    weights.in_channels = in_channels;
    weights.out_channels = out_channels;
    weights.kernel_size = kernel_size;
    weights.weight = store.load_tensor_as_shape(
        source,
        codec_name(prefix + ".weight"),
        storage_type,
        {in_channels, out_channels, kernel_size},
        core::TensorShape::from_dims({in_channels, out_channels * kernel_size}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(
            source,
            codec_name(prefix + ".bias"),
            {out_channels});
    }
    return weights;
}

SnakeActivationWeights load_snake_activation(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    SnakeActivationWeights weights;
    weights.alpha = store.make_from_f32(
        core::TensorShape::from_dims({channels}),
        assets::TensorStorageType::F32,
        source.require_f32(codec_name(name), {1, channels, 1}));
    return weights;
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    assets::TensorStorageType storage_type,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(
        source,
        codec_name(prefix + ".weight"),
        storage_type,
        {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(
            source,
            codec_name(prefix + ".bias"),
            {out_features});
    }
    return weights;
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_tensor(source, codec_name(prefix + ".weight"), assets::TensorStorageType::F32, {hidden_size}),
        store.load_tensor(source, codec_name(prefix + ".bias"), assets::TensorStorageType::F32, {hidden_size}),
    };
}

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    assets::TensorStorageType storage_type) {
    modules::AttentionWeights weights;
    weights.q_weight = store.load_tensor(source, codec_name(prefix + ".q_proj.weight"), storage_type, {hidden_size, hidden_size});
    weights.q_bias = store.load_f32_tensor(source, codec_name(prefix + ".q_proj.bias"), {hidden_size});
    weights.k_weight = store.load_tensor(source, codec_name(prefix + ".k_proj.weight"), storage_type, {hidden_size, hidden_size});
    weights.k_bias = store.load_f32_tensor(source, codec_name(prefix + ".k_proj.bias"), {hidden_size});
    weights.v_weight = store.load_tensor(source, codec_name(prefix + ".v_proj.weight"), storage_type, {hidden_size, hidden_size});
    weights.v_bias = store.load_f32_tensor(source, codec_name(prefix + ".v_proj.bias"), {hidden_size});
    weights.out_weight = store.load_tensor(source, codec_name(prefix + ".out_proj.weight"), storage_type, {hidden_size, hidden_size});
    weights.out_bias = store.load_f32_tensor(source, codec_name(prefix + ".out_proj.bias"), {hidden_size});
    return weights;
}

modules::FeedForwardWeights load_feed_forward(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    int64_t intermediate_size,
    assets::TensorStorageType storage_type) {
    return {
        store.load_tensor(source, codec_name(prefix + ".intermediate_dense.weight"), storage_type, {intermediate_size, hidden_size}),
        store.load_f32_tensor(source, codec_name(prefix + ".intermediate_dense.bias"), {intermediate_size}),
        store.load_tensor(source, codec_name(prefix + ".output_dense.weight"), storage_type, {hidden_size, intermediate_size}),
        store.load_f32_tensor(source, codec_name(prefix + ".output_dense.bias"), {hidden_size}),
    };
}

int64_t checked_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("Higgs TTS codec expected positive ") + name);
    }
    return value;
}

struct NormalizedReferenceAudio {
    std::vector<float> acoustic_samples_24k;
    std::vector<float> semantic_samples_16k_padded;
    int64_t frames = 0;
};

std::vector<float> to_mono(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Higgs TTS reference audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Higgs TTS reference audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Higgs TTS reference audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Higgs TTS interleaved reference audio has invalid channel layout");
    }
    return engine::audio::mixdown_interleaved_to_mono_average(
        audio.samples,
        audio.channels,
        engine::audio::MonoMixAccumulation::Float64);
}

std::vector<float> resample_sinc_hann_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate,
    int lowpass_filter_width = 6,
    double rolloff = 0.99) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("Higgs TTS resample sample rates must be positive");
    }
    if (input.empty() || input_sample_rate == output_sample_rate) {
        return input;
    }
    const int rate_gcd = std::gcd(input_sample_rate, output_sample_rate);
    const int orig = input_sample_rate / rate_gcd;
    const int next = output_sample_rate / rate_gcd;
    if (orig <= 0 || next <= 0) {
        throw std::runtime_error("Higgs TTS resample reduced rates must be positive");
    }

    const double base_freq = static_cast<double>(std::min(orig, next)) * rolloff;
    const int64_t width = static_cast<int64_t>(
        std::ceil(static_cast<double>(lowpass_filter_width * orig) / base_freq));
    const int64_t kernel_size = (2 * width) + orig;
    const double scale = base_freq / static_cast<double>(orig);
    const double lowpass = static_cast<double>(lowpass_filter_width);
    const double pi = std::acos(-1.0);

    std::vector<double> kernels(static_cast<size_t>(next * kernel_size), 0.0);
    for (int phase = 0; phase < next; ++phase) {
        const double phase_offset = -static_cast<double>(phase) / static_cast<double>(next);
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            const double idx = static_cast<double>(tap - width) / static_cast<double>(orig);
            double t = (phase_offset + idx) * base_freq;
            t = std::clamp(t, -lowpass, lowpass);
            const double window = std::pow(std::cos(t * pi / lowpass / 2.0), 2.0);
            const double angle = t * pi;
            const double sinc = std::abs(angle) < 1.0e-12 ? 1.0 : std::sin(angle) / angle;
            kernels[static_cast<size_t>(phase * kernel_size + tap)] = sinc * window * scale;
        }
    }

    std::vector<float> padded(static_cast<size_t>(width) + input.size() + static_cast<size_t>(width + orig), 0.0F);
    std::copy(input.begin(), input.end(), padded.begin() + static_cast<std::ptrdiff_t>(width));

    const int64_t steps =
        (static_cast<int64_t>(padded.size()) - kernel_size) / static_cast<int64_t>(orig) + 1;
    const int64_t target_length = static_cast<int64_t>(
        std::ceil(static_cast<double>(next) * static_cast<double>(input.size()) / static_cast<double>(orig)));
    std::vector<float> output(static_cast<size_t>(target_length), 0.0F);
    for (int64_t index = 0; index < target_length; ++index) {
        const int phase = static_cast<int>(index % next);
        const int64_t step = index / next;
        if (step >= steps) {
            break;
        }
        const int64_t start = step * static_cast<int64_t>(orig);
        double sum = 0.0;
        const double * kernel = kernels.data() + static_cast<std::ptrdiff_t>(phase * kernel_size);
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            sum += static_cast<double>(padded[static_cast<size_t>(start + tap)]) * kernel[tap];
        }
        output[static_cast<size_t>(index)] = static_cast<float>(sum);
    }
    return output;
}

int64_t conv1d_output_length(int64_t input, int64_t kernel, int stride, int padding, int dilation) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0) {
        throw std::runtime_error("Higgs TTS conv1d_output_length received an invalid shape");
    }
    return ((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
}

int64_t acoustic_encoder_output_length(int64_t input_samples, const HiggsAudioCodecConfig & config) {
    int64_t length = input_samples;
    length = conv1d_output_length(length, 7, 1, 3, 1);
    for (const int64_t stride_value : config.downsampling_ratios) {
        const int stride = static_cast<int>(checked_positive(stride_value, "downsampling ratio"));
        length = conv1d_output_length(length, 2 * stride, stride, (stride + 1) / 2, 1);
    }
    length = conv1d_output_length(length, 3, 1, 1, 1);
    return length;
}

int64_t semantic_downsample_factor(const HiggsAudioCodecConfig & config) {
    const double factor =
        static_cast<double>(checked_positive(config.hop_length, "hop_length")) /
        (static_cast<double>(checked_positive(config.sample_rate, "sample_rate")) /
         static_cast<double>(checked_positive(config.semantic_sample_rate, "semantic_sample_rate"))) /
        static_cast<double>(checked_positive(config.downsample_factor, "downsample_factor"));
    const auto rounded = static_cast<int64_t>(std::llround(factor));
    if (rounded <= 0 || std::fabs(factor - static_cast<double>(rounded)) > 1.0e-6) {
        throw std::runtime_error("Higgs TTS semantic_downsample_factor is not an integer");
    }
    return rounded;
}

int64_t semantic_feature_frames(const HiggsAudioCodecConfig & config, int64_t semantic_samples) {
    int64_t length = semantic_samples;
    for (size_t i = 0; i < config.semantic.conv_dim.size(); ++i) {
        const int stride = static_cast<int>(config.semantic.conv_stride[i]);
        length = conv1d_output_length(length, config.semantic.conv_kernel[i], stride, 0, 1);
    }
    return length;
}

std::vector<float> pad_acoustic_input_if_needed(
    const std::vector<float> & input,
    const HiggsAudioCodecConfig & config,
    int64_t frame_count) {
    const int64_t output_frames = acoustic_encoder_output_length(static_cast<int64_t>(input.size()), config);
    if (output_frames == frame_count) {
        return input;
    }
    std::vector<float> padded(static_cast<size_t>(input.size() + config.hop_length), 0.0F);
    std::copy(input.begin(), input.end(), padded.begin() + static_cast<std::ptrdiff_t>(config.hop_length / 2));
    if (acoustic_encoder_output_length(static_cast<int64_t>(padded.size()), config) != frame_count) {
        throw std::runtime_error("Higgs TTS acoustic/semantic reference frame alignment is unsupported");
    }
    return padded;
}

NormalizedReferenceAudio normalize_reference_audio(
    const runtime::AudioBuffer & audio,
    const HiggsAudioCodecConfig & config) {
    auto mono = to_mono(audio);
    if (audio.sample_rate != config.sample_rate) {
        mono = resample_sinc_hann_mono(mono, audio.sample_rate, config.sample_rate);
    }
    if (static_cast<int64_t>(mono.size()) < config.sample_rate) {
        mono.resize(static_cast<size_t>(config.sample_rate), 0.0F);
    }

    auto semantic = resample_sinc_hann_mono(mono, config.sample_rate, config.semantic_sample_rate);
    semantic.insert(semantic.begin(), 160, 0.0F);
    semantic.insert(semantic.end(), 160, 0.0F);
    const int64_t feature_frames = semantic_feature_frames(config, static_cast<int64_t>(semantic.size()));
    const int64_t downsample = semantic_downsample_factor(config);
    const int64_t frames = (feature_frames + downsample - 1) / downsample;
    if (frames <= 0) {
        throw std::runtime_error("Higgs TTS reference audio produced no codec frames");
    }

    NormalizedReferenceAudio normalized;
    normalized.frames = frames;
    normalized.semantic_samples_16k_padded = std::move(semantic);
    normalized.acoustic_samples_24k = pad_acoustic_input_if_needed(mono, config, frames);
    return normalized;
}

std::vector<float> apply_positional_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    if (static_cast<int64_t>(g.size()) != kernel_size ||
        static_cast<int64_t>(v.size()) != out_channels * in_channels * kernel_size) {
        throw std::runtime_error("Higgs TTS positional weight-norm tensor shape mismatch");
    }
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
        double squared_norm = 0.0;
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t offset = static_cast<size_t>(((out * in_channels) + in) * kernel_size + kernel_index);
                const double value = static_cast<double>(v[offset]);
                squared_norm += value * value;
            }
        }
        const double scale = static_cast<double>(g[static_cast<size_t>(kernel_index)]) /
            std::sqrt(std::max(squared_norm, 1.0e-20));
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t offset = static_cast<size_t>(((out * in_channels) + in) * kernel_size + kernel_index);
                weight[offset] = static_cast<float>(static_cast<double>(v[offset]) * scale);
            }
        }
    }
    return weight;
}

std::vector<float> select_group_kernel(
    const std::vector<float> & values,
    int64_t total_out,
    int64_t total_in_per_group,
    int64_t kernel_size,
    int64_t groups,
    int64_t group_index) {
    const int64_t out_per_group = total_out / groups;
    std::vector<float> group_values(static_cast<size_t>(out_per_group * total_in_per_group * kernel_size), 0.0F);
    for (int64_t out = 0; out < out_per_group; ++out) {
        const int64_t global_out = group_index * out_per_group + out;
        const size_t src_offset = static_cast<size_t>(global_out * total_in_per_group * kernel_size);
        const size_t dst_offset = static_cast<size_t>(out * total_in_per_group * kernel_size);
        std::copy_n(
            values.begin() + static_cast<std::ptrdiff_t>(src_offset),
            total_in_per_group * kernel_size,
            group_values.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }
    return group_values;
}

std::vector<float> select_group_bias(
    const std::vector<float> & bias,
    int64_t groups,
    int64_t group_index) {
    const int64_t out_per_group = static_cast<int64_t>(bias.size()) / groups;
    std::vector<float> group_bias(static_cast<size_t>(out_per_group), 0.0F);
    std::copy_n(
        bias.begin() + static_cast<std::ptrdiff_t>(group_index * out_per_group),
        out_per_group,
        group_bias.begin());
    return group_bias;
}

FlatConv1dWeights reshape_conv1d_weight(core::ModuleBuildContext & ctx, const FlatConv1dWeights & flat) {
    FlatConv1dWeights reshaped = flat;
    reshaped.weight = core::reshape_tensor(
        ctx,
        flat.weight,
        core::TensorShape::from_dims({flat.out_channels, flat.in_channels, flat.kernel_size}));
    return reshaped;
}

FlatConvTranspose1dWeights reshape_conv_transpose1d_weight(
    core::ModuleBuildContext & ctx,
    const FlatConvTranspose1dWeights & flat) {
    FlatConvTranspose1dWeights reshaped = flat;
    reshaped.weight = core::reshape_tensor(
        ctx,
        flat.weight,
        core::TensorShape::from_dims({flat.in_channels, flat.out_channels, flat.kernel_size}));
    return reshaped;
}

modules::Conv1dWeights make_conv1d_weights(core::ModuleBuildContext & ctx, const FlatConv1dWeights & flat) {
    const auto reshaped = reshape_conv1d_weight(ctx, flat);
    return {reshaped.weight, reshaped.bias};
}

modules::ConvTranspose1dWeights make_conv_transpose1d_weights(
    core::ModuleBuildContext & ctx,
    const FlatConvTranspose1dWeights & flat) {
    const auto reshaped = reshape_conv_transpose1d_weight(ctx, flat);
    return {reshaped.weight, reshaped.bias};
}

std::shared_ptr<const HiggsAudioCodecWeights> load_weights(
    const HiggsTTSAssets & assets_ref,
    const HiggsAudioCodecConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & source = *assets_ref.model_weights;
    auto weights = std::make_shared<HiggsAudioCodecWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "higgs_tts.codec.weights",
        weight_context_bytes);
    const auto & semantic = config.semantic;
    weights->feature_extractor.convs.reserve(semantic.conv_dim.size());
    int64_t feature_in_channels = 1;
    for (size_t i = 0; i < semantic.conv_dim.size(); ++i) {
        const std::string prefix = "semantic_model.feature_extractor.conv_layers." + std::to_string(i) + ".conv";
        weights->feature_extractor.convs.push_back(load_flat_conv1d(
            *weights->store,
            source,
            prefix,
            semantic.conv_dim[i],
            feature_in_channels,
            semantic.conv_kernel[i],
            storage_type,
            false));
        feature_in_channels = semantic.conv_dim[i];
    }
    weights->feature_extractor.first_group_norm = {
        weights->store->load_tensor(
            source,
            codec_name("semantic_model.feature_extractor.conv_layers.0.layer_norm.weight"),
            assets::TensorStorageType::F32,
            {semantic.conv_dim.front()}),
        weights->store->load_tensor(
            source,
            codec_name("semantic_model.feature_extractor.conv_layers.0.layer_norm.bias"),
            assets::TensorStorageType::F32,
            {semantic.conv_dim.front()}),
    };
    weights->feature_projection.layer_norm = {
        weights->store->load_tensor(
            source,
            codec_name("semantic_model.feature_projection.layer_norm.weight"),
            assets::TensorStorageType::F32,
            {semantic.conv_dim.back()}),
        weights->store->load_tensor(
            source,
            codec_name("semantic_model.feature_projection.layer_norm.bias"),
            assets::TensorStorageType::F32,
            {semantic.conv_dim.back()}),
    };
    weights->feature_projection.projection = load_linear(
        *weights->store,
        source,
        "semantic_model.feature_projection.projection",
        semantic.hidden_size,
        semantic.conv_dim.back(),
        storage_type,
        true);

    const auto pos_g = source.require_f32(
        codec_name("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
        {1, 1, semantic.num_conv_pos_embeddings});
    const auto pos_v = source.require_f32(
        codec_name("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
        {semantic.hidden_size, semantic.hidden_size / semantic.num_conv_pos_embedding_groups, semantic.num_conv_pos_embeddings});
    const auto pos_bias = source.require_f32(
        codec_name("semantic_model.encoder.pos_conv_embed.conv.bias"),
        {semantic.hidden_size});
    const auto pos_weight = apply_positional_weight_norm(
        pos_g,
        pos_v,
        semantic.hidden_size,
        semantic.hidden_size / semantic.num_conv_pos_embedding_groups,
        semantic.num_conv_pos_embeddings);
    const int64_t pos_groups = semantic.num_conv_pos_embedding_groups;
    const int64_t pos_channels_per_group = semantic.hidden_size / pos_groups;
    weights->positional_conv_groups.reserve(static_cast<size_t>(pos_groups));
    for (int64_t group_index = 0; group_index < pos_groups; ++group_index) {
        const auto group_weight = select_group_kernel(
            pos_weight,
            semantic.hidden_size,
            pos_channels_per_group,
            semantic.num_conv_pos_embeddings,
            pos_groups,
            group_index);
        const auto group_bias = select_group_bias(pos_bias, pos_groups, group_index);
        HubertPositionConvGroupWeights group;
        group.conv.out_channels = pos_channels_per_group;
        group.conv.in_channels = pos_channels_per_group;
        group.conv.kernel_size = semantic.num_conv_pos_embeddings;
        group.conv.weight = weights->store->make_from_f32(
            core::TensorShape::from_dims({pos_channels_per_group, pos_channels_per_group * semantic.num_conv_pos_embeddings}),
            storage_type,
            group_weight);
        group.conv.bias = weights->store->make_from_f32(
            core::TensorShape::from_dims({pos_channels_per_group}),
            assets::TensorStorageType::F32,
            group_bias);
        weights->positional_conv_groups.push_back(std::move(group));
    }

    weights->encoder_input_layer_norm = load_norm(
        *weights->store,
        source,
        "semantic_model.encoder.layer_norm",
        semantic.hidden_size);
    weights->hubert_layers.reserve(static_cast<size_t>(semantic.num_hidden_layers));
    for (int64_t layer = 0; layer < semantic.num_hidden_layers; ++layer) {
        const std::string prefix = "semantic_model.encoder.layers." + std::to_string(layer);
        HubertEncoderLayerWeights layer_weights;
        layer_weights.attention = load_attention(
            *weights->store,
            source,
            prefix + ".attention",
            semantic.hidden_size,
            storage_type);
        layer_weights.layer_norm = load_norm(*weights->store, source, prefix + ".layer_norm", semantic.hidden_size);
        layer_weights.feed_forward = load_feed_forward(
            *weights->store,
            source,
            prefix + ".feed_forward",
            semantic.hidden_size,
            semantic.intermediate_size,
            storage_type);
        layer_weights.final_layer_norm = load_norm(
            *weights->store,
            source,
            prefix + ".final_layer_norm",
            semantic.hidden_size);
        weights->hubert_layers.push_back(std::move(layer_weights));
    }

    weights->semantic_encoder.conv = load_flat_conv1d(
        *weights->store,
        source,
        "encoder_semantic.conv",
        semantic.hidden_size,
        semantic.hidden_size,
        config.kernel_size,
        storage_type,
        false);
    weights->semantic_encoder.blocks.reserve(config.strides.size());
    for (size_t block_index = 0; block_index < config.strides.size(); ++block_index) {
        SemanticEncoderBlockWeights block;
        block.residual_conv1.reserve(config.block_dilations.size());
        block.residual_conv2.reserve(config.block_dilations.size());
        for (size_t residual_index = 0; residual_index < config.block_dilations.size(); ++residual_index) {
            const std::string prefix =
                "encoder_semantic.conv_blocks." + std::to_string(block_index) +
                ".res_units." + std::to_string(residual_index);
            block.residual_conv1.push_back(load_flat_conv1d(
                *weights->store,
                source,
                prefix + ".conv1",
                semantic.hidden_size,
                semantic.hidden_size,
                config.unit_kernel_size,
                storage_type,
                false));
            block.residual_conv2.push_back(load_flat_conv1d(
                *weights->store,
                source,
                prefix + ".conv2",
                semantic.hidden_size,
                semantic.hidden_size,
                1,
                storage_type,
                false));
        }
        block.conv = load_flat_conv1d(
            *weights->store,
            source,
            "encoder_semantic.conv_blocks." + std::to_string(block_index) + ".conv",
            semantic.hidden_size,
            semantic.hidden_size,
            config.strides[block_index] == 1 ? 3 : (2 * config.strides[block_index]),
            storage_type,
            true);
        weights->semantic_encoder.blocks.push_back(std::move(block));
    }

    weights->acoustic_encoder.conv1 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_encoder.conv1",
        config.acoustic_encoder_hidden_size,
        1,
        7,
        storage_type,
        true);
    weights->acoustic_encoder.blocks.reserve(config.downsampling_ratios.size());
    int64_t encoder_block_input = config.acoustic_encoder_hidden_size;
    for (size_t block_index = 0; block_index < config.downsampling_ratios.size(); ++block_index) {
        const int64_t stride = config.downsampling_ratios[block_index];
        const int64_t output_channels =
            config.acoustic_encoder_hidden_size * (int64_t{1} << (static_cast<int64_t>(block_index) + 1));
        const std::string block_prefix = "acoustic_encoder.block." + std::to_string(block_index);
        auto load_encoder_residual_unit = [&](const std::string & prefix, int64_t channels) {
            ResidualUnitWeights unit;
            unit.snake1 = load_snake_activation(*weights->store, source, prefix + ".snake1.alpha", channels);
            unit.conv1 = load_flat_conv1d(*weights->store, source, prefix + ".conv1", channels, channels, 7, storage_type, true);
            unit.snake2 = load_snake_activation(*weights->store, source, prefix + ".snake2.alpha", channels);
            unit.conv2 = load_flat_conv1d(*weights->store, source, prefix + ".conv2", channels, channels, 1, storage_type, true);
            return unit;
        };
        AcousticEncoderBlockWeights block;
        block.res1 = load_encoder_residual_unit(block_prefix + ".res_unit1", encoder_block_input);
        block.res2 = load_encoder_residual_unit(block_prefix + ".res_unit2", encoder_block_input);
        block.res3 = load_encoder_residual_unit(block_prefix + ".res_unit3", encoder_block_input);
        block.snake = load_snake_activation(*weights->store, source, block_prefix + ".snake1.alpha", encoder_block_input);
        block.conv = load_flat_conv1d(
            *weights->store,
            source,
            block_prefix + ".conv1",
            output_channels,
            encoder_block_input,
            2 * stride,
            storage_type,
            true);
        weights->acoustic_encoder.blocks.push_back(std::move(block));
        encoder_block_input = output_channels;
    }
    weights->acoustic_encoder.snake =
        load_snake_activation(*weights->store, source, "acoustic_encoder.snake1.alpha", encoder_block_input);
    weights->acoustic_encoder.conv2 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_encoder.conv2",
        config.acoustic_hidden_size,
        encoder_block_input,
        3,
        storage_type,
        true);

    weights->fc = load_linear(
        *weights->store,
        source,
        "fc",
        config.hidden_size,
        config.hidden_size,
        storage_type,
        true);

    weights->quantizers.reserve(static_cast<size_t>(assets_ref.config.audio_encoder.num_codebooks));
    for (int64_t quantizer_index = 0; quantizer_index < assets_ref.config.audio_encoder.num_codebooks; ++quantizer_index) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(quantizer_index);
        QuantizerWeights quantizer;
        const auto codebook = source.require_f32(codec_name(prefix + ".codebook.embed"), {config.codebook_size, config.codebook_dim});
        std::vector<float> score_weight(static_cast<size_t>(config.codebook_size * config.codebook_dim), 0.0F);
        std::vector<float> score_bias(static_cast<size_t>(config.codebook_size), 0.0F);
        for (int64_t row = 0; row < config.codebook_size; ++row) {
            double norm = 0.0;
            for (int64_t col = 0; col < config.codebook_dim; ++col) {
                const float value = codebook[static_cast<size_t>(row * config.codebook_dim + col)];
                score_weight[static_cast<size_t>(row * config.codebook_dim + col)] = 2.0F * value;
                norm += static_cast<double>(value) * static_cast<double>(value);
            }
            score_bias[static_cast<size_t>(row)] = static_cast<float>(-norm);
        }
        quantizer.project_in = load_linear(
            *weights->store,
            source,
            prefix + ".project_in",
            config.codebook_dim,
            config.hidden_size,
            storage_type,
            true);
        quantizer.score = {
            weights->store->make_from_f32(
                core::TensorShape::from_dims({config.codebook_size, config.codebook_dim}),
                storage_type,
                score_weight),
            weights->store->make_from_f32(
                core::TensorShape::from_dims({config.codebook_size}),
                assets::TensorStorageType::F32,
                score_bias),
        };
        quantizer.codebook = weights->store->load_tensor(
            source,
            codec_name(prefix + ".codebook.embed"),
            storage_type,
            {config.codebook_size, config.codebook_dim});
        quantizer.project_out = load_linear(
            *weights->store,
            source,
            prefix + ".project_out",
            config.hidden_size,
            config.codebook_dim,
            storage_type,
            true);
        weights->quantizers.push_back(std::move(quantizer));
    }

    weights->fc2 = load_linear(
        *weights->store,
        source,
        "fc2",
        config.acoustic_hidden_size,
        config.hidden_size,
        storage_type,
        true);
    weights->acoustic_decoder.conv1 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_decoder.conv1",
        config.decoder_hidden_size,
        config.acoustic_hidden_size,
        7,
        storage_type,
        true);
    weights->acoustic_decoder.blocks.reserve(config.upsampling_ratios.size());
    for (size_t block_index = 0; block_index < config.upsampling_ratios.size(); ++block_index) {
        const int64_t stride = config.upsampling_ratios[block_index];
        const int64_t input_channels = config.decoder_hidden_size / (int64_t{1} << static_cast<int64_t>(block_index));
        const int64_t output_channels = config.decoder_hidden_size / (int64_t{1} << (static_cast<int64_t>(block_index) + 1));
        const std::string block_prefix = "acoustic_decoder.block." + std::to_string(block_index);
        auto load_residual_unit = [&](const std::string & prefix, int64_t channels) {
            ResidualUnitWeights unit;
            unit.snake1 = load_snake_activation(*weights->store, source, prefix + ".snake1.alpha", channels);
            unit.conv1 = load_flat_conv1d(*weights->store, source, prefix + ".conv1", channels, channels, 7, storage_type, true);
            unit.snake2 = load_snake_activation(*weights->store, source, prefix + ".snake2.alpha", channels);
            unit.conv2 = load_flat_conv1d(*weights->store, source, prefix + ".conv2", channels, channels, 1, storage_type, true);
            return unit;
        };
        AcousticDecoderBlockWeights block;
        block.snake = load_snake_activation(*weights->store, source, block_prefix + ".snake1.alpha", input_channels);
        block.conv_t = load_flat_conv_transpose1d(
            *weights->store,
            source,
            block_prefix + ".conv_t1",
            input_channels,
            output_channels,
            2 * stride,
            storage_type,
            true);
        block.res1 = load_residual_unit(block_prefix + ".res_unit1", output_channels);
        block.res2 = load_residual_unit(block_prefix + ".res_unit2", output_channels);
        block.res3 = load_residual_unit(block_prefix + ".res_unit3", output_channels);
        weights->acoustic_decoder.blocks.push_back(std::move(block));
    }
    const int64_t final_channels = config.decoder_hidden_size /
        (int64_t{1} << static_cast<int64_t>(config.upsampling_ratios.size()));
    weights->acoustic_decoder.snake = load_snake_activation(
        *weights->store,
        source,
        "acoustic_decoder.snake1.alpha",
        final_channels);
    weights->acoustic_decoder.conv2 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_decoder.conv2",
        1,
        final_channels,
        7,
        storage_type,
        true);
    weights->store->upload();
    return weights;
}

core::TensorValue transpose_btc_to_bct(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1}, value.shape.rank}).build(ctx, value);
}

core::TensorValue transpose_bct_to_btc(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1}, value.shape.rank}).build(ctx, value);
}

core::TensorValue build_snake1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SnakeActivationWeights & weights,
    int64_t channels) {
    return modules::Snake1dModule({channels}).build(ctx, input, {weights.alpha});
}

core::TensorValue ensure_dense_layout(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor) &&
        !ggml_is_transposed(value.tensor) &&
        value.tensor->nb[0] == ggml_type_size(value.tensor->type)) {
        return value;
    }
    return core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
}

core::TensorValue build_conv_transpose_with_crop(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConvTranspose1dWeights & weights,
    int stride,
    int padding,
    int output_padding) {
    const auto dense_input = ensure_dense_layout(ctx, input);
    auto output = modules::ConvTranspose1dModule({
        weights.in_channels,
        weights.out_channels,
        weights.kernel_size,
        stride,
        0,
        1,
        weights.bias.has_value(),
    }).build(ctx, dense_input, make_conv_transpose1d_weights(ctx, weights));
    const int64_t cropped_length =
        (input.shape.dims[2] - 1) * stride - 2 * padding + weights.kernel_size + output_padding;
    if (cropped_length <= 0 || cropped_length > output.shape.dims[2]) {
        throw std::runtime_error("Higgs TTS codec ConvTranspose crop length is invalid");
    }
    auto * view = ggml_view_3d(
        ctx.ggml,
        output.tensor,
        cropped_length,
        weights.out_channels,
        1,
        output.tensor->nb[1],
        output.tensor->nb[2],
        static_cast<size_t>(padding) * output.tensor->nb[0]);
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, view),
        core::TensorShape::from_dims({1, weights.out_channels, cropped_length}),
        GGML_TYPE_F32);
}

core::TensorValue build_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels,
    int dilation) {
    auto x = build_snake1d(ctx, input, weights.snake1, channels);
    x = modules::Conv1dModule({channels, channels, 7, 1, 3 * dilation, dilation, true})
            .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, weights.conv1));
    x = build_snake1d(ctx, x, weights.snake2, channels);
    x = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true})
            .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, weights.conv2));
    return modules::AddModule().build(ctx, input, x);
}

core::TensorValue ensure_f32_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const modules::NormWeights & weights,
    core::DeferredTensorWriter & writer) {
    core::TensorValue output;
    if (input.shape.rank == 3 && groups == input.shape.dims[1]) {
        auto input_f32 = ensure_f32_tensor(ctx, input);
        auto mean = modules::ReduceMeanModule({2}).build(ctx, input_f32);
        auto mean_rep = modules::RepeatModule(modules::RepeatConfig{input_f32.shape}).build(ctx, mean);
        auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
        auto centered_sq = modules::MulModule{}.build(ctx, centered, centered);
        auto variance = modules::ReduceMeanModule({2}).build(ctx, centered_sq);
        auto eps_tensor = writer.make_f32_tensor(ctx, core::TensorShape::from_dims({1, 1, 1}), {eps});
        auto eps_rep = modules::RepeatModule(modules::RepeatConfig{variance.shape}).build(ctx, eps_tensor);
        auto std = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, variance, eps_rep));
        auto std_rep = modules::RepeatModule(modules::RepeatConfig{input_f32.shape}).build(ctx, std);
        output = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, std_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    } else {
        output = core::wrap_tensor(ggml_group_norm(ctx.ggml, input.tensor, groups, eps), input.shape, GGML_TYPE_F32);
    }
    if (weights.weight.has_value()) {
        auto weight = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

core::TensorValue build_semantic_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConv1dWeights & conv1,
    const FlatConv1dWeights & conv2,
    int64_t channels,
    int dilation) {
    auto x = modules::EluModule().build(ctx, input);
    const int padding = static_cast<int>(((conv1.kernel_size - 1) / 2) * static_cast<int64_t>(dilation));
    x = modules::Conv1dModule({channels, channels, conv1.kernel_size, 1, padding, dilation, false})
            .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, conv1));
    x = modules::EluModule().build(ctx, x);
    x = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, false})
            .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, conv2));
    return modules::AddModule().build(ctx, input, x);
}

core::TensorValue build_conv1d_im2col_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConv1dWeights & weights,
    int stride,
    int padding,
    int dilation) {
    core::validate_rank_between(input, 3, 3, "input");
    const int64_t output_frames =
        (input.shape.dims[2] + 2 * padding - dilation * (weights.kernel_size - 1) - 1) / stride + 1;
    if (output_frames <= 0) {
        throw std::runtime_error("Higgs TTS exact conv1d computed non-positive output length");
    }
    auto input_f32 = ensure_f32_tensor(ctx, ensure_dense_layout(ctx, input));
    auto reshaped = reshape_conv1d_weight(ctx, weights);
    auto weight_f32 = ensure_f32_tensor(ctx, ensure_dense_layout(ctx, reshaped.weight));
    ggml_tensor * im2col = ggml_im2col(
        ctx.ggml,
        weight_f32.tensor,
        input_f32.tensor,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    auto im2col_btk = core::wrap_tensor(
        im2col,
        core::TensorShape::from_dims({input.shape.dims[0], output_frames, weights.in_channels * weights.kernel_size}),
        GGML_TYPE_F32);
    im2col_btk = ensure_dense_layout(ctx, im2col_btk);
    modules::LinearWeights linear_weights;
    linear_weights.weight = core::reshape_tensor(
        ctx,
        reshaped.weight,
        core::TensorShape::from_dims({weights.out_channels, weights.in_channels * weights.kernel_size}));
    if (weights.bias.has_value()) {
        linear_weights.bias = *weights.bias;
    }
    auto output_bto = modules::LinearModule({
        weights.in_channels * weights.kernel_size,
        weights.out_channels,
        weights.bias.has_value(),
    }).build(ctx, im2col_btk, linear_weights);
    return transpose_btc_to_bct(ctx, output_bto);
}

core::TensorValue build_grouped_positional_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::vector<HubertPositionConvGroupWeights> & groups,
    int64_t total_channels,
    int64_t kernel_size) {
    if (groups.empty()) {
        throw std::runtime_error("Higgs TTS positional conv group weights are empty");
    }
    const int64_t channels_per_group = total_channels / static_cast<int64_t>(groups.size());
    std::optional<core::TensorValue> output;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto input_slice = modules::SliceModule({
            1,
            static_cast<int64_t>(group_index) * channels_per_group,
            channels_per_group,
        }).build(ctx, input);
        auto conv = modules::Conv1dModule({
            channels_per_group,
            channels_per_group,
            kernel_size,
            1,
            static_cast<int>(kernel_size / 2),
            1,
            true,
        }).build(ctx, ensure_dense_layout(ctx, input_slice), make_conv1d_weights(ctx, groups[group_index].conv));
        output = output.has_value() ? modules::ConcatModule({1}).build(ctx, *output, conv) : conv;
    }
    auto combined = *output;
    if (kernel_size % 2 == 0) {
        combined = modules::SliceModule({2, 0, input.shape.dims[2]}).build(ctx, combined);
    }
    return modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, combined);
}

core::TensorValue build_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::AttentionWeights & weights,
    int64_t hidden_size,
    int64_t num_heads) {
    if (hidden_size <= 0 || num_heads <= 0 || (hidden_size % num_heads) != 0) {
        throw std::runtime_error("Higgs TTS HuBERT self-attention received an invalid hidden shape");
    }
    const int64_t head_dim = hidden_size / num_heads;
    const modules::LinearModule q_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule k_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule v_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule out_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    auto q = q_proj.build(ctx, input, {weights.q_weight, weights.q_bias});
    auto k = k_proj.build(ctx, input, {weights.k_weight, weights.k_bias});
    auto v = v_proj.build(ctx, input, {weights.v_weight, weights.v_bias});
    q = core::reshape_tensor(ctx, ensure_dense_layout(ctx, q), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    k = core::reshape_tensor(ctx, ensure_dense_layout(ctx, k), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    v = core::reshape_tensor(ctx, ensure_dense_layout(ctx, v), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = ensure_dense_layout(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    context = core::reshape_tensor(
        ctx,
        ensure_dense_layout(ctx, modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context)),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    return out_proj.build(ctx, context, {weights.out_weight, weights.out_bias});
}

core::TensorValue build_hubert_sequence_mean(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const HiggsAudioCodecWeights & weights,
    const HiggsAudioCodecConfig & config) {
    auto x = hidden;
    auto summed = x;
    const modules::LayerNormModule layer_norm({config.semantic.hidden_size, config.semantic.layer_norm_eps, true, true});
    const modules::FeedForwardModule feed_forward({
        config.semantic.hidden_size,
        config.semantic.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
    });
    for (const auto & layer_weights : weights.hubert_layers) {
        auto attn = build_self_attention(
            ctx,
            x,
            layer_weights.attention,
            config.semantic.hidden_size,
            config.semantic.num_attention_heads);
        x = modules::AddModule().build(ctx, x, attn);
        x = layer_norm.build(ctx, x, layer_weights.layer_norm);
        auto ff = feed_forward.build(ctx, x, layer_weights.feed_forward);
        x = modules::AddModule().build(ctx, x, ff);
        x = layer_norm.build(ctx, x, layer_weights.final_layer_norm);
        summed = modules::AddModule().build(ctx, summed, x);
    }
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, summed.tensor, 1.0F / static_cast<float>(weights.hubert_layers.size() + 1)),
        summed.shape,
        GGML_TYPE_F32);
}

core::TensorValue build_semantic_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioCodecWeights & weights,
    const HiggsAudioCodecConfig & config) {
    auto x = modules::Conv1dModule({
        config.semantic.hidden_size,
        config.semantic.hidden_size,
        config.kernel_size,
        1,
        static_cast<int>(config.kernel_size / 2),
        1,
        false,
    }).build(ctx, ensure_dense_layout(ctx, input_bct), make_conv1d_weights(ctx, weights.semantic_encoder.conv));
    for (size_t block_index = 0; block_index < weights.semantic_encoder.blocks.size(); ++block_index) {
        const auto & block = weights.semantic_encoder.blocks[block_index];
        for (size_t residual_index = 0; residual_index < block.residual_conv1.size(); ++residual_index) {
            x = build_semantic_residual_unit(
                ctx,
                x,
                block.residual_conv1[residual_index],
                block.residual_conv2[residual_index],
                config.semantic.hidden_size,
                static_cast<int>(config.block_dilations[residual_index]));
        }
        const int stride = static_cast<int>(config.strides[block_index]);
        const int kernel = stride == 1 ? 3 : static_cast<int>(2 * config.strides[block_index]);
        const int padding = (kernel - 1) / 2;
        x = modules::Conv1dModule({
            config.semantic.hidden_size,
            config.semantic.hidden_size,
            kernel,
            stride,
            padding,
            1,
            true,
        }).build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, block.conv));
    }
    return x;
}

core::TensorValue build_acoustic_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioCodecWeights & weights,
    const HiggsAudioCodecConfig & config) {
    auto x = modules::Conv1dModule({1, config.acoustic_encoder_hidden_size, 7, 1, 3, 1, true})
                 .build(ctx, ensure_dense_layout(ctx, input_bct), make_conv1d_weights(ctx, weights.acoustic_encoder.conv1));
    int64_t channels = config.acoustic_encoder_hidden_size;
    for (size_t block_index = 0; block_index < weights.acoustic_encoder.blocks.size(); ++block_index) {
        const auto & block = weights.acoustic_encoder.blocks[block_index];
        x = build_residual_unit(ctx, x, block.res1, channels, 1);
        x = build_residual_unit(ctx, x, block.res2, channels, 3);
        x = build_residual_unit(ctx, x, block.res3, channels, 9);
        x = build_snake1d(ctx, x, block.snake, channels);
        const int stride = static_cast<int>(config.downsampling_ratios[block_index]);
        const int64_t next_channels =
            config.acoustic_encoder_hidden_size * (int64_t{1} << (static_cast<int64_t>(block_index) + 1));
        x = modules::Conv1dModule({
            channels,
            next_channels,
            2 * stride,
            stride,
            (stride + 1) / 2,
            1,
            true,
        }).build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, block.conv));
        channels = next_channels;
    }
    x = build_snake1d(ctx, x, weights.acoustic_encoder.snake, channels);
    return modules::Conv1dModule({channels, config.acoustic_hidden_size, 3, 1, 1, 1, true})
        .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, weights.acoustic_encoder.conv2));
}

core::TensorValue build_quantizer_decode_sequence(
    core::ModuleBuildContext & ctx,
    const std::vector<ggml_tensor *> & code_inputs,
    const HiggsAudioCodecWeights & weights,
    int64_t frames) {
    std::optional<core::TensorValue> latent;
    for (size_t quantizer_index = 0; quantizer_index < code_inputs.size(); ++quantizer_index) {
        const auto codes = core::wrap_tensor(
            code_inputs[quantizer_index],
            core::TensorShape::from_dims({1, frames}),
            GGML_TYPE_I32);
        const auto & quantizer = weights.quantizers[quantizer_index];
        auto embedded = modules::CodebookLookupModule({
            quantizer.codebook.shape.dims[0],
            quantizer.codebook.shape.dims[1],
        }).build(ctx, codes, quantizer.codebook);
        auto projected = modules::LinearModule({
            embedded.shape.dims[2],
            quantizer.project_out.weight.shape.dims[0],
            true,
        }).build(ctx, embedded, quantizer.project_out);
        auto projected_bct = transpose_btc_to_bct(ctx, projected);
        latent = latent.has_value() ? modules::AddModule().build(ctx, *latent, projected_bct) : projected_bct;
    }
    if (!latent.has_value()) {
        throw std::runtime_error("Higgs TTS codec quantizer decode requires at least one codebook");
    }
    return *latent;
}

core::TensorValue build_acoustic_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioCodecWeights & weights,
    const HiggsAudioCodecConfig & config) {
    auto x = modules::Conv1dModule({
        config.acoustic_hidden_size,
        config.decoder_hidden_size,
        7,
        1,
        3,
        1,
        true,
    }).build(ctx, ensure_dense_layout(ctx, input_bct), make_conv1d_weights(ctx, weights.acoustic_decoder.conv1));
    int64_t channels = config.decoder_hidden_size;
    for (size_t block_index = 0; block_index < weights.acoustic_decoder.blocks.size(); ++block_index) {
        const auto & block = weights.acoustic_decoder.blocks[block_index];
        const int stride = static_cast<int>(config.upsampling_ratios[block_index]);
        const int padding = (stride + 1) / 2;
        const int output_padding = stride % 2;
        x = build_snake1d(ctx, x, block.snake, channels);
        x = build_conv_transpose_with_crop(ctx, x, block.conv_t, stride, padding, output_padding);
        channels = block.conv_t.out_channels;
        x = build_residual_unit(ctx, x, block.res1, channels, 1);
        x = build_residual_unit(ctx, x, block.res2, channels, 3);
        x = build_residual_unit(ctx, x, block.res3, channels, 9);
    }
    x = build_snake1d(ctx, x, weights.acoustic_decoder.snake, channels);
    return modules::Conv1dModule({channels, 1, 7, 1, 3, 1, true})
        .build(ctx, ensure_dense_layout(ctx, x), make_conv1d_weights(ctx, weights.acoustic_decoder.conv2));
}

HiggsAudioCodeMatrix trim_decodable_codes(const HiggsAudioCodeMatrix & raw_codes, int64_t codebook_size) {
    if (raw_codes.frames <= 0 || raw_codes.codebooks <= 0 ||
        static_cast<int64_t>(raw_codes.token_ids.size()) != raw_codes.frames * raw_codes.codebooks) {
        throw std::runtime_error("Higgs TTS codec received an invalid raw code matrix");
    }
    int64_t frames = 0;
    for (; frames < raw_codes.frames; ++frames) {
        bool valid = true;
        for (int64_t codebook = 0; codebook < raw_codes.codebooks; ++codebook) {
            const int32_t code = raw_codes.token_ids[static_cast<size_t>(frames * raw_codes.codebooks + codebook)];
            if (code < 0 || static_cast<int64_t>(code) >= codebook_size) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            break;
        }
    }
    if (frames <= 0) {
        throw std::runtime_error("Higgs TTS codec did not receive any decodable audio-code frames");
    }
    HiggsAudioCodeMatrix out;
    out.frames = frames;
    out.codebooks = raw_codes.codebooks;
    out.token_ids.assign(
        raw_codes.token_ids.begin(),
        raw_codes.token_ids.begin() + static_cast<std::ptrdiff_t>(frames * raw_codes.codebooks));
    return out;
}

int64_t env_int64_or_default(
    const char * name,
    int64_t fallback,
    int64_t minimum,
    int64_t maximum) {
    const char * raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    try {
        return std::clamp(std::stoll(raw), minimum, maximum);
    } catch (...) {
        return fallback;
    }
}

int64_t codec_decode_fallback_frames() {
    return env_int64_or_default(
        "HIGGS_TTS_CODEC_FALLBACK_FRAMES",
        kCodecDecodeFallbackFrames,
        kCodecDecodeFallbackMinFrames,
        kCodecDecodeFallbackMaxFrames);
}

int64_t codec_decode_fallback_overlap_frames() {
    return env_int64_or_default(
        "HIGGS_TTS_CODEC_FALLBACK_OVERLAP_FRAMES",
        kCodecDecodeFallbackOverlapFrames,
        0,
        kCodecDecodeFallbackMaxOverlapFrames);
}

bool is_codec_graph_retryable_failure(const std::runtime_error & error) {
    const std::string message = error.what();
    return message.find("allocate Higgs TTS codec graph") != std::string::npos ||
           message.find("initialize Higgs TTS codec graph context") != std::string::npos ||
           message.find("Higgs TTS codec graph compute failed") != std::string::npos ||
           message.find("Higgs TTS codec decoder produced fewer samples") != std::string::npos ||
           message.find("Higgs TTS codec decoder output trim is not symmetric") != std::string::npos;
}

HiggsAudioCodeMatrix slice_code_matrix(
    const HiggsAudioCodeMatrix & codes,
    int64_t begin_frame,
    int64_t end_frame) {
    if (begin_frame < 0 || end_frame <= begin_frame || end_frame > codes.frames || codes.codebooks <= 0) {
        throw std::runtime_error("Higgs TTS codec slice range is invalid");
    }
    HiggsAudioCodeMatrix out;
    out.frames = end_frame - begin_frame;
    out.codebooks = codes.codebooks;
    out.token_ids.reserve(static_cast<size_t>(out.frames * out.codebooks));
    for (int64_t frame = begin_frame; frame < end_frame; ++frame) {
        const auto start = codes.token_ids.begin() + static_cast<std::ptrdiff_t>(frame * codes.codebooks);
        out.token_ids.insert(
            out.token_ids.end(),
            start,
            start + static_cast<std::ptrdiff_t>(codes.codebooks));
    }
    return out;
}

class EncoderGraph {
public:
    EncoderGraph(
        std::shared_ptr<const HiggsAudioCodecWeights> weights,
        HiggsAudioCodecConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes,
        int64_t acoustic_samples,
        int64_t semantic_samples,
        int64_t frames)
        : weights_(std::move(weights)),
          config_(std::move(config)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          acoustic_sample_capacity_(acoustic_samples),
          semantic_sample_capacity_(semantic_samples),
          frame_capacity_(frames) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Higgs TTS codec encoder backend is not initialized");
        }
        if (acoustic_sample_capacity_ <= 0 || semantic_sample_capacity_ <= 0 || frame_capacity_ <= 0) {
            throw std::runtime_error("Higgs TTS codec encoder graph requires positive capacities");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS codec encoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.codec.encode", backend_type_};
        semantic_downsample_factor_ = semantic_downsample_factor(config_);
        acoustic_input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, acoustic_sample_capacity_}));
        semantic_input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, semantic_sample_capacity_}));
        downsample_indices_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frame_capacity_}));
        ggml_set_input(acoustic_input_.tensor);
        ggml_set_input(semantic_input_.tensor);
        ggml_set_input(downsample_indices_.tensor);

        auto hubert = semantic_input_;
        for (size_t i = 0; i < weights_->feature_extractor.convs.size(); ++i) {
            const auto & conv_weights = weights_->feature_extractor.convs[i];
            hubert = build_conv1d_im2col_f32(
                ctx,
                hubert,
                conv_weights,
                static_cast<int>(config_.semantic.conv_stride[i]),
                0,
                1);
            hubert = ensure_dense_layout(ctx, hubert);
            if (i == 0) {
                hubert = group_norm_affine(
                    ctx,
                    hubert,
                    config_.semantic.conv_dim.front(),
                    config_.semantic.layer_norm_eps,
                    weights_->feature_extractor.first_group_norm,
                    tensor_writer_);
            }
            hubert = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hubert);
        }
        hubert = transpose_bct_to_btc(ctx, hubert);
        if (config_.semantic.feat_proj_layer_norm) {
            hubert = modules::LayerNormModule({
                config_.semantic.conv_dim.back(),
                config_.semantic.layer_norm_eps,
                true,
                true,
            }).build(ctx, hubert, weights_->feature_projection.layer_norm);
        }
        hubert = modules::LinearModule({
            config_.semantic.conv_dim.back(),
            config_.semantic.hidden_size,
            true,
            GGML_PREC_F32,
        }).build(ctx, hubert, weights_->feature_projection.projection);

        auto position = build_grouped_positional_conv(
            ctx,
            transpose_btc_to_bct(ctx, hubert),
            weights_->positional_conv_groups,
            config_.semantic.hidden_size,
            config_.semantic.num_conv_pos_embeddings);
        hubert = modules::AddModule().build(ctx, hubert, transpose_bct_to_btc(ctx, position));
        hubert = modules::LayerNormModule({
            config_.semantic.hidden_size,
            config_.semantic.layer_norm_eps,
            true,
            true,
        }).build(ctx, hubert, weights_->encoder_input_layer_norm);
        hubert = build_hubert_sequence_mean(ctx, hubert, *weights_, config_);

        auto hubert_flat = core::reshape_tensor(
            ctx,
            ensure_dense_layout(ctx, hubert),
            core::TensorShape::from_dims({hubert.shape.dims[1], hubert.shape.dims[2]}));
        auto semantic_downsampled = modules::EmbeddingModule({
            hubert.shape.dims[1],
            hubert.shape.dims[2],
        }).build(ctx, downsample_indices_, hubert_flat);
        semantic_downsampled = core::reshape_tensor(
            ctx,
            semantic_downsampled,
            core::TensorShape::from_dims({1, frame_capacity_, config_.semantic.hidden_size}));
        auto semantic_encoded = build_semantic_encoder(ctx, transpose_btc_to_bct(ctx, semantic_downsampled), *weights_, config_);
        auto acoustic_encoded = build_acoustic_encoder(ctx, acoustic_input_, *weights_, config_);
        if (semantic_encoded.shape.dims[2] != acoustic_encoded.shape.dims[2]) {
            throw std::runtime_error("Higgs TTS semantic/acoustic encoder frame counts do not match");
        }
        auto embeddings_bct = modules::ConcatModule({1}).build(ctx, acoustic_encoded, semantic_encoded);
        auto embeddings_btc = transpose_bct_to_btc(ctx, embeddings_bct);
        embeddings_btc = modules::LinearModule({
            config_.hidden_size,
            config_.hidden_size,
            true,
            GGML_PREC_F32,
        }).build(ctx, embeddings_btc, weights_->fc);
        auto residual_bct = transpose_btc_to_bct(ctx, embeddings_btc);

        code_outputs_.reserve(weights_->quantizers.size());
        for (size_t quantizer_index = 0; quantizer_index < weights_->quantizers.size(); ++quantizer_index) {
            const auto & quantizer = weights_->quantizers[quantizer_index];
            auto residual_btc = transpose_bct_to_btc(ctx, residual_bct);
            auto projected = modules::LinearModule({
                config_.hidden_size,
                config_.codebook_dim,
                true,
                GGML_PREC_F32,
            }).build(ctx, residual_btc, quantizer.project_in);
            auto logits = modules::LinearModule({
                config_.codebook_dim,
                config_.codebook_size,
                true,
                GGML_PREC_F32,
            }).build(ctx, projected, quantizer.score);
            auto logits_flat = core::reshape_tensor(
                ctx,
                ensure_dense_layout(ctx, logits),
                core::TensorShape::from_dims({frame_capacity_, config_.codebook_size}));
            auto * ids_raw = ggml_argmax(ctx.ggml, logits_flat.tensor);
            ggml_set_output(ids_raw);
            code_outputs_.push_back(ids_raw);
            auto ids = core::reshape_tensor(
                ctx,
                core::wrap_tensor(ids_raw, core::TensorShape::from_dims({frame_capacity_}), GGML_TYPE_I32),
                core::TensorShape::from_dims({1, frame_capacity_}));
            auto embedded = modules::CodebookLookupModule({
                config_.codebook_size,
                config_.codebook_dim,
            }).build(ctx, ids, quantizer.codebook);
            auto quantized = modules::LinearModule({
                config_.codebook_dim,
                config_.hidden_size,
                true,
                GGML_PREC_F32,
            }).build(ctx, embedded, quantizer.project_out);
            auto quantized_bct = transpose_btc_to_bct(ctx, quantized);
            residual_bct = core::wrap_tensor(
                ggml_sub(ctx.ggml, residual_bct.tensor, quantized_bct.tensor),
                residual_bct.shape,
                GGML_TYPE_F32);
        }

        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (ggml_tensor * code_output : code_outputs_) {
            ggml_build_forward_expand(graph_, code_output);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Higgs TTS codec encoder graph");
        }
        tensor_writer_.flush();
        debug::timing_log_scalar("higgs_tts.codec.encoder.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("higgs_tts.codec.encoder.frames", frame_capacity_);
        debug::trace_log_scalar("higgs_tts.codec.encoder.acoustic_samples", acoustic_sample_capacity_);
        debug::trace_log_scalar("higgs_tts.codec.encoder.semantic_samples", semantic_sample_capacity_);
    }

    ~EncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        ggml_backend_t backend,
        int threads,
        int64_t acoustic_samples,
        int64_t semantic_samples,
        int64_t frames) const noexcept {
        return backend_ == backend &&
            threads_ == threads &&
            acoustic_sample_capacity_ >= acoustic_samples &&
            semantic_sample_capacity_ >= semantic_samples &&
            frame_capacity_ >= frames;
    }

    HiggsAudioCodeMatrix run(const NormalizedReferenceAudio & audio) {
        if (static_cast<int64_t>(audio.acoustic_samples_24k.size()) > acoustic_sample_capacity_ ||
            static_cast<int64_t>(audio.semantic_samples_16k_padded.size()) > semantic_sample_capacity_ ||
            audio.frames > frame_capacity_) {
            throw std::runtime_error("Higgs TTS codec encoder request exceeds prepared capacity");
        }
        auto timing_start = Clock::now();
        std::vector<float> acoustic_padded(static_cast<size_t>(acoustic_sample_capacity_), 0.0F);
        std::copy(audio.acoustic_samples_24k.begin(), audio.acoustic_samples_24k.end(), acoustic_padded.begin());
        core::write_tensor_f32(acoustic_input_, acoustic_padded);
        std::vector<float> semantic_padded(static_cast<size_t>(semantic_sample_capacity_), 0.0F);
        std::copy(audio.semantic_samples_16k_padded.begin(), audio.semantic_samples_16k_padded.end(), semantic_padded.begin());
        core::write_tensor_f32(semantic_input_, semantic_padded);
        std::vector<int32_t> downsample(static_cast<size_t>(frame_capacity_), 0);
        for (int64_t i = 0; i < audio.frames; ++i) {
            downsample[static_cast<size_t>(i)] = static_cast<int32_t>(i * semantic_downsample_factor_);
        }
        core::write_tensor_i32(downsample_indices_, downsample);
        tensor_writer_.flush();
        debug::timing_log_scalar("higgs_tts.codec.encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(backend_, threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("higgs_tts.codec.encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS codec encoder graph compute failed");
        }

        HiggsAudioCodeMatrix codes;
        codes.frames = audio.frames;
        codes.codebooks = static_cast<int64_t>(code_outputs_.size());
        codes.token_ids.assign(static_cast<size_t>(codes.frames * codes.codebooks), 0);
        timing_start = Clock::now();
        for (size_t codebook = 0; codebook < code_outputs_.size(); ++codebook) {
            std::vector<int32_t> values(static_cast<size_t>(frame_capacity_), 0);
            ggml_backend_tensor_get(
                code_outputs_[codebook],
                values.data(),
                0,
                values.size() * sizeof(int32_t));
            for (int64_t frame = 0; frame < codes.frames; ++frame) {
                codes.token_ids[static_cast<size_t>(frame * codes.codebooks + static_cast<int64_t>(codebook))] =
                    values[static_cast<size_t>(frame)];
            }
        }
        debug::timing_log_scalar("higgs_tts.codec.encoder.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return codes;
    }

private:
    std::shared_ptr<const HiggsAudioCodecWeights> weights_;
    HiggsAudioCodecConfig config_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t acoustic_sample_capacity_ = 0;
    int64_t semantic_sample_capacity_ = 0;
    int64_t frame_capacity_ = 0;
    int64_t semantic_downsample_factor_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::DeferredTensorWriter tensor_writer_;
    core::TensorValue acoustic_input_;
    core::TensorValue semantic_input_;
    core::TensorValue downsample_indices_;
    std::vector<ggml_tensor *> code_outputs_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class DecoderGraph {
public:
    DecoderGraph(
        std::shared_ptr<const HiggsAudioCodecWeights> weights,
        HiggsAudioCodecConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes,
        int64_t frames,
        int64_t codebooks)
        : weights_(std::move(weights)),
          config_(std::move(config)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          frames_(frames),
          codebooks_(codebooks) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Higgs TTS codec backend is not initialized");
        }
        if (frames_ <= 0 || codebooks_ <= 0) {
            throw std::runtime_error("Higgs TTS codec graph requires positive frame and codebook counts");
        }
        if (static_cast<size_t>(codebooks_) > weights_->quantizers.size()) {
            throw std::runtime_error("Higgs TTS codec graph codebook count exceeds loaded weights");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS codec graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "higgs_tts.codec.decode", backend_type_};
        code_inputs_.reserve(static_cast<size_t>(codebooks_));
        for (int64_t codebook = 0; codebook < codebooks_; ++codebook) {
            auto input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frames_}));
            ggml_set_input(input.tensor);
            code_inputs_.push_back(input.tensor);
        }
        auto latent_bct = build_quantizer_decode_sequence(ctx, code_inputs_, *weights_, frames_);
        auto acoustic_btc = modules::LinearModule({
            config_.hidden_size,
            config_.acoustic_hidden_size,
            true,
        }).build(ctx, transpose_bct_to_btc(ctx, latent_bct), weights_->fc2);
        auto audio = build_acoustic_decoder(ctx, transpose_btc_to_bct(ctx, acoustic_btc), *weights_, config_);
        const int64_t expected_samples = frames_ * product(config_.upsampling_ratios);
        if (audio.shape.dims[2] < expected_samples) {
            throw std::runtime_error("Higgs TTS codec decoder produced fewer samples than expected");
        }
        if (audio.shape.dims[2] != expected_samples) {
            const int64_t trim = audio.shape.dims[2] - expected_samples;
            if ((trim % 2) != 0) {
                throw std::runtime_error("Higgs TTS codec decoder output trim is not symmetric");
            }
            audio = modules::SliceModule({2, trim / 2, expected_samples}).build(ctx, audio);
        }
        output_ = audio.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS codec graph");
        }
        code_input_host_.resize(static_cast<size_t>(codebooks_));
        for (auto & ids : code_input_host_) {
            ids.assign(static_cast<size_t>(frames_), 0);
        }
        debug::timing_log_scalar("higgs_tts.codec.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("higgs_tts.codec.frames", frames_);
        debug::trace_log_scalar("higgs_tts.codec.samples", expected_samples);
    }

    ~DecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(ggml_backend_t backend, int threads, int64_t frames, int64_t codebooks) const noexcept {
        return backend_ == backend && threads_ == threads && frames_ == frames && codebooks_ == codebooks;
    }

    runtime::AudioBuffer run(const HiggsAudioCodeMatrix & codes) {
        if (codes.frames != frames_ || codes.codebooks != codebooks_) {
            throw std::runtime_error("Higgs TTS codec graph code matrix shape mismatch");
        }
        auto timing_start = Clock::now();
        for (int64_t codebook = 0; codebook < codebooks_; ++codebook) {
            auto & ids = code_input_host_[static_cast<size_t>(codebook)];
            for (int64_t frame = 0; frame < frames_; ++frame) {
                ids[static_cast<size_t>(frame)] =
                    codes.token_ids[static_cast<size_t>(frame * codebooks_ + codebook)];
            }
            ggml_backend_tensor_set(
                code_inputs_[static_cast<size_t>(codebook)],
                ids.data(),
                0,
                ids.size() * sizeof(int32_t));
        }
        debug::timing_log_scalar("higgs_tts.codec.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(backend_, threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("higgs_tts.codec.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS codec graph compute failed");
        }
        runtime::AudioBuffer audio;
        audio.sample_rate = config_.sample_rate;
        audio.channels = 1;
        const int64_t samples = frames_ * product(config_.upsampling_ratios);
        audio.samples.resize(static_cast<size_t>(samples));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, audio.samples.data(), 0, audio.samples.size() * sizeof(float));
        debug::timing_log_scalar("higgs_tts.codec.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return audio;
    }

private:
    std::shared_ptr<const HiggsAudioCodecWeights> weights_;
    HiggsAudioCodecConfig config_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t frames_ = 0;
    int64_t codebooks_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::vector<ggml_tensor *> code_inputs_;
    std::vector<std::vector<int32_t>> code_input_host_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class HiggsAudioCodecWeightsRuntime {
public:
    HiggsAudioCodecWeightsRuntime(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          config_(codec_config_from_assets(*assets_)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Higgs TTS codec runtime requires assets");
        }
        validate_codec_config(config_, assets_->config);
        weights_ = load_weights(*assets_, config_, backend_, backend_type_, weight_context_bytes, storage_type);
    }

    const HiggsAudioCodecConfig & config() const noexcept {
        return config_;
    }

    const std::shared_ptr<const HiggsAudioCodecWeights> & weights() const noexcept {
        return weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const HiggsTTSAssets> assets_;
    HiggsAudioCodecConfig config_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const HiggsAudioCodecWeights> weights_;
};

}  // namespace

struct HiggsAudioCodecDecoderRuntime::Impl {
    Impl(
        std::shared_ptr<const HiggsTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<HiggsAudioCodecWeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          graph_arena_bytes(graph_arena_bytes) {}

    HiggsAudioCodeMatrix encode_reference_audio(const runtime::AudioBuffer & audio) {
        const auto normalized = normalize_reference_audio(audio, weights->config());
        if (encoder_graph == nullptr ||
            !encoder_graph->matches(
                weights->backend(),
                weights->threads(),
                static_cast<int64_t>(normalized.acoustic_samples_24k.size()),
                static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()),
                normalized.frames)) {
            encoder_graph = std::make_unique<EncoderGraph>(
                weights->weights(),
                weights->config(),
                weights->backend(),
                weights->backend_type(),
                weights->threads(),
                graph_arena_bytes,
                static_cast<int64_t>(normalized.acoustic_samples_24k.size()),
                static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()),
                normalized.frames);
        } else {
            debug::timing_log_scalar("higgs_tts.codec.encoder.graph.build_ms", 0.0);
            debug::trace_log_scalar("higgs_tts.codec.encoder.frames", normalized.frames);
            debug::trace_log_scalar(
                "higgs_tts.codec.encoder.acoustic_samples",
                static_cast<int64_t>(normalized.acoustic_samples_24k.size()));
            debug::trace_log_scalar(
                "higgs_tts.codec.encoder.semantic_samples",
                static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()));
        }
        return encoder_graph->run(normalized);
    }

    runtime::AudioBuffer decode_with_current_graph(const HiggsAudioCodeMatrix & codes) {
        if (graph == nullptr ||
            !graph->matches(weights->backend(), weights->threads(), codes.frames, codes.codebooks)) {
            graph = std::make_unique<DecoderGraph>(
                weights->weights(),
                weights->config(),
                weights->backend(),
                weights->backend_type(),
                weights->threads(),
                graph_arena_bytes,
                codes.frames,
                codes.codebooks);
        } else {
            debug::timing_log_scalar("higgs_tts.codec.graph.build_ms", 0.0);
            debug::trace_log_scalar("higgs_tts.codec.frames", codes.frames);
            debug::trace_log_scalar("higgs_tts.codec.samples", codes.frames * product(weights->config().upsampling_ratios));
        }
        return graph->run(codes);
    }

    runtime::AudioBuffer decode_chunked(const HiggsAudioCodeMatrix & codes) {
        const int64_t samples_per_frame = product(weights->config().upsampling_ratios);
        int64_t chunk_frames = std::min<int64_t>(codec_decode_fallback_frames(), codes.frames);
        const int64_t configured_overlap = codec_decode_fallback_overlap_frames();
        std::string last_error;
        while (chunk_frames >= kCodecDecodeFallbackMinFrames) {
            try {
                runtime::AudioBuffer merged;
                graph.reset();
                for (int64_t start = 0; start < codes.frames; start += chunk_frames) {
                    const int64_t keep_frames = std::min<int64_t>(chunk_frames, codes.frames - start);
                    const int64_t overlap = std::min<int64_t>(
                        configured_overlap,
                        std::max<int64_t>(0, chunk_frames / 4));
                    const int64_t begin = std::max<int64_t>(0, start - overlap);
                    const int64_t end = std::min<int64_t>(codes.frames, start + keep_frames + overlap);
                    auto decoded = decode_with_current_graph(slice_code_matrix(codes, begin, end));
                    const int64_t sample_begin = (start - begin) * samples_per_frame;
                    const int64_t sample_count = keep_frames * samples_per_frame;
                    const int64_t sample_end = sample_begin + sample_count;
                    if (sample_begin < 0 || sample_end > static_cast<int64_t>(decoded.samples.size())) {
                        throw std::runtime_error("Higgs TTS chunked codec decode produced an invalid sample range");
                    }
                    runtime::AudioBuffer segment;
                    segment.sample_rate = decoded.sample_rate;
                    segment.channels = decoded.channels;
                    segment.samples.assign(
                        decoded.samples.begin() + static_cast<std::ptrdiff_t>(sample_begin),
                        decoded.samples.begin() + static_cast<std::ptrdiff_t>(sample_end));
                    runtime::append_audio_buffer(merged, segment);
                }
                graph.reset();
                debug::trace_log_scalar("higgs_tts.codec.chunked_decode_frames", chunk_frames);
                debug::trace_log_scalar("higgs_tts.codec.chunked_decode_overlap_frames", configured_overlap);
                return merged;
            } catch (const std::runtime_error & error) {
                if (!is_codec_graph_retryable_failure(error)) {
                    throw;
                }
                last_error = error.what();
                graph.reset();
                if (chunk_frames == kCodecDecodeFallbackMinFrames) {
                    break;
                }
                chunk_frames /= 2;
            }
        }
        throw std::runtime_error(last_error.empty() ? "failed to allocate Higgs TTS codec graph" : last_error);
    }

    runtime::AudioBuffer decode(const HiggsAudioCodeMatrix & raw_codes) {
        const auto codes = trim_decodable_codes(raw_codes, weights->config().codebook_size);
        try {
            return decode_with_current_graph(codes);
        } catch (const std::runtime_error & error) {
            if (!is_codec_graph_retryable_failure(error)) {
                throw;
            }
            debug::trace_log_scalar("higgs_tts.codec.chunked_decode_fallback", 1);
            debug::trace_log_scalar("higgs_tts.codec.chunked_decode_original_frames", codes.frames);
            debug::trace_log_scalar("higgs_tts.codec.chunked_decode_initial_frames", codec_decode_fallback_frames());
            graph.reset();
            return decode_chunked(codes);
        }
    }

    void release_runtime_cache() {
        encoder_graph.reset();
        graph.reset();
    }

    std::shared_ptr<HiggsAudioCodecWeightsRuntime> weights;
    size_t graph_arena_bytes = 0;
    std::unique_ptr<EncoderGraph> encoder_graph;
    std::unique_ptr<DecoderGraph> graph;
};

HiggsAudioCodecDecoderRuntime::HiggsAudioCodecDecoderRuntime(
    std::shared_ptr<const HiggsTTSAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

HiggsAudioCodecDecoderRuntime::~HiggsAudioCodecDecoderRuntime() = default;

HiggsAudioCodeMatrix HiggsAudioCodecDecoderRuntime::encode_reference_audio(const runtime::AudioBuffer & audio) {
    return impl_->encode_reference_audio(audio);
}

runtime::AudioBuffer HiggsAudioCodecDecoderRuntime::decode(const HiggsAudioCodeMatrix & raw_codes) {
    return impl_->decode(raw_codes);
}

void HiggsAudioCodecDecoderRuntime::release_runtime_cache() {
    impl_->release_runtime_cache();
}

}  // namespace engine::models::higgs_tts
