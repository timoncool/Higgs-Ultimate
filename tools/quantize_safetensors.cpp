#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/safetensors.h"

#include <gguf.h>
#include <ggml.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class OutputFormat {
    SafeTensors,
    Gguf,
};

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    engine::assets::TensorStorageType target_type = engine::assets::TensorStorageType::Q8_0;
    std::string policy = "all";
    OutputFormat output_format = OutputFormat::SafeTensors;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgufContextDeleter {
    void operator()(gguf_context * ctx) const noexcept {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char & ch : out) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return out;
}

OutputFormat output_format_from_path(const std::filesystem::path & path) {
    const std::string extension = lower_ascii(path.extension().string());
    if (extension == ".safetensors") {
        return OutputFormat::SafeTensors;
    }
    if (extension == ".gguf") {
        return OutputFormat::Gguf;
    }
    throw std::runtime_error("output extension must be .safetensors or .gguf");
}

bool is_higgs_tts_projection_weight(std::string_view name) {
    if (!starts_with(name, "body.layers.")) {
        return false;
    }
    return ends_with(name, ".self_attn.q_proj.weight") ||
           ends_with(name, ".self_attn.k_proj.weight") ||
           ends_with(name, ".self_attn.v_proj.weight") ||
           ends_with(name, ".self_attn.o_proj.weight") ||
           ends_with(name, ".mlp.gate_proj.weight") ||
           ends_with(name, ".mlp.up_proj.weight") ||
           ends_with(name, ".mlp.down_proj.weight");
}

bool is_higgs_tts_embedding_weight(std::string_view name) {
    return name == "tied.embedding.text_embedding.weight" ||
           name == "tied.embedding.modality_embeddings.0.embedding.weight";
}

bool is_supported_source_dtype(std::string_view dtype) {
    try {
        (void) engine::assets::tensor_storage_type_for_dtype(dtype);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool shape_can_quantize(const std::vector<int64_t> & shape, ggml_type target_type) {
    if (!ggml_is_quantized(target_type)) {
        return false;
    }
    if (ggml_quantize_requires_imatrix(target_type)) {
        return false;
    }
    if (shape.size() < 2) {
        return false;
    }
    const int64_t last_dim = shape.back();
    return last_dim > 0 && last_dim % ggml_blck_size(target_type) == 0;
}

bool should_quantize_higgs_tts(
    std::string_view name,
    engine::assets::TensorStorageType target_type) {
    if (is_higgs_tts_projection_weight(name)) {
        return true;
    }
    // Q8_0 embeddings are already exercised by the Higgs runtime weight_type=q8_0
    // path. Q4 embeddings are intentionally left native unless that path is
    // validated, because embeddings use ggml_get_rows rather than ggml_mul_mat.
    return target_type == engine::assets::TensorStorageType::Q8_0 &&
           is_higgs_tts_embedding_weight(name);
}

std::string dtype_for_storage(engine::assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case engine::assets::TensorStorageType::F32:
            return "F32";
        case engine::assets::TensorStorageType::F16:
            return "F16";
        case engine::assets::TensorStorageType::BF16:
            return "BF16";
        case engine::assets::TensorStorageType::Q4_0:
            return "Q4_0";
        case engine::assets::TensorStorageType::Q4_1:
            return "Q4_1";
        case engine::assets::TensorStorageType::Q5_0:
            return "Q5_0";
        case engine::assets::TensorStorageType::Q5_1:
            return "Q5_1";
        case engine::assets::TensorStorageType::Q4_K:
            return "Q4_K";
        case engine::assets::TensorStorageType::Q5_K:
            return "Q5_K";
        case engine::assets::TensorStorageType::Q6_K:
            return "Q6_K";
        case engine::assets::TensorStorageType::Q8_0:
            return "Q8_0";
        case engine::assets::TensorStorageType::Native:
            break;
    }
    throw std::runtime_error("native is not a writable quantized dtype");
}

ggml_type ggml_type_for_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32" || normalized == "float32") {
        return GGML_TYPE_F32;
    }
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") {
        return GGML_TYPE_F16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return GGML_TYPE_BF16;
    }
    if (normalized == "i64") {
        return GGML_TYPE_I64;
    }
    if (normalized == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (normalized == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (normalized == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (normalized == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (normalized == "q4_k") {
        return GGML_TYPE_Q4_K;
    }
    if (normalized == "q5_k") {
        return GGML_TYPE_Q5_K;
    }
    if (normalized == "q6_k") {
        return GGML_TYPE_Q6_K;
    }
    if (normalized == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    throw std::runtime_error("unsupported output tensor dtype for GGUF: " + std::string(dtype));
}

std::vector<unsigned char> to_unsigned_bytes(const std::vector<std::byte> & bytes) {
    std::vector<unsigned char> out(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(out.data(), bytes.data(), bytes.size());
    }
    return out;
}

bool should_quantize(
    const engine::assets::TensorMetadata & metadata,
    const Options & options,
    ggml_type target_ggml_type) {
    if (!is_supported_source_dtype(metadata.dtype)) {
        return false;
    }
    if (!shape_can_quantize(metadata.shape, target_ggml_type)) {
        return false;
    }
    if (options.policy == "all") {
        return true;
    }
    if (options.policy == "higgs_tts") {
        return should_quantize_higgs_tts(metadata.name, options.target_type);
    }
    throw std::runtime_error("unknown quantization policy: " + options.policy);
}

void write_gguf_file(
    const std::filesystem::path & output,
    const std::vector<engine::io::SafeTensorWriteEntry> & entries) {
    if (entries.empty()) {
        throw std::runtime_error("GGUF writer requires at least one tensor");
    }
    gguf_context * raw_gguf = gguf_init_empty();
    if (raw_gguf == nullptr) {
        throw std::runtime_error("failed to initialize GGUF writer");
    }
    std::unique_ptr<gguf_context, GgufContextDeleter> gguf(raw_gguf);

    gguf_set_val_str(gguf.get(), "general.architecture", "higgs_tts");
    gguf_set_val_str(gguf.get(), "general.name", "higgs_audio_v3_tts_4b");
    gguf_set_val_str(gguf.get(), "audiocpp.tensor_container", "gguf");
    gguf_set_val_u32(gguf.get(), "audiocpp.format_version", 1);

    const size_t context_bytes = (entries.size() + 1) * ggml_tensor_overhead() + 1024 * 1024;
    ggml_init_params params{context_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize GGUF tensor metadata context");
    }

    std::vector<int32_t> ranks;
    std::vector<const char *> original_names;
    ranks.reserve(entries.size());
    original_names.reserve(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto & entry = entries[index];
        if (entry.shape.empty() || entry.shape.size() > GGML_MAX_DIMS) {
            throw std::runtime_error("GGUF tensor rank must be between 1 and 4: " + entry.name);
        }
        ranks.push_back(static_cast<int32_t>(entry.shape.size()));
        original_names.push_back(entry.name.c_str());
        std::array<int64_t, GGML_MAX_DIMS> dims = {1, 1, 1, 1};
        for (size_t i = 0; i < entry.shape.size(); ++i) {
            dims[i] = entry.shape[entry.shape.size() - 1 - i];
        }

        const ggml_type type = ggml_type_for_dtype(entry.dtype);
        ggml_tensor * tensor = ggml_new_tensor(ctx.get(), type, static_cast<int>(entry.shape.size()), dims.data());
        if (tensor == nullptr) {
            throw std::runtime_error("failed to create GGUF tensor metadata: " + entry.name);
        }
        const std::string stored_name = "t" + std::to_string(index);
        ggml_set_name(tensor, stored_name.c_str());
        if (ggml_nbytes(tensor) != entry.data.size()) {
            throw std::runtime_error("GGUF tensor byte size mismatch: " + entry.name);
        }
        gguf_add_tensor(gguf.get(), tensor);
        gguf_set_tensor_data(gguf.get(), stored_name.c_str(), entry.data.data());
    }

    gguf_set_arr_str(
        gguf.get(),
        "audiocpp.tensor_names",
        original_names.data(),
        original_names.size());
    gguf_set_arr_data(
        gguf.get(),
        "audiocpp.tensor_ranks",
        GGUF_TYPE_INT32,
        ranks.data(),
        ranks.size());

    if (!gguf_write_to_file(gguf.get(), output.string().c_str(), false)) {
        throw std::runtime_error("failed to write GGUF file: " + output.string());
    }
}

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--input") {
            options.input = require_value("--input");
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else if (arg == "--type") {
            options.target_type = engine::assets::parse_tensor_storage_type(require_value("--type"));
        } else if (arg == "--policy") {
            options.policy = require_value("--policy");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: quantize_safetensors --input model.safetensors --output model.safetensors|model.gguf "
                   "--type q8_0|q4_0|q4_k --policy all|higgs_tts\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    options.output_format = output_format_from_path(options.output);
    const ggml_type target_ggml_type = engine::assets::ggml_type_for_tensor_storage(options.target_type);
    if (!ggml_is_quantized(target_ggml_type)) {
        throw std::runtime_error("--type must be a quantized ggml storage type");
    }
    if (ggml_quantize_requires_imatrix(target_ggml_type)) {
        throw std::runtime_error("--type requires an importance matrix and is not supported by this tool");
    }
    if (options.policy != "all" && options.policy != "higgs_tts") {
        throw std::runtime_error("--policy must be all or higgs_tts");
    }
    return options;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto source = engine::assets::open_tensor_source(options.input);
        const auto metadata = source->tensors();
        const ggml_type target_ggml_type = engine::assets::ggml_type_for_tensor_storage(options.target_type);
        const std::string target_dtype = dtype_for_storage(options.target_type);

        std::vector<engine::io::SafeTensorWriteEntry> entries;
        entries.reserve(metadata.size());
        size_t quantized = 0;
        size_t preserved = 0;
        uint64_t output_bytes = 0;

        std::cout << "input: " << options.input.string() << "\n";
        std::cout << "output: " << options.output.string() << "\n";
        std::cout << "format: " << (options.output_format == OutputFormat::Gguf ? "gguf" : "safetensors") << "\n";
        std::cout << "target: " << target_dtype << "\n";
        std::cout << "policy: " << options.policy << "\n";

        for (const auto & tensor : metadata) {
            engine::io::SafeTensorWriteEntry entry;
            entry.name = tensor.name;
            entry.shape = tensor.shape;

            if (should_quantize(tensor, options, target_ggml_type)) {
                const auto data = source->require_tensor(tensor.name, options.target_type, tensor.shape);
                entry.dtype = target_dtype;
                entry.data = to_unsigned_bytes(data.bytes);
                ++quantized;
                std::cout << "quantized " << tensor.name << " " << tensor.dtype << " -> " << target_dtype << "\n";
            } else {
                const auto raw = source->require_tensor_data(tensor.name);
                entry.dtype = raw.metadata.dtype;
                entry.data = to_unsigned_bytes(raw.bytes);
                ++preserved;
            }

            output_bytes += static_cast<uint64_t>(entry.data.size());
            entries.push_back(std::move(entry));
        }

        if (options.output_format == OutputFormat::Gguf) {
            write_gguf_file(options.output, entries);
        } else {
            engine::io::write_safetensors_file(options.output, entries);
        }

        std::cout << "wrote " << entries.size() << " tensors, quantized " << quantized
                  << ", preserved " << preserved << ", payload bytes " << output_bytes << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "quantize_safetensors: " << ex.what() << "\n";
        return 1;
    }
}
