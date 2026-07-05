#include "engine/framework/assets/tensor_source.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    engine::assets::TensorStorageType type = engine::assets::TensorStorageType::Q8_0;
    std::string policy = "higgs_tts";
};

std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char & ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

void usage() {
    std::cerr
        << "usage: quantize_safetensors --input model.safetensors --output model.gguf "
        << "--type f32|f16|bf16|q8_0|q6_k|q5_k|q4_0|q4_k --policy all|higgs_tts\n";
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (key == "--input") {
            args.input = require_value("--input");
        } else if (key == "--output") {
            args.output = require_value("--output");
        } else if (key == "--type") {
            args.type = engine::assets::parse_tensor_storage_type(require_value("--type"));
        } else if (key == "--policy") {
            args.policy = lower_ascii(require_value("--policy"));
        } else if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.input.empty() || args.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    if (args.policy != "all" && args.policy != "higgs_tts") {
        throw std::runtime_error("--policy must be all or higgs_tts");
    }
    return args;
}

ggml_type raw_dtype_to_ggml_type(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32") {
        return GGML_TYPE_F32;
    }
    if (normalized == "f16") {
        return GGML_TYPE_F16;
    }
    if (normalized == "bf16") {
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
    throw std::runtime_error("unsupported tensor dtype: " + std::string(dtype));
}

bool is_float_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    return normalized == "f32" || normalized == "f16" || normalized == "bf16";
}

bool is_higgs_projection_weight(std::string_view name) {
    if (!starts_with(name, "body.layers.")) {
        return false;
    }
    return ends_with(name, ".mlp.down_proj.weight") ||
           ends_with(name, ".mlp.gate_proj.weight") ||
           ends_with(name, ".mlp.up_proj.weight") ||
           ends_with(name, ".self_attn.k_proj.weight") ||
           ends_with(name, ".self_attn.o_proj.weight") ||
           ends_with(name, ".self_attn.q_proj.weight") ||
           ends_with(name, ".self_attn.v_proj.weight");
}

bool should_quantize(
    const engine::assets::TensorMetadata & meta,
    const std::string & policy,
    engine::assets::TensorStorageType target) {
    if (!is_float_dtype(meta.dtype) || meta.shape.size() < 2) {
        return false;
    }
    if (policy == "all") {
        return true;
    }
    if (is_higgs_projection_weight(meta.name)) {
        return true;
    }
    if (target == engine::assets::TensorStorageType::Q8_0) {
        return meta.name == "tied.embedding.modality_embeddings.0.embedding.weight" ||
               meta.name == "tied.embedding.text_embedding.weight";
    }
    return false;
}

std::vector<int64_t> ggml_ne_from_shape(const std::vector<int64_t> & shape) {
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) {
        throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    std::vector<int64_t> ne(GGML_MAX_DIMS, 1);
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t dim = shape[shape.size() - 1 - i];
        if (dim <= 0) {
            throw std::runtime_error("tensor shape contains a non-positive dimension");
        }
        ne[i] = dim;
    }
    return ne;
}

struct OutputTensor {
    std::string name;
    std::vector<int64_t> shape;
    ggml_type type = GGML_TYPE_F32;
    std::vector<std::byte> bytes;
    bool quantized = false;
};

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto source = engine::assets::open_tensor_source(args.input);
        auto tensors = source->tensors();
        std::sort(tensors.begin(), tensors.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.name < rhs.name;
        });

        std::vector<OutputTensor> output_tensors;
        output_tensors.reserve(tensors.size());
        size_t quantized_count = 0;
        size_t preserved_count = 0;
        size_t payload_bytes = 0;

        for (const auto & meta : tensors) {
            OutputTensor out;
            out.name = meta.name;
            out.shape = meta.shape;
            if (should_quantize(meta, args.policy, args.type)) {
                try {
                    const auto tensor = source->require_tensor(meta.name, args.type, meta.shape);
                    out.type = tensor.type;
                    out.bytes = std::move(tensor.bytes);
                    out.quantized = true;
                    ++quantized_count;
                    std::cout << "quantized " << meta.name << " " << meta.dtype
                              << " -> " << ggml_type_name(out.type) << "\n";
                } catch (const std::exception & e) {
                    const auto raw = source->require_tensor_data(meta.name);
                    out.type = raw_dtype_to_ggml_type(raw.metadata.dtype);
                    out.bytes = std::move(raw.bytes);
                    ++preserved_count;
                    std::cout << "preserved " << meta.name << " (" << e.what() << ")\n";
                }
            } else {
                const auto raw = source->require_tensor_data(meta.name);
                out.type = raw_dtype_to_ggml_type(raw.metadata.dtype);
                out.bytes = std::move(raw.bytes);
                ++preserved_count;
            }
            payload_bytes += out.bytes.size();
            output_tensors.push_back(std::move(out));
        }

        if (args.output.has_parent_path()) {
            std::filesystem::create_directories(args.output.parent_path());
        }

        gguf_context * gguf = gguf_init_empty();
        if (gguf == nullptr) {
            throw std::runtime_error("failed to create GGUF context");
        }

        const size_t tensor_meta_bytes = std::max<size_t>(16 * 1024 * 1024, output_tensors.size() * 4096);
        ggml_init_params params{};
        params.mem_size = tensor_meta_bytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * ggml = ggml_init(params);
        if (ggml == nullptr) {
            gguf_free(gguf);
            throw std::runtime_error("failed to create GGML metadata context");
        }

        std::vector<const char *> name_ptrs;
        std::vector<int32_t> ranks;
        name_ptrs.reserve(output_tensors.size());
        ranks.reserve(output_tensors.size());

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            const auto & tensor = output_tensors[i];
            const std::string storage_name = "t" + std::to_string(i);
            const auto ne = ggml_ne_from_shape(tensor.shape);
            ggml_tensor * ggml_tensor =
                ggml_new_tensor(ggml, tensor.type, static_cast<int>(tensor.shape.size()), ne.data());
            if (ggml_tensor == nullptr) {
                throw std::runtime_error("failed to create GGML tensor metadata: " + tensor.name);
            }
            ggml_set_name(ggml_tensor, storage_name.c_str());
            gguf_add_tensor(gguf, ggml_tensor);
            gguf_set_tensor_data(gguf, ggml_tensor->name, tensor.bytes.data());
            name_ptrs.push_back(tensor.name.c_str());
            ranks.push_back(static_cast<int32_t>(tensor.shape.size()));
        }

        gguf_set_val_str(gguf, "audiocpp.format", "tensor-archive");
        gguf_set_val_str(gguf, "audiocpp.source", args.input.string().c_str());
        gguf_set_val_str(gguf, "audiocpp.policy", args.policy.c_str());
        gguf_set_arr_str(gguf, "audiocpp.tensor_names", name_ptrs.data(), name_ptrs.size());
        gguf_set_arr_data(gguf, "audiocpp.tensor_ranks", GGUF_TYPE_INT32, ranks.data(), ranks.size());

        const bool ok = gguf_write_to_file(gguf, args.output.string().c_str(), false);
        ggml_free(ggml);
        gguf_free(gguf);
        if (!ok) {
            throw std::runtime_error("failed to write GGUF: " + args.output.string());
        }

        std::cout << "wrote " << output_tensors.size() << " tensors, quantized "
                  << quantized_count << ", preserved " << preserved_count
                  << ", payload bytes " << payload_bytes << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        usage();
        return 1;
    }
}
