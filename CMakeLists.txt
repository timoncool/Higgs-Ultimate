cmake_minimum_required(VERSION 3.20)

project(AudioCpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(ENGINE_DEFAULT_ENABLE_CUDA OFF)
set(ENGINE_DEFAULT_ENABLE_VULKAN OFF)
set(ENGINE_DEFAULT_ENABLE_METAL OFF)
set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ON)
set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ON)
if (APPLE)
    set(ENGINE_DEFAULT_ENABLE_METAL ON)
endif()
if (DEFINED GGML_CUDA)
    set(ENGINE_DEFAULT_ENABLE_CUDA ${GGML_CUDA})
endif()
if (DEFINED GGML_VULKAN)
    set(ENGINE_DEFAULT_ENABLE_VULKAN ${GGML_VULKAN})
endif()
if (DEFINED GGML_METAL)
    set(ENGINE_DEFAULT_ENABLE_METAL ${GGML_METAL})
endif()
if (DEFINED GGML_LLAMAFILE)
    set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ${GGML_LLAMAFILE})
endif()
if (DEFINED GGML_CUDA_GRAPHS)
    set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ${GGML_CUDA_GRAPHS})
endif()

option(ENGINE_ENABLE_CUDA "Build ggml with CUDA backend support" ${ENGINE_DEFAULT_ENABLE_CUDA})
option(ENGINE_ENABLE_VULKAN "Build ggml with Vulkan backend support" ${ENGINE_DEFAULT_ENABLE_VULKAN})
option(ENGINE_ENABLE_METAL "Build ggml with Metal backend support" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM support" ${ENGINE_DEFAULT_ENABLE_LLAMAFILE})
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs support" ${ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS})
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP support" ON)
option(ENGINE_BUILD_DESKTOP_API "Build the audiocpp_engine shared library for the Tauri desktop app" OFF)

if (ENGINE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "Keep upstream warnings quiet in this sample" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
set(GGML_CUDA ${ENGINE_ENABLE_CUDA} CACHE BOOL "Build ggml with CUDA backend support" FORCE)
set(GGML_VULKAN ${ENGINE_ENABLE_VULKAN} CACHE BOOL "Build ggml with Vulkan backend support" FORCE)
set(GGML_METAL ${ENGINE_ENABLE_METAL} CACHE BOOL "Build ggml with Metal backend support" FORCE)
set(GGML_LLAMAFILE ${ENGINE_ENABLE_LLAMAFILE} CACHE BOOL "Build ggml with llamafile SGEMM support" FORCE)
set(GGML_CUDA_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml CUDA graphs support" FORCE)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build sentencepiece shared libs" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "Do not build sentencepiece tests" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Do not probe for tcmalloc" FORCE)

add_subdirectory(external/ggml)
add_subdirectory(external/sentencepiece EXCLUDE_FROM_ALL)

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-pedantic-errors>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->
)

add_library(engine_runtime STATIC
    src/framework/assets/resource_bundle.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/deferred_tensor_writer.cpp
    src/framework/core/execution_context.cpp
    src/framework/runtime/registry.cpp
    src/framework/runtime/model.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/artifacts.cpp
    src/framework/runtime/cache.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/graph_optimizer.cpp
    src/framework/runtime/host_ops.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session_base.cpp
    src/framework/runtime/workspace.cpp
    src/framework/io/filesystem.cpp
    src/framework/io/config.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chunking.cpp
    src/framework/io/yaml.cpp
    src/framework/io/binary.cpp
    src/framework/io/safetensors.cpp
    src/framework/debug/trace.cpp
    src/framework/debug/profiler.cpp
    src/framework/decoders/tdt_decoder_runner.cpp
    src/framework/decoders/tdt_decoder_greedy_customized.cpp
    src/framework/decoders/tdt_decoder_greedy_duration_loop.cpp
    src/framework/decoders/tdt_decoder_greedy_separate_heads.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/whisper_embedding.cpp
    src/framework/modules/campplus_encoder.cpp
    src/framework/modules/bigvgan_vocoder.cpp
    src/framework/modules/hift_vocoder.cpp
    src/framework/modules/hubert_encoder.cpp
    src/framework/modules/wavlm_encoder.cpp
    src/framework/modules/optimizations/fast_conv_modules.cpp
    src/framework/modules/recurrent_modules.cpp
    src/framework/modules/streaming_conv_modules.cpp
    src/framework/modules/attention/positional_encoding.cpp
    src/framework/modules/attention/feed_forward.cpp
    src/framework/modules/attention/self_attention.cpp
    src/framework/modules/attention/longformer_attention.cpp
    src/framework/modules/attention/longformer_attention_gpu.cpp
    src/framework/modules/attention/common_relative_attention.cpp
    src/framework/modules/attention/relative_attention.cpp
    src/framework/modules/attention/qwen_decoder.cpp
    src/framework/modules/attention/cross_attention.cpp
    src/framework/modules/attention/transformer_blocks.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/conformer_modules.cpp
    src/framework/modules/predictor_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/hf_tokenizer_json.cpp
    src/framework/tokenizers/llama_bpe.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/tokenizers/sentencepiece.cpp
    src/framework/audio/fft.cpp
    src/framework/audio/activity.cpp
    src/framework/audio/conversion.cpp
    src/framework/audio/deepfilternet2.cpp
    src/framework/audio/dsp.cpp
    src/framework/audio/flashsr.cpp
    src/framework/audio/istft_graph.cpp
    src/framework/audio/mixing.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/output.cpp
    src/framework/audio/rnnoise.cpp
    src/framework/audio/utility_api.cpp
    src/framework/audio/waveform_ops.cpp
    src/framework/audio/wav_reader.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/audio/zipenhancer.cpp
    src/framework/modules/ecapa_tdnn_runtime.cpp
    src/framework/modules/titanet_runtime.cpp
    src/framework/codecs/neural_audio.cpp
    src/framework/sampling/decode_modules.cpp
    src/framework/sampling/diffusion_math.cpp
    src/framework/sampling/noise.cpp
    src/framework/sampling/torch_random.cpp
    src/models/higgs_tts/assets.cpp
    src/models/higgs_tts/audio_codes.cpp
    src/models/higgs_tts/codec.cpp
    src/models/higgs_tts/generator.cpp
    src/models/higgs_tts/tokenizer.cpp
    src/models/higgs_tts/session.cpp
    src/models/higgs_tts/loader.cpp
)

add_library(cjson_vendor STATIC
    external/cJSON/cJSON.c
)

target_include_directories(cjson_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
)

add_library(yaml_vendor STATIC
    external/libyaml/src/api.c
    external/libyaml/src/reader.c
    external/libyaml/src/scanner.c
    external/libyaml/src/parser.c
    external/libyaml/src/loader.c
)

target_include_directories(yaml_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include
)

target_compile_definitions(yaml_vendor PUBLIC
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING=\"0.2.5\"
)

target_include_directories(engine_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(engine_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/sentencepiece/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
)

target_link_libraries(engine_runtime PUBLIC ggml)
target_link_libraries(engine_runtime PRIVATE sentencepiece cjson_vendor yaml_vendor)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(engine_runtime PRIVATE OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_CUDA)
    target_sources(engine_runtime PRIVATE
        src/framework/audio/istft_cuda_runtime.cu
    )
    set_source_files_properties(src/framework/audio/istft_cuda_runtime.cu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_ISTFT=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart CUDA::cufft)
endif()

if (ENGINE_BUILD_DESKTOP_API)
    add_subdirectory(app/desktop_api)
endif()
