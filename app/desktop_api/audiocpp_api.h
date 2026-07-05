#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*

  Audio.cpp Studio — C API for FFI from the Tauri/Rust backend.

  Ownership contract for audiocpp_audio_result:
  ------------------------------------------------
  After any audiocpp_generate_* call returns AUDIOCPP_OK:
    - `samples` is a heap-allocated float array of `sample_count` elements.
    - `error` is NULL.
  After any audiocpp_generate_* call returns non-OK:
    - `samples` is NULL, `sample_count` is 0.
    - `error` is a heap-allocated C-string describing the failure.

  In BOTH cases the caller MUST pass the result to audiocpp_free_result()
  when done.  That function calls free() on both pointers.  If you need to
  keep the data, memcpy/strdup BEFORE calling audiocpp_free_result().

  The result is safe to hold for as long as you like — it is an independent
  heap allocation, not tied to engine internals.

  Thread-safety:
  --------------
  A single audiocpp_engine is NOT safe to use from multiple threads
  concurrently.  The Rust side should serialise access.  Read-only queries
  (audiocpp_is_model_loaded, audiocpp_get_model_info, audiocpp_last_error)
  are safe to call concurrently with generation because they read atomic
  flags / immutable strings without taking the internal lock.

*/

#ifdef AUDIOCPP_API_EXPORT
#    if defined(_WIN32)
#        define AUDIOCPP_API __declspec(dllexport)
#    else
#        define AUDIOCPP_API __attribute__((visibility("default")))
#    endif
#else
#    define AUDIOCPP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct audiocpp_engine audiocpp_engine;

typedef enum audiocpp_status {
    AUDIOCPP_OK                = 0,
    AUDIOCPP_ERR_INVALID_PARAM = 1,
    AUDIOCPP_ERR_NOT_LOADED    = 2,
    AUDIOCPP_ERR_RUNTIME       = 3,
    AUDIOCPP_ERR_UNSUPPORTED   = 4,
    AUDIOCPP_ERR_CANCELLED     = 5,
} audiocpp_status;

typedef enum audiocpp_backend {
    AUDIOCPP_BACKEND_BEST   = 0,
    AUDIOCPP_BACKEND_CPU    = 1,
    AUDIOCPP_BACKEND_CUDA   = 2,
    AUDIOCPP_BACKEND_VULKAN = 3,
    AUDIOCPP_BACKEND_METAL  = 4,
} audiocpp_backend;

/* Result of a generate call — see ownership contract above. */
typedef struct audiocpp_audio_result {
    int32_t  sample_rate;
    int32_t  channels;
    size_t   sample_count;
    float *  samples;   /* heap-allocated, caller-owned, free with audiocpp_free_result */
    char *   error;     /* heap-allocated if non-NULL, caller-owned, free with audiocpp_free_result */
} audiocpp_audio_result;

typedef struct audiocpp_model_info {
    const char * family;
    const char * display_name;
    const char * weight_type;
    const char * model_root;
} audiocpp_model_info;

/* Progress callback: current step, total steps, phase label, user data. */
typedef void (*audiocpp_progress_fn)(int32_t current, int32_t total, const char * phase, void * user_data);

/* Streaming audio callback. `samples` is only valid for the duration of the callback. */
typedef void (*audiocpp_audio_chunk_fn)(
    int32_t sample_rate,
    int32_t channels,
    int64_t start_sample,
    const float * samples,
    size_t sample_count,
    bool is_final,
    void * user_data);

/* --- lifecycle --- */

AUDIOCPP_API audiocpp_engine * audiocpp_create(void);
AUDIOCPP_API void              audiocpp_destroy(audiocpp_engine * engine);

/* --- model management --- */

AUDIOCPP_API audiocpp_status audiocpp_load_model(
    audiocpp_engine *    engine,
    const char *         model_root,
    audiocpp_backend     backend,
    int32_t              device,
    int32_t              threads,
    const char *         weight_type,
    const char *         session_options_json);

AUDIOCPP_API void audiocpp_unload_model(audiocpp_engine * engine);

/* Read-only queries — safe to call concurrently with generation (read atomics). */
AUDIOCPP_API bool audiocpp_is_model_loaded(const audiocpp_engine * engine);
AUDIOCPP_API bool audiocpp_is_generating(const audiocpp_engine * engine);

AUDIOCPP_API audiocpp_status audiocpp_get_model_info(
    const audiocpp_engine * engine,
    audiocpp_model_info *   out_info);

/* Request cancellation of an in-flight generation. The generate call will
   return AUDIOCPP_ERR_CANCELLED as soon as the next progress check fires. */
AUDIOCPP_API void audiocpp_cancel(audiocpp_engine * engine);

/* --- generation --- */

AUDIOCPP_API audiocpp_status audiocpp_generate_tts(
    audiocpp_engine *       engine,
    const char *            text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result);

AUDIOCPP_API audiocpp_status audiocpp_generate_voice_clone(
    audiocpp_engine *       engine,
    const char *            text,
    const char *            ref_audio_path,
    const char *            ref_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result);

AUDIOCPP_API audiocpp_status audiocpp_generate_finish_sentence(
    audiocpp_engine *       engine,
    const char *            audio_path,
    const char *            continuation_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    void *                  user_data,
    audiocpp_audio_result * out_result);

AUDIOCPP_API audiocpp_status audiocpp_generate_tts_stream(
    audiocpp_engine *       engine,
    const char *            text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result);

AUDIOCPP_API audiocpp_status audiocpp_generate_voice_clone_stream(
    audiocpp_engine *       engine,
    const char *            text,
    const char *            ref_audio_path,
    const char *            ref_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result);

AUDIOCPP_API audiocpp_status audiocpp_generate_finish_sentence_stream(
    audiocpp_engine *       engine,
    const char *            audio_path,
    const char *            continuation_text,
    const char *            options_json,
    audiocpp_progress_fn    progress,
    audiocpp_audio_chunk_fn audio_chunk,
    void *                  user_data,
    audiocpp_audio_result * out_result);

/* --- result / error helpers --- */

/* Frees result->samples and result->error (both heap-allocated). Safe on NULL. */
AUDIOCPP_API void audiocpp_free_result(audiocpp_audio_result * result);

AUDIOCPP_API const char * audiocpp_last_error(const audiocpp_engine * engine);
AUDIOCPP_API const char * audiocpp_version(void);

/* One-shot transcription: loads whisper model, transcribes the WAV, frees the
   model immediately to release VRAM. Returns AUDIOCPP_OK on success with the
   transcribed text in out_text. */
AUDIOCPP_API audiocpp_status audiocpp_transcribe(
    audiocpp_engine *    engine,
    const char *         whisper_model_path,
    const char *         wav_path,
    const char *         language,
    char *               out_text,
    size_t               out_text_size);

#ifdef __cplusplus
}
#endif
