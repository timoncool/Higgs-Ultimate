import type { TtsModelPreset, WhisperModelPreset } from "./types";

export const APP_VERSION = "0.3.1";
// Original author (engine + upstream app) — credit preserved in the UI footer.
export const GITHUB_URL = "https://github.com/Saganaki22/Higgs-Audio-v3-Studio";
// This Russian portable fork.
export const FORK_URL = "https://github.com/timoncool/Higgs-Ultimate";
export const RELEASES_URL = `${FORK_URL}/releases`;
// Portable-build authors.
export const NERUAL_DREMING_URL = "https://t.me/nerual_dreming";
export const NEUROPORT_URL = "https://t.me/neuroport";
export const CUDA_DOWNLOAD_URL = "https://developer.nvidia.com/cuda-downloads";
export const NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx";
export const VC_REDIST_X64_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe";
export const HIGGS_MODEL_RESOLVE_BASE = "https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main";
export const HIGGS_RECOMMENDED_MODEL = "higgs-q8_0";
export const ENGINE_PACKAGE_URL = `${HIGGS_MODEL_RESOLVE_BASE}/engines`;
export const ENGINE_DLL_URL = `${ENGINE_PACKAGE_URL}/audiocpp_engine.dll`;
// ASR = Parakeet (NVIDIA parakeet-tdt-0.6b-v3, multilingual incl. Russian) via
// the parakeet-rs crate (ONNX Runtime, CPU). Models are ONNX file sets from
// istupakov's repo — a folder of files, not a single GGUF weight.
// (The WHISPER_* names are kept to avoid churn across the download machinery.)
export const WHISPER_MODELS_URL = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx";
export const WHISPER_MODEL_TREE_URL = `${WHISPER_MODELS_URL}/tree/main`;
export const WHISPER_MODEL_RESOLVE_BASE = `${WHISPER_MODELS_URL}/resolve/main`;
export const WHISPER_RECOMMENDED_MODEL = "tdt-0.6b-v3-int8";
export const SPEAKER_PERSONA_STORAGE_KEY = "higgsAudio.speakerPersonas";
export const MODEL_PATH_STORAGE_KEY = "higgsAudio.selectedModelPath";
export const ENGINE_PATH_STORAGE_KEY = "higgsAudio.selectedEnginePath";
export const MINIMIZE_TO_TRAY_STORAGE_KEY = "higgsAudio.minimizeToTray";
export const STREAM_PLAYBACK_STORAGE_KEY = "higgsAudio.streamPlayback";
export const API_LOG_STORAGE_KEY = "higgsAudio.apiLogs";
export const API_COMMAND_POPOUT_LABEL = "api-command-centre";
export const FIRST_RUN_WIZARD_STORAGE_KEY = "higgsAudio.setupWizardDismissed";
export const HIGGS_REFERENCE_MAX_SECONDS = 30;
export const HIGGS_MODEL_ASSET_FILES = [
  "chat_template.jinja",
  "config.json",
  "higgs_audio_v2_tokenizer_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
];

export const HIGGS_MODEL_PRESETS: TtsModelPreset[] = [
  {
    id: "higgs-q8_0",
    label: "Higgs Audio v3 Q8_0",
    folder: "higgs-q8_0",
    filename: "q8_0.gguf",
    size: "recommended · 12 GB VRAM",
    recommended: true,
  },
  {
    id: "higgs-q6_k",
    label: "Higgs Audio v3 Q6_K",
    folder: "higgs-q6_k",
    filename: "q6_k.gguf",
    size: "10 GB VRAM",
  },
  {
    id: "higgs-q5_k",
    label: "Higgs Audio v3 Q5_K",
    folder: "higgs-q5_k",
    filename: "q5_k.gguf",
    size: "9 GB VRAM",
  },
  {
    id: "higgs-q4_k_m",
    label: "Higgs Audio v3 Q4_K_M",
    folder: "higgs-q4_k_m",
    filename: "q4_k_m.gguf",
    size: "8 GB VRAM",
  },
  {
    id: "higgs-bf16",
    label: "Higgs Audio v3 BF16",
    folder: "higgs-bf16",
    filename: "bf16.gguf",
    size: "16 GB VRAM",
  },
];

// Parakeet TDT v3 ONNX variants from istupakov/parakeet-tdt-0.6b-v3-onnx (all
// multilingual, incl. RU). parakeet-rs loads a whole folder of ONNX files, so
// each variant lists the exact file set to fetch. The loader prefers the plain
// (fp32) filenames when present, so int8 lives in its own folder with only the
// .int8.onnx files — that's what makes int8 the one actually loaded.
//
// vocab.txt is required by the loader; config.json and nemo128.onnx are pulled
// for completeness/forward-compat even though the current loader hardcodes the
// preprocessor config.
export const WHISPER_MODEL_PRESETS: WhisperModelPreset[] = [
  {
    id: "tdt-0.6b-v3-int8",
    label: "Parakeet TDT v3 · INT8",
    folder: "tdt-0.6b-v3-int8",
    files: [
      "encoder-model.int8.onnx",
      "decoder_joint-model.int8.onnx",
      "vocab.txt",
      "config.json",
      "nemo128.onnx",
    ],
    size: "~670 MB · рекомендуется",
    recommended: true,
  },
  {
    id: "tdt-0.6b-v3-fp32",
    label: "Parakeet TDT v3 · FP32",
    folder: "tdt-0.6b-v3-fp32",
    files: [
      "encoder-model.onnx",
      "encoder-model.onnx.data",
      "decoder_joint-model.onnx",
      "vocab.txt",
      "config.json",
      "nemo128.onnx",
    ],
    size: "~2.5 GB · максимум качества",
  },
];
