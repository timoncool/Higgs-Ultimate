import type { TtsModelPreset, WhisperModelPreset } from "./types";

export const APP_VERSION = "0.2.4";
// Original author (engine + upstream app) — credit preserved in the UI footer.
export const GITHUB_URL = "https://github.com/Saganaki22/Higgs-Audio-v3-Studio";
// This Russian portable fork.
export const FORK_URL = "https://github.com/timoncool/Higgs-Audio-v3-Studio-rus";
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
// parakeet.cpp — replaces whisper. Single GGUF from mudler's official repo.
// (The WHISPER_* names are kept to avoid churn across the download machinery.)
export const WHISPER_MODELS_URL = "https://huggingface.co/mudler/parakeet-cpp-gguf";
export const WHISPER_MODEL_TREE_URL = `${WHISPER_MODELS_URL}/tree/main`;
export const WHISPER_MODEL_RESOLVE_BASE = `${WHISPER_MODELS_URL}/resolve/main`;
export const WHISPER_RECOMMENDED_MODEL = "tdt-0.6b-v3-q8_0";
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

export const WHISPER_MODEL_PRESETS: WhisperModelPreset[] = [
  { id: "tdt-0.6b-v3-q8_0", size: "940 MB · рус/мультиязычный", sha: "", recommended: true },
];
