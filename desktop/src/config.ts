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
export const WHISPER_MODELS_URL = "https://huggingface.co/ggerganov/whisper.cpp";
export const WHISPER_MODEL_TREE_URL = `${WHISPER_MODELS_URL}/tree/main`;
export const WHISPER_MODEL_RESOLVE_BASE = `${WHISPER_MODELS_URL}/resolve/main`;
// Multilingual default (not the English-only .en models) so voice-clone
// auto-transcription works for Russian and other languages.
export const WHISPER_RECOMMENDED_MODEL = "large-v3-turbo-q5_0";
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
  { id: "tiny", size: "75 MiB", sha: "bd577a113a864445d4c299885e0cb97d4ba92b5f" },
  { id: "tiny-q5_1", size: "31 MiB", sha: "2827a03e495b1ed3048ef28a6a4620537db4ee51" },
  { id: "tiny-q8_0", size: "42 MiB", sha: "19e8118f6652a650569f5a949d962154e01571d9" },
  { id: "tiny.en", size: "75 MiB", sha: "c78c86eb1a8faa21b369bcd33207cc90d64ae9df" },
  { id: "tiny.en-q5_1", size: "31 MiB", sha: "3fb92ec865cbbc769f08137f22470d6b66e071b6" },
  { id: "tiny.en-q8_0", size: "42 MiB", sha: "802d6668e7d411123e672abe4cb6c18f12306abb" },
  { id: "base", size: "142 MiB", sha: "465707469ff3a37a2b9b8d8f89f2f99de7299dac" },
  { id: "base-q5_1", size: "57 MiB", sha: "a3733eda680ef76256db5fc5dd9de8629e62c5e7" },
  { id: "base-q8_0", size: "78 MiB", sha: "7bb89bb49ed6955013b166f1b6a6c04584a20fbe" },
  { id: "base.en", size: "142 MiB", sha: "137c40403d78fd54d454da0f9bd998f78703390c" },
  { id: "base.en-q5_1", size: "57 MiB", sha: "d26d7ce5a1b6e57bea5d0431b9c20ae49423c94a" },
  { id: "base.en-q8_0", size: "78 MiB", sha: "bb1574182e9b924452bf0cd1510ac034d323e948" },
  { id: "small", size: "466 MiB", sha: "55356645c2b361a969dfd0ef2c5a50d530afd8d5" },
  { id: "small-q5_1", size: "181 MiB", sha: "6fe57ddcfdd1c6b07cdcc73aaf620810ce5fc771" },
  { id: "small-q8_0", size: "252 MiB", sha: "bcad8a2083f4e53d648d586b7dbc0cd673d8afad" },
  { id: "small.en", size: "466 MiB", sha: "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022" },
  { id: "small.en-q5_1", size: "181 MiB", sha: "20f54878d608f94e4a8ee3ae56016571d47cba34" },
  { id: "small.en-q8_0", size: "252 MiB", sha: "9d75ff4ccfa0a8217870d7405cf8cef0a5579852" },
  { id: "small.en-tdrz", size: "465 MiB", sha: "b6c6e7e89af1a35c08e6de56b66ca6a02a2fdfa1" },
  { id: "medium", size: "1.5 GiB", sha: "fd9727b6e1217c2f614f9b698455c4ffd82463b4" },
  { id: "medium-q5_0", size: "514 MiB", sha: "7718d4c1ec62ca96998f058114db98236937490e" },
  { id: "medium-q8_0", size: "785 MiB", sha: "e66645948aff4bebbec71b3485c576f3d63af5d6" },
  { id: "medium.en", size: "1.5 GiB", sha: "8c30f0e44ce9560643ebd10bbe50cd20eafd3723" },
  { id: "medium.en-q5_0", size: "514 MiB", sha: "bb3b5281bddd61605d6fc76bc5b92d8f20284c3b" },
  { id: "medium.en-q8_0", size: "785 MiB", sha: "b1cf48c12c807e14881f634fb7b6c6ca867f6b38" },
  { id: "large-v1", size: "2.9 GiB", sha: "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299" },
  { id: "large-v2", size: "2.9 GiB", sha: "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6" },
  { id: "large-v2-q5_0", size: "1.1 GiB", sha: "00e39f2196344e901b3a2bd5814807a769bd1630" },
  { id: "large-v2-q8_0", size: "1.5 GiB", sha: "da97d6ca8f8ffbeeb5fd147f79010eeea194ba38" },
  { id: "large-v3", size: "2.9 GiB", sha: "ad82bf6a9043ceed055076d0fd39f5f186ff8062" },
  { id: "large-v3-q5_0", size: "1.1 GiB", sha: "e6e2ed78495d403bef4b7cff42ef4aaadcfea8de" },
  { id: "large-v3-turbo", size: "1.5 GiB", sha: "4af2b29d7ec73d781377bfd1758ca957a807e941" },
  { id: "large-v3-turbo-q5_0", size: "547 MiB", sha: "e050f7970618a659205450ad97eb95a18d69c9ee", recommended: true },
  { id: "large-v3-turbo-q8_0", size: "834 MiB", sha: "01bf15bedffe9f39d65c1b6ff9b687ea91f59e0e" },
];
