export type Mode = "tts" | "clone" | "speakers" | "finish" | "multi" | "api" | "history" | "batch" | "transcript";

// Один сегмент результата диаризации/транскрипции (camelCase из Rust).
export type TranscriptSegment = {
  start: number;
  end: number;
  text: string;
  speaker: number;
};

export type TranscriptResult = {
  segments: TranscriptSegment[];
  nSpeakers: number;
  diarized: boolean;
};
export type SaveFormat = "wav" | "mp3";
export type DownloadKind = "model" | "whisper" | "engine" | "cuda" | "vcruntime";

// Класс системных библиотек для автозакачки (совпадает с Rust envdeps::DepKind).
export type EnvDepKind = "cuda" | "vcruntime";

export type EnvDriverStatus = {
  ok: boolean;
  version?: string | null;
};

export type EnvFoundDll = {
  name: string;
  path: string;
};

export type EnvDllGroupStatus = {
  ok: boolean;
  missing: string[];
  found: EnvFoundDll[];
  downloadMb: number;
};

// Результат команды env_check: драйвер + CUDA runtime + VC++ runtime.
export type EnvCheck = {
  driver: EnvDriverStatus;
  cuda: EnvDllGroupStatus;
  vcruntime: EnvDllGroupStatus;
};

// Результат команды download_env_deps.
export type DownloadEnvResult = {
  kind: string;
  installed: string[];
  missing: string[];
  ok: boolean;
};

export type TtsModelPreset = {
  id: string;
  label: string;
  folder: string;
  filename: string;
  size: string;
  recommended?: boolean;
};

// A Parakeet ASR model variant. `folder` is the sub-directory under
// models/parakeet where its ONNX files live; `files` is the exact set of files
// to download (the whole set — never just one, or the model fails to load).
export type WhisperModelPreset = {
  id: string;
  label: string;
  folder: string;
  files: string[];
  size: string;
  recommended?: boolean;
};

export type ModelListing = {
  name: string;
  path: string;
  format: string;
  sizeBytes: number;
  hasConfig: boolean;
};

export type GenerationResult = {
  sampleRate: number;
  channels: number;
  sampleCount: number;
  wavBase64: string;
};

export type HardwareSnapshot = {
  gpuName: string;
  totalVram: number;
  usedVram: number;
  freeVram: number;
  gpuUtilization: number;
  temperature: number;
  powerDraw: number;
  powerLimit: number;
  processRam: number;
  totalRam: number;
  usedRam: number;
  message: string;
};

export type ProgressEvent = {
  current: number;
  total: number;
  phase: string;
};

export type GenerationAudioChunkEvent = {
  sampleRate: number;
  channels: number;
  startSample: number;
  sampleCount: number;
  wavBase64: string;
  isFinal: boolean;
};

export type DownloadProgressEvent = {
  downloaded: number;
  total: number;
  speedMbps: number;
  percent: number;
  status: "running" | "paused" | "cancelled" | "complete";
};

export type ModelStatusEvent = {
  engineLoaded: boolean;
  modelLoaded: boolean;
  supportsStreaming?: boolean;
  family?: string;
  displayName?: string;
  weightType?: string;
};

export type EngineDependencyStatus = {
  name: string;
  pattern: string;
  category: string;
  required: boolean;
  foundPath?: string | null;
  fix: string;
};

export type EngineDependencyDiagnostic = {
  ok: boolean;
  platform: string;
  enginePath: string;
  detected: EngineDependencyStatus[];
  missing: EngineDependencyStatus[];
  optionalMissing: EngineDependencyStatus[];
  searchDirs: string[];
  suggestions: string[];
  rawError?: string | null;
};

export type ApiServerStatus = {
  running: boolean;
  host?: string;
  port?: number;
  baseUrl?: string;
  apiKeySet?: boolean;
  uptimeSeconds?: number;
};

export type ApiLogEvent = {
  level: "info" | "warn" | "error";
  kind: "server" | "request" | "job" | string;
  method: string;
  route: string;
  status: number;
  latencyMs: number;
  message: string;
  jobId: string;
};

export type ApiLogEntry = ApiLogEvent & {
  time: number;
};

export type StudioJobEvent = {
  id: string;
  source: string;
  workflow: string;
  status: "queued" | "generating" | "complete" | "failed" | "cancelled" | string;
  label: string;
  phase: string;
  current?: number | null;
  total?: number | null;
  message: string;
};

export type ApiExampleKind = "curl" | "python" | "javascript" | "powershell";

export type HistoryEntry = {
  id: string;
  mode: Mode;
  label: string;
  timestamp: number;
  wavBase64: string;
  sampleRate: number;
  channels: number;
};

export type MultiSpeaker = {
  id: string;
  personaId: string;
  name: string;
  refPath: string | null;
  refName: string;
  refText: string;
  notes: string;
  photoPath: string;
  cachePath: string;
  normalize: boolean;
  open: boolean;
};

export type MultiLine = {
  id: string;
  speakerId: string;
  text: string;
  overridePath: string | null;
  overrideName: string;
  overrideText: string;
  open: boolean;
};

export type GenerationMode = "tts" | "clone" | "finish" | "multi";

export type GenerationJob = {
  id: string;
  mode: GenerationMode;
  label: string;
  createdAt: number;
  options: Record<string, number | string | boolean>;
  deliveryPrefix: string;
  payload:
    | { kind: "tts"; text: string }
    | { kind: "clone"; text: string; refPath: string; refName: string; refText?: string; normalize: boolean; personaId?: string }
    | { kind: "finish"; text: string; refPath: string; refName: string; transcript: string; normalize: boolean; includeSource: boolean; personaId?: string }
    | { kind: "multi"; speakers: MultiSpeaker[]; lines: MultiLine[] };
};

export type SpeakerPersona = {
  id: string;
  name: string;
  refPath: string;
  refName: string;
  refText: string;
  notes: string;
  photoPath: string;
  cachePath: string;
  normalize: boolean;
  createdAt: number;
  updatedAt: number;
};

export type LinePointerDrag = {
  id: string;
  pointerId: number;
  grip: HTMLElement;
  active: boolean;
};

export type RefPreviewKind = "clone" | "finish" | "gallery";

export type RefPlayer = {
  audio: HTMLAudioElement;
  play: HTMLButtonElement;
  seek: HTMLInputElement;
  time: HTMLElement;
  canvas: HTMLCanvasElement;
  peaks: number[];
  waveformPath: string;
  waveformLoading: boolean;
  raf: number | null;
  previewObjectUrl: string;
  previewToken: number;
};

export type WavPcm = {
  sampleRate: number;
  channels: number;
  samples: Int16Array;
};

export type LiveStreamState = {
  context: AudioContext;
  processor: ScriptProcessorNode;
  sampleRate: number;
  channels: number;
  receivedFrames: number;
  playbackFrame: number;
  playing: boolean;
  pcm: Float32Array;
  finalized: boolean;
  finalObjectUrl: string | null;
  finalDuration: number;
};
