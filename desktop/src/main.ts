import "./styles.css";
import { getLang, setLang, t, translateStaticDom } from "./i18n";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  API_COMMAND_POPOUT_LABEL,
  API_LOG_STORAGE_KEY,
  APP_VERSION,
  CUDA_DOWNLOAD_URL,
  ENGINE_PACKAGE_URL,
  ENGINE_PATH_STORAGE_KEY,
  FIRST_RUN_WIZARD_STORAGE_KEY,
  GITHUB_URL,
  NERUAL_DREMING_URL,
  NEUROPORT_URL,
  HIGGS_MODEL_ASSET_FILES,
  HIGGS_MODEL_PRESETS,
  HIGGS_MODEL_RESOLVE_BASE,
  HIGGS_REFERENCE_MAX_SECONDS,
  HIGGS_RECOMMENDED_MODEL,
  MINIMIZE_TO_TRAY_STORAGE_KEY,
  NVIDIA_DRIVER_URL,
  MODEL_PATH_STORAGE_KEY,
  RELEASES_URL,
  SPEAKER_PERSONA_STORAGE_KEY,
  STREAM_PLAYBACK_STORAGE_KEY,
  VC_REDIST_X64_URL,
  WHISPER_MODEL_PRESETS,
  WHISPER_MODEL_RESOLVE_BASE,
  WHISPER_MODEL_TREE_URL,
  WHISPER_RECOMMENDED_MODEL,
} from "./config";
import {
  base64ToBlobAsync,
  base64ToBytesAsync,
  bytesToBase64,
  concatenateWavResults,
  encodeMp3FromWav,
  parseWavPcm,
} from "./audio";
import { buildApiExample as buildApiExampleSnippet } from "./apiExamples";
import {
  cssVar,
  el,
  escapeHtml,
  formatBytes,
  formatDuration,
  formatTime,
  nextFrame,
  setProgress,
  setText,
} from "./dom";
import type {
  ApiExampleKind,
  ApiLogEntry,
  ApiLogEvent,
  ApiServerStatus,
  DownloadKind,
  DownloadProgressEvent,
  EngineDependencyDiagnostic,
  EngineDependencyStatus,
  GenerationAudioChunkEvent,
  GenerationJob,
  GenerationMode,
  GenerationResult,
  HardwareSnapshot,
  HistoryEntry,
  LinePointerDrag,
  LiveStreamState,
  Mode,
  ModelListing,
  ModelStatusEvent,
  MultiLine,
  MultiSpeaker,
  ProgressEvent,
  RefPlayer,
  RefPreviewKind,
  SaveFormat,
  SpeakerPersona,
  StudioJobEvent,
  TtsModelPreset,
  WhisperModelPreset,
  WavPcm,
} from "./types";
async function openExternalUrl(url: string): Promise<void> {
  try {
    await invoke("open_external_url", { url });
  } catch (e) {
    showToast(`Could not open link: ${e}`, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════

let currentMode: Mode = "tts";
let isGenerating = false;
let cloneRefPath: string | null = null;
let finishRefPath: string | null = null;
let clonePersonaId = "";
let finishPersonaId = "";
let cloneRefName = "";
let finishRefName = "";
let selectedGalleryPersonaId = "";
let lastResult: GenerationResult | null = null;
const outputByMode: Partial<Record<Mode, GenerationResult>> = {};
let history: HistoryEntry[] = [];
let activeWork = 0;
let genStartedAt = 0;
let genTimer: number | null = null;
let cancelRequested = false;
let idCounter = 0;
let selectedSaveFormat: SaveFormat = (localStorage.getItem("higgsAudio.saveFormat") as SaveFormat) || "wav";
let currentProgressLabels: string[] = [];
let draggedLineId: string | null = null;
let linePointerDrag: LinePointerDrag | null = null;
let activeGenerationJob: GenerationJob | null = null;
const generationQueue: GenerationJob[] = [];
const externalStudioJobs = new Map<string, StudioJobEvent>();
let generationProgressPrefix = "Elapsed";
let activeDownloadKind: DownloadKind = "model";
let downloadActive = false;
let downloadPaused = false;
let activeDownloadFileLabel = "";
const apiLogs: ApiLogEntry[] = [];
let engineDiagnosticVisible = false;
let setupWizardOpen = false;
type DropzoneHandler = {
  selector: string;
  onFile: (path: string, name: string) => void | Promise<void>;
};
const dropzoneHandlers: DropzoneHandler[] = [];
type PreparedReferenceUpload = {
  path: string;
  fileName: string;
  durationSeconds: number;
  cropped: boolean;
};
let apiRunning = false;
let selectedApiExample: ApiExampleKind = "curl";
let liveStream: LiveStreamState | null = null;

// Settings state
let currentTheme = localStorage.getItem("higgsAudio.theme") || "dark";
let currentAccent = localStorage.getItem("higgsAudio.accent") || "teal";
let currentUiScale = parseInt(localStorage.getItem("higgsAudio.uiScale") ?? "100", 10);
let minimizeToTray = localStorage.getItem(MINIMIZE_TO_TRAY_STORAGE_KEY) === "true";
let streamPlayback = localStorage.getItem(STREAM_PLAYBACK_STORAGE_KEY) === "true";
let engineSupportsStreaming = false;

// Hardware state
let hardwarePollMs = parseInt(localStorage.getItem("higgsAudio.hardwarePollMs") ?? "1000", 10);
const hardwareHistory: HardwareSnapshot[] = [];
const hardwareHistoryLimit = 1200; // ~20 min at 1s/tick
const hardwareGraphPoints = 120;   // visible window width
let hardwareViewOffset = 0;        // 0 = live (right edge), positive = looking back
let hardwareFollowLive = true;     // when true, graph auto-scrolls to newest
let hardwareScrubDrag: { startX: number; startOffset: number } | null = null;
let hardwareHover: { x: number; idx: number } | null = null;

const audioPlayer = el<HTMLAudioElement>("#audio-player");

// Output playback volume (0..1). Applied to the final <audio> element and, for
// the live-stream preview, mixed into the PCM buffer in processLiveStreamAudio.
const OUTPUT_VOLUME_STORAGE_KEY = "higgsAudio.outputVolume";
let outputVolume = (() => {
  const saved = Number(localStorage.getItem(OUTPUT_VOLUME_STORAGE_KEY));
  return Number.isFinite(saved) && saved >= 0 && saved <= 1 ? saved : 1;
})();

let speakerPersonas: SpeakerPersona[] = [];
const multiSpeakers: MultiSpeaker[] = [];
const multiLines: MultiLine[] = [];
const refPlayers: Partial<Record<RefPreviewKind, RefPlayer>> = {};
const speakerSyncTimers = new Map<string, number>();

function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}_${Date.now()}_${idCounter}`;
}

function isGenerationMode(mode: Mode): mode is GenerationMode {
  return mode === "tts" || mode === "clone" || mode === "finish" || mode === "multi";
}

function cloneMultiSpeaker(speaker: MultiSpeaker): MultiSpeaker {
  return { ...speaker, cachePath: speaker.cachePath || "" };
}

function cloneMultiLine(line: MultiLine): MultiLine {
  return { ...line };
}

// ═══════════════════════════════════════════════════════════════════════════
// Toast
// ═══════════════════════════════════════════════════════════════════════════

let toastTimer: number | null = null;

function showToast(message: string, tone: "success" | "warning" | "error" = "success"): void {
  const toast = el<HTMLDivElement>("#toast");
  toast.textContent = t(message);
  toast.className = "toast";
  if (tone !== "success") toast.classList.add(tone);
  toast.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("show");
    toastTimer = null;
  }, 3200);
}

let lastEngineDiagnostic: EngineDependencyDiagnostic | null = null;

function dependencyLabel(dep: EngineDependencyStatus): string {
  return `${dep.pattern} · ${dep.category}`;
}

function renderDependencyList(container: HTMLElement, deps: EngineDependencyStatus[], statusClass: "missing" | "detected"): void {
  container.innerHTML = "";
  if (!deps.length) {
    const empty = document.createElement("div");
    empty.className = `engine-diagnostic-item ${statusClass}`;
    empty.innerHTML = `<strong>${statusClass === "missing" ? t("None") : t("Nothing detected yet")}</strong>`;
    container.appendChild(empty);
    return;
  }

  for (const dep of deps) {
    const item = document.createElement("div");
    item.className = `engine-diagnostic-item ${statusClass}`;
    item.innerHTML = `
      <strong>${escapeHtml(dependencyLabel(dep))}</strong>
      ${dep.foundPath ? `<code>${escapeHtml(dep.foundPath)}</code>` : `<span>${escapeHtml(dep.fix)}</span>`}
    `;
    container.appendChild(item);
  }
}

function engineDiagnosticText(diagnostic: EngineDependencyDiagnostic): string {
  const lines = [
    "Higgs Audio v3 Studio Engine Diagnostics",
    "",
    `App version: ${APP_VERSION}`,
    `Platform: ${diagnostic.platform}`,
    `Engine path: ${diagnostic.enginePath || "(none)"}`,
  ];

  if (diagnostic.rawError) {
    lines.push("", "Raw loader error:", diagnostic.rawError);
  }

  lines.push("", "Missing required DLLs:");
  if (diagnostic.missing.length) {
    for (const dep of diagnostic.missing) {
      lines.push(`- ${dep.pattern} (${dep.category})`);
      lines.push(`  Fix: ${dep.fix}`);
    }
  } else {
    lines.push("- none");
  }

  lines.push("", "Detected DLLs:");
  if (diagnostic.detected.length) {
    for (const dep of diagnostic.detected) {
      lines.push(`- ${dep.pattern}: ${dep.foundPath || "found"}`);
    }
  } else {
    lines.push("- none");
  }

  if (diagnostic.optionalMissing.length) {
    lines.push("", "Optional/informational checks not found:");
    for (const dep of diagnostic.optionalMissing) {
      lines.push(`- ${dep.pattern} (${dep.category})`);
    }
  }

  lines.push("", "Search paths checked:");
  for (const dir of diagnostic.searchDirs) lines.push(`- ${dir}`);

  lines.push("", "Repair links:");
  lines.push(`- Higgs engine DLL package: ${ENGINE_PACKAGE_URL}`);
  lines.push(`- NVIDIA driver: ${NVIDIA_DRIVER_URL}`);
  lines.push(`- CUDA Toolkit 13.x: ${CUDA_DOWNLOAD_URL}`);
  lines.push(`- VC++ Redistributable x64: ${VC_REDIST_X64_URL}`);

  return lines.join("\n");
}

function showEngineDiagnosticModal(diagnostic: EngineDependencyDiagnostic): void {
  lastEngineDiagnostic = diagnostic;
  hideSetupWizard();
  engineDiagnosticVisible = true;
  const modal = el<HTMLDivElement>("#engine-diagnostic-modal");
  const missingCount = diagnostic.missing.length;
  setText(
    "#engine-diagnostic-summary",
    missingCount
      ? `The engine DLL was found, but Windows cannot load ${missingCount} required dependency${missingCount === 1 ? "" : "ies"}. Click Download Engine DLLs or install the missing runtime(s), restart the app, then load the engine again.`
      : diagnostic.rawError
        ? "The known dependency check passed, but Windows still rejected the engine DLL. Copy diagnostics and include them in the GitHub issue."
        : "The engine dependency check passed.",
  );
  setText("#engine-diagnostic-subtitle", diagnostic.enginePath || "No engine path");
  renderDependencyList(el("#engine-diagnostic-missing"), diagnostic.missing, "missing");
  renderDependencyList(el("#engine-diagnostic-detected"), diagnostic.detected, "detected");
  el<HTMLPreElement>("#engine-diagnostic-paths").textContent = diagnostic.searchDirs.join("\n");
  modal.classList.remove("hidden");
}

async function diagnoseEngineForPath(libraryPath?: string): Promise<EngineDependencyDiagnostic | null> {
  try {
    return await invoke<EngineDependencyDiagnostic>("diagnose_engine_load", { libraryPath });
  } catch {
    return null;
  }
}

function initEngineDiagnosticModal(): void {
  const modal = el<HTMLDivElement>("#engine-diagnostic-modal");
  const close = () => {
    modal.classList.add("hidden");
    engineDiagnosticVisible = false;
  };
  el("#engine-diagnostic-close").addEventListener("click", close);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  el("#engine-diagnostic-copy").addEventListener("click", () => {
    if (!lastEngineDiagnostic) return;
    void copyText(engineDiagnosticText(lastEngineDiagnostic), "Engine diagnostics");
  });
  el("#engine-diagnostic-driver").addEventListener("click", () => openExternalUrl(NVIDIA_DRIVER_URL));
  el("#engine-diagnostic-cuda").addEventListener("click", () => openExternalUrl(CUDA_DOWNLOAD_URL));
  el("#engine-diagnostic-vc").addEventListener("click", () => openExternalUrl(VC_REDIST_X64_URL));
}

function hideSetupWizard(persist = false): void {
  const modal = document.querySelector<HTMLDivElement>("#setup-wizard-modal");
  if (!modal) return;
  modal.classList.add("hidden");
  setupWizardOpen = false;
  if (persist) localStorage.setItem(FIRST_RUN_WIZARD_STORAGE_KEY, "true");
}

function setSetupCheck(
  rowId: string,
  iconId: string,
  detailId: string,
  state: "ready" | "missing" | "optional",
  icon: string,
  detail: string,
): void {
  const row = el<HTMLElement>(rowId);
  row.classList.toggle("ready", state === "ready");
  row.classList.toggle("missing", state === "missing");
  row.classList.toggle("optional", state === "optional");
  setText(iconId, icon);
  setText(detailId, detail);
}

async function refreshSetupWizardState(): Promise<void> {
  let engineStatus: ModelStatusEvent | null = null;
  try {
    engineStatus = await invoke<ModelStatusEvent>("engine_status");
  } catch {
    engineStatus = null;
  }

  const savedEnginePath = localStorage.getItem(ENGINE_PATH_STORAGE_KEY) || "";
  const bundledEnginePath = await invoke<string | null>("bundled_engine_path").catch(() => null);
  let enginePath = savedEnginePath || bundledEnginePath || "";
  let engineReady = Boolean(engineStatus?.engineLoaded);
  let enginePathValid = Boolean(engineStatus?.engineLoaded);
  let engineDetail = engineStatus?.engineLoaded
    ? t("Loaded and ready")
    : enginePath
      ? `${t("Found:")} ${enginePath}`
      : t("Not found. Download the engine package or browse to audiocpp_engine.dll.");
  if (!engineReady && enginePath) {
    let diagnostic = await diagnoseEngineForPath(enginePath);
    if (!diagnostic && savedEnginePath && bundledEnginePath) {
      localStorage.removeItem(ENGINE_PATH_STORAGE_KEY);
      enginePath = bundledEnginePath;
      diagnostic = await diagnoseEngineForPath(enginePath);
    }
    enginePathValid = Boolean(diagnostic);
    engineReady = Boolean(diagnostic?.ok);
    if (!diagnostic && savedEnginePath) {
      engineDetail = `${t("Saved engine path not found:")} ${savedEnginePath}`;
    } else if (diagnostic?.ok) {
      engineDetail = `${t("Found:")} ${enginePath}`;
    }
    if (diagnostic && !diagnostic.ok) {
      engineDetail = getLang() === "ru"
        ? `Движок найден, но не хватает ${diagnostic.missing.length} системных DLL.`
        : `Found engine, but missing ${diagnostic.missing.length} required runtime DLL${diagnostic.missing.length === 1 ? "" : "s"}.`;
    }
  }
  setSetupCheck(
    "#setup-check-engine",
    "#setup-engine-icon",
    "#setup-engine-detail",
    engineReady ? "ready" : "missing",
    engineReady ? "✓" : "!",
    engineDetail,
  );
  el<HTMLButtonElement>("#setup-wizard-load-engine").disabled = !enginePathValid || Boolean(engineStatus?.engineLoaded);

  let models: ModelListing[] = [];
  try {
    models = await invoke<ModelListing[]>("list_models");
  } catch {
    models = [];
  }
  const modelSelect = el<HTMLSelectElement>("#model-select");
  let selectedModel = modelSelect.value || localStorage.getItem(MODEL_PATH_STORAGE_KEY) || "";
  if (!selectedModel && models.length > 0) {
    const recommended = [...models].sort((a, b) => modelSortRank(a) - modelSortRank(b))[0];
    selectedModel = recommended.path;
    localStorage.setItem(MODEL_PATH_STORAGE_KEY, selectedModel);
    addLocalModelOption(selectedModel, modelDisplayName(recommended));
  }
  const modelReady = Boolean(engineStatus?.modelLoaded || selectedModel || models.length > 0);
  const modelDetail = engineStatus?.modelLoaded
    ? t("Loaded and ready")
    : selectedModel
      ? `${t("Selected:")} ${selectedModel}`
      : models.length > 0
        ? (getLang() === "ru"
            ? `Найдено папок моделей: ${models.length}. Выберите или загрузите.`
            : `${models.length} model folder${models.length === 1 ? "" : "s"} found. Select or load one.`)
        : t("No Higgs model folder found. Download a model folder or browse to one.");
  setSetupCheck(
    "#setup-check-model",
    "#setup-model-icon",
    "#setup-model-detail",
    modelReady ? "ready" : "missing",
    modelReady ? "✓" : "!",
    modelDetail,
  );
  el<HTMLButtonElement>("#setup-wizard-load-model").disabled = !selectedModel || Boolean(engineStatus?.modelLoaded);

  const whisperPath = localStorage.getItem("higgsAudio.whisperModel") || "";
  setSetupCheck(
    "#setup-check-whisper",
    "#setup-whisper-icon",
    "#setup-whisper-detail",
    whisperPath ? "ready" : "optional",
    whisperPath ? "✓" : "i",
    whisperPath ? `${t("Selected:")} ${whisperPath}` : t("Optional. Download only if you want auto-transcription."),
  );

  const readyCount = Number(engineReady) + Number(modelReady);
  setText(
    "#setup-wizard-summary",
    readyCount === 2
      ? t("Core setup found. You can load anything not already loaded, or close this wizard.")
      : t("Some core setup files are missing. Use Browse if you already have them, or Download to fetch them."),
  );
}

async function maybeShowSetupWizard(): Promise<void> {
  const modal = document.querySelector<HTMLDivElement>("#setup-wizard-modal");
  const diagnostic = document.querySelector<HTMLDivElement>("#engine-diagnostic-modal");
  if (!modal) return;
  if (localStorage.getItem(FIRST_RUN_WIZARD_STORAGE_KEY) === "true") return;
  if (setupWizardOpen) return;
  if (engineDiagnosticVisible || (diagnostic && !diagnostic.classList.contains("hidden"))) return;
  await refreshSetupWizardState();
  modal.classList.remove("hidden");
  setupWizardOpen = true;
}

function initSetupWizard(): void {
  const modal = el<HTMLDivElement>("#setup-wizard-modal");
  const close = () => hideSetupWizard(true);
  el("#setup-wizard-close").addEventListener("click", close);
  el("#setup-wizard-skip").addEventListener("click", close);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  el("#setup-wizard-refresh").addEventListener("click", () => {
    void refreshSetupWizardState();
  });
  el("#setup-wizard-browse-engine").addEventListener("click", () => {
    void doBrowseEngine().then(() => refreshSetupWizardState());
  });
  el("#setup-wizard-download-engine").addEventListener("click", () => {
    hideSetupWizard(false);
    el<HTMLButtonElement>("#download-engine-btn").click();
  });
  el("#setup-wizard-load-engine").addEventListener("click", () => {
    void doLoadEngine().then(() => refreshSetupWizardState());
  });
  el("#setup-wizard-browse-model").addEventListener("click", () => {
    void doBrowseModel().then(() => refreshSetupWizardState());
  });
  el("#setup-wizard-download-model").addEventListener("click", () => {
    hideSetupWizard(false);
    el<HTMLButtonElement>("#download-trigger").click();
  });
  el("#setup-wizard-load-model").addEventListener("click", () => {
    void doLoadModel().then(() => refreshSetupWizardState());
  });
  el("#setup-wizard-download-whisper").addEventListener("click", () => {
    hideSetupWizard(false);
    el<HTMLButtonElement>("#whisper-download-trigger").click();
  });
  el("#setup-wizard-continue").addEventListener("click", close);
}

// ═══════════════════════════════════════════════════════════════════════════
// Settings system
// ═══════════════════════════════════════════════════════════════════════════

function applyTheme(theme: string): void {
  currentTheme = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("higgsAudio.theme", currentTheme);
  el<HTMLButtonElement>("#theme-dark").classList.toggle("active", currentTheme === "dark");
  el<HTMLButtonElement>("#theme-light").classList.toggle("active", currentTheme === "light");
  requestAnimationFrame(() => {
    drawHardwareGraph();
    drawWaveform();
    redrawAllRefPreviews();
  });
}

function applyAccent(accent: string): void {
  const allowed = ["teal", "blue", "green", "red", "yellow"];
  currentAccent = allowed.includes(accent) ? accent : "teal";
  document.documentElement.setAttribute("data-accent", currentAccent);
  localStorage.setItem("higgsAudio.accent", currentAccent);
  for (const btn of document.querySelectorAll<HTMLButtonElement>("[data-accent-choice]")) {
    btn.classList.toggle("active", btn.dataset.accentChoice === currentAccent);
  }
  requestAnimationFrame(() => {
    drawHardwareGraph();
    drawWaveform();
    redrawAllRefPreviews();
  });
}

function applyUiScale(percent: number): void {
  currentUiScale = Math.min(115, Math.max(90, percent || 100));
  document.documentElement.style.setProperty("--ui-scale", (currentUiScale / 100).toFixed(2));
  localStorage.setItem("higgsAudio.uiScale", String(currentUiScale));
  el<HTMLInputElement>("#ui-scale").value = String(currentUiScale);
  setText("#ui-scale-label", `${currentUiScale}%`);
  requestAnimationFrame(() => {
    drawHardwareGraph();
    drawWaveform();
  });
}

function applySavedVisualShell(): void {
  const theme = localStorage.getItem("higgsAudio.theme") === "light" ? "light" : "dark";
  const accent = localStorage.getItem("higgsAudio.accent") || "teal";
  const uiScale = parseInt(localStorage.getItem("higgsAudio.uiScale") ?? "100", 10);
  document.documentElement.setAttribute("data-theme", theme);
  document.documentElement.setAttribute("data-accent", ["teal", "blue", "green", "red", "yellow"].includes(accent) ? accent : "teal");
  document.documentElement.style.setProperty("--ui-scale", (Math.min(115, Math.max(90, uiScale || 100)) / 100).toFixed(2));
}

let activeTip: HTMLElement | null = null;

function positionTooltip(): void {
  if (!activeTip) return;
  const layer = el<HTMLDivElement>("#tooltip-layer");
  const targetRect = activeTip.getBoundingClientRect();
  const layerRect = layer.getBoundingClientRect();
  const margin = 12;
  const gap = 10;
  const left = Math.min(
    window.innerWidth - layerRect.width - margin,
    Math.max(margin, targetRect.left + targetRect.width / 2 - layerRect.width / 2),
  );
  let top = targetRect.bottom + gap;
  if (top + layerRect.height > window.innerHeight - margin) {
    top = targetRect.top - layerRect.height - gap;
  }
  if (top < margin) top = margin;
  layer.style.left = `${left}px`;
  layer.style.top = `${top}px`;
}

function showTooltip(trigger: HTMLElement): void {
  const text = trigger.dataset.tip?.trim();
  if (!text) return;
  activeTip = trigger;
  const layer = el<HTMLDivElement>("#tooltip-layer");
  layer.textContent = text;
  layer.classList.remove("hidden", "visible");
  requestAnimationFrame(() => {
    positionTooltip();
    layer.classList.add("visible");
  });
}

function hideTooltip(trigger?: HTMLElement): void {
  if (trigger && activeTip !== trigger) return;
  activeTip = null;
  const layer = el<HTMLDivElement>("#tooltip-layer");
  layer.classList.remove("visible");
  layer.classList.add("hidden");
}

function initTooltips(): void {
  document.addEventListener("pointerover", (event) => {
    const trigger = (event.target as Element | null)?.closest<HTMLElement>(".tip[data-tip]");
    if (trigger) showTooltip(trigger);
  });
  document.addEventListener("pointerout", (event) => {
    const trigger = (event.target as Element | null)?.closest<HTMLElement>(".tip[data-tip]");
    if (trigger && !trigger.contains(event.relatedTarget as Node | null)) hideTooltip(trigger);
  });
  document.addEventListener("focusin", (event) => {
    const trigger = (event.target as Element | null)?.closest<HTMLElement>(".tip[data-tip]");
    if (trigger) showTooltip(trigger);
  });
  document.addEventListener("focusout", (event) => {
    const trigger = (event.target as Element | null)?.closest<HTMLElement>(".tip[data-tip]");
    if (trigger) hideTooltip(trigger);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
  document.addEventListener("scroll", () => {
    if (activeTip) positionTooltip();
  }, true);
  window.addEventListener("resize", () => {
    if (activeTip) positionTooltip();
  });
}

function initSettings(): void {
  const button = el<HTMLButtonElement>("#settings-button");
  const popover = el<HTMLDivElement>("#settings-popover");
  const setOpen = (open: boolean) => {
    popover.hidden = !open;
    button.classList.toggle("active", open);
  };
  button.addEventListener("click", (e) => { e.stopPropagation(); setOpen(popover.hidden); });
  const settingsClose = document.querySelector<HTMLElement>("#settings-close");
  if (settingsClose) settingsClose.addEventListener("click", (e) => { e.stopPropagation(); setOpen(false); });
  document.addEventListener("pointerdown", (event) => {
    const target = event.target as Node;
    if (!popover.hidden && !popover.contains(target) && !button.contains(target)) {
      setOpen(false);
    }
  });

  const langRu = el<HTMLButtonElement>("#lang-ru");
  const langEn = el<HTMLButtonElement>("#lang-en");
  langRu.classList.toggle("active", getLang() === "ru");
  langEn.classList.toggle("active", getLang() === "en");
  langRu.addEventListener("click", () => { if (getLang() !== "ru") setLang("ru"); });
  langEn.addEventListener("click", () => { if (getLang() !== "en") setLang("en"); });

  el("#theme-dark").addEventListener("click", () => applyTheme("dark"));
  el("#theme-light").addEventListener("click", () => applyTheme("light"));
  for (const accentBtn of document.querySelectorAll<HTMLButtonElement>("[data-accent-choice]")) {
    accentBtn.addEventListener("click", () => applyAccent(accentBtn.dataset.accentChoice || "teal"));
  }
  el<HTMLInputElement>("#ui-scale").addEventListener("input", (event) => {
    applyUiScale(parseInt((event.target as HTMLInputElement).value, 10));
  });
  const streamToggle = el<HTMLInputElement>("#stream-playback");
  streamToggle.checked = streamPlayback;
  streamToggle.addEventListener("change", () => {
    streamPlayback = streamToggle.checked;
    localStorage.setItem(STREAM_PLAYBACK_STORAGE_KEY, String(streamPlayback));
    if (streamPlayback && !engineSupportsStreaming) {
      showToast("Streaming needs an updated Higgs engine DLL; current DLL falls back to normal generation.", "warning");
    }
  });
  const trayToggle = el<HTMLInputElement>("#minimize-to-tray");
  trayToggle.checked = minimizeToTray;
  trayToggle.addEventListener("change", () => {
    minimizeToTray = trayToggle.checked;
    localStorage.setItem(MINIMIZE_TO_TRAY_STORAGE_KEY, String(minimizeToTray));
    void invoke("set_minimize_to_tray", { enabled: minimizeToTray }).catch((error) => {
      showToast(`Tray setting failed: ${String(error)}`, "error");
    });
  });
  el<HTMLButtonElement>("#quit-app-btn").addEventListener("click", () => {
    clearApiLogsForFreshSession();
    void invoke("quit_app").catch((error) => {
      showToast(`Quit failed: ${String(error)}`, "error");
    });
  });

  applyAccent(currentAccent);
  applyTheme(currentTheme);
  applyUiScale(currentUiScale);
  void invoke("set_minimize_to_tray", { enabled: minimizeToTray }).catch(() => {});
}

function initExternalLinks(): void {
  setText("#version-link", `v${APP_VERSION}`);
  el("#github-link").addEventListener("click", () => openExternalUrl(GITHUB_URL));
  el("#version-link").addEventListener("click", () => openExternalUrl(RELEASES_URL));
  const nerual = document.querySelector<HTMLButtonElement>("#nerual-link");
  if (nerual) nerual.addEventListener("click", () => openExternalUrl(NERUAL_DREMING_URL));
  const neuroport = document.querySelector<HTMLButtonElement>("#neuroport-link");
  if (neuroport) neuroport.addEventListener("click", () => openExternalUrl(NEUROPORT_URL));
}

function ttsPresetUrl(preset: TtsModelPreset): string {
  return `${HIGGS_MODEL_RESOLVE_BASE}/models/${preset.folder}/${preset.filename}`;
}

function ttsPresetPackageFiles(preset: TtsModelPreset): string[] {
  return [preset.filename, ...HIGGS_MODEL_ASSET_FILES];
}

function ttsPresetPackageEntries(preset: TtsModelPreset): Array<{ url: string; destDir: string; filename: string }> {
  return ttsPresetPackageFiles(preset).map((filename) => ({
    url: `${HIGGS_MODEL_RESOLVE_BASE}/models/${preset.folder}/${filename}`,
    destDir: `models/${preset.folder}`,
    filename,
  }));
}

function ttsPresetFromModelDownloadUrl(url: string): TtsModelPreset | null {
  try {
    const parsed = new URL(url);
    const match = parsed.pathname.match(/\/models\/([^/]+)\/([^/]+)$/);
    if (!match) return null;
    const folder = decodeURIComponent(match[1]);
    const filename = decodeURIComponent(match[2]);
    return HIGGS_MODEL_PRESETS.find((preset) =>
      preset.folder === folder && ttsPresetPackageFiles(preset).includes(filename)) || null;
  } catch {
    return null;
  }
}

function ttsPresetById(id: string | null | undefined): TtsModelPreset {
  return HIGGS_MODEL_PRESETS.find((preset) => preset.id === id) ||
    HIGGS_MODEL_PRESETS.find((preset) => preset.id === HIGGS_RECOMMENDED_MODEL) ||
    HIGGS_MODEL_PRESETS[0];
}

function inferTtsPresetFromModelSelection(): TtsModelPreset {
  const select = el<HTMLSelectElement>("#model-select");
  const selectedText = select.selectedOptions[0]?.textContent || "";
  const value = select.value || "";
  const haystack = `${selectedText} ${value}`.toLowerCase();
  if (haystack.includes("q6_k") || haystack.includes("q6-k")) return ttsPresetById("higgs-q6_k");
  if (haystack.includes("q5_k") || haystack.includes("q5-k")) return ttsPresetById("higgs-q5_k");
  if (haystack.includes("q4_k") || haystack.includes("q4-k")) return ttsPresetById("higgs-q4_k_m");
  if (haystack.includes("bf16")) return ttsPresetById("higgs-bf16");
  if (haystack.includes("q8_0") || haystack.includes("q8")) return ttsPresetById("higgs-q8_0");
  return ttsPresetById(localStorage.getItem("higgsAudio.ttsDownloadPreset") || HIGGS_RECOMMENDED_MODEL);
}

function populateTtsDownloadPresetSelect(): void {
  const select = el<HTMLSelectElement>("#download-model-preset");
  select.innerHTML = "";
  for (const preset of HIGGS_MODEL_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.recommended ? "★ " : ""}${preset.label} · ${preset.size}`;
    option.title = ttsPresetUrl(preset);
    select.appendChild(option);
  }
}

function setTtsDownloadPreset(preset: TtsModelPreset, syncUrl = true): void {
  const select = el<HTMLSelectElement>("#download-model-preset");
  select.value = preset.id;
  localStorage.setItem("higgsAudio.ttsDownloadPreset", preset.id);
  if (syncUrl) {
    const input = el<HTMLInputElement>("#download-url-input");
    input.value = ttsPresetUrl(preset);
    input.title = `${preset.label} | ${preset.filename}`;
  }
}

function setWhisperModelPath(path: string): void {
  el<HTMLInputElement>("#whisper-model-path").value = path;
  localStorage.setItem("higgsAudio.whisperModel", path);
}

function whisperPresetFilename(preset: WhisperModelPreset): string {
  return `ggml-${preset.id}.bin`;
}

function whisperPresetUrl(preset: WhisperModelPreset): string {
  return `${WHISPER_MODEL_RESOLVE_BASE}/${whisperPresetFilename(preset)}`;
}

function selectedWhisperPreset(): WhisperModelPreset {
  const selectedId = el<HTMLSelectElement>("#whisper-model-select").value || WHISPER_RECOMMENDED_MODEL;
  return WHISPER_MODEL_PRESETS.find((preset) => preset.id === selectedId) || WHISPER_MODEL_PRESETS[0];
}

function populateWhisperModelSelect(): void {
  const select = el<HTMLSelectElement>("#whisper-model-select");
  const savedPreset = localStorage.getItem("higgsAudio.whisperPreset") || WHISPER_RECOMMENDED_MODEL;
  select.innerHTML = "";
  for (const preset of WHISPER_MODEL_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.recommended ? "★ " : ""}${preset.id} · ${preset.size}${preset.recommended ? " · recommended" : ""}`;
    option.title = `${whisperPresetFilename(preset)} | SHA1 ${preset.sha}`;
    select.appendChild(option);
  }
  select.value = WHISPER_MODEL_PRESETS.some((preset) => preset.id === savedPreset) ? savedPreset : WHISPER_RECOMMENDED_MODEL;
}

function initWhisperPanel(): void {
  const whisperInput = el<HTMLInputElement>("#whisper-model-path");
  const whisperSelect = el<HTMLSelectElement>("#whisper-model-select");
  populateWhisperModelSelect();
  whisperInput.value = localStorage.getItem("higgsAudio.whisperModel") || "";
  whisperInput.addEventListener("change", () => setWhisperModelPath(whisperInput.value.trim()));
  whisperSelect.addEventListener("change", () => {
    localStorage.setItem("higgsAudio.whisperPreset", whisperSelect.value);
  });
  el("#whisper-browse-btn").addEventListener("click", async () => {
    const selected = await open({
      filters: [{ name: "Whisper Model", extensions: ["bin"] }],
    });
    if (selected) setWhisperModelPath(selected);
  });
  el("#whisper-models-link").addEventListener("click", () => openExternalUrl(WHISPER_MODEL_TREE_URL));
}

// ═══════════════════════════════════════════════════════════════════════════
// Local API page
// ═══════════════════════════════════════════════════════════════════════════

function generateApiKey(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function copyText(text: string, label: string): Promise<void> {
  await navigator.clipboard.writeText(text);
  showToast(`${label} copied`);
}

function getApiBaseUrl(): string {
  return el<HTMLInputElement>("#api-base-url").value.trim() || "http://127.0.0.1:7077/v1";
}

function getApiRootUrl(): string {
  return getApiBaseUrl().replace(/\/v1\/?$/, "");
}

function getApiKeyForExample(): string {
  return el<HTMLInputElement>("#api-key").value.trim() || "YOUR_API_KEY";
}

function apiSpeakerPayload(): Array<{ id: string; name: string; refPath: string; refText: string; cachePath: string; normalize: boolean }> {
  return speakerPersonas.map((persona) => ({
    id: persona.id,
    name: persona.name,
    refPath: persona.refPath,
    refText: persona.refText,
    cachePath: persona.cachePath,
    normalize: persona.normalize,
  }));
}

function buildCurrentApiExample(kind: ApiExampleKind): string {
  const firstSpeaker = speakerPersonas[0];
  const speakerVoice = firstSpeaker ? `speaker:${firstSpeaker.id}` : "speaker:YOUR_SPEAKER_ID";
  return buildApiExampleSnippet(kind, {
    base: getApiBaseUrl(),
    root: getApiRootUrl(),
    key: getApiKeyForExample(),
    speakerVoice,
  });
}

function renderApiExample(): void {
  const code = el<HTMLElement>("#api-example-code");
  code.textContent = buildCurrentApiExample(selectedApiExample);
  for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-api-example]")) {
    const active = tab.dataset.apiExample === selectedApiExample;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  }
}

function updateApiBaseUrl(): void {
  const host = el<HTMLInputElement>("#api-host").value.trim() || "127.0.0.1";
  const port = parseInt(el<HTMLInputElement>("#api-port").value, 10) || 7077;
  el<HTMLInputElement>("#api-base-url").value = `http://${host}:${port}/v1`;
  renderApiExample();
}

function setApiVisualStatus(status: "stopped" | "starting" | "running" | "error", text?: string): void {
  const button = el<HTMLButtonElement>("#api-status-button");
  button.classList.remove("stopped", "starting", "running", "error");
  button.classList.add(status);
  button.classList.toggle("hidden", status === "stopped" && currentMode !== "api");
  button.title = text || `API server ${status}`;
  setText("#api-status-text", text || status[0].toUpperCase() + status.slice(1));
  el<HTMLButtonElement>("#api-start-btn").disabled = status === "running" || status === "starting";
  el<HTMLButtonElement>("#api-stop-btn").disabled = status !== "running";
}

function appendApiLog(event: Partial<ApiLogEvent> & { message: string }): void {
  apiLogs.push({
    time: Date.now(),
    level: event.level || "info",
    kind: event.kind || "server",
    method: event.method || "",
    route: event.route || "",
    status: event.status || 0,
    latencyMs: event.latencyMs || 0,
    message: event.message,
    jobId: event.jobId || "",
  });
  while (apiLogs.length > 300) apiLogs.shift();
  persistApiLogs();
  renderApiLogs();
}

function loadApiLogsFromStorage(): void {
  try {
    const parsed = JSON.parse(localStorage.getItem(API_LOG_STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return;
    apiLogs.splice(0, apiLogs.length, ...parsed.map((entry) => ({
      time: Number(entry.time || Date.now()),
      level: entry.level === "warn" || entry.level === "error" ? entry.level : "info",
      kind: String(entry.kind || "server"),
      method: String(entry.method || ""),
      route: String(entry.route || ""),
      status: Number(entry.status || 0),
      latencyMs: Number(entry.latencyMs || entry.latency_ms || 0),
      message: String(entry.message || ""),
      jobId: String(entry.jobId || entry.job_id || ""),
    })).filter((entry) => entry.message).slice(-300));
  } catch {
    apiLogs.splice(0, apiLogs.length);
  }
}

function persistApiLogs(): void {
  try {
    localStorage.setItem(API_LOG_STORAGE_KEY, JSON.stringify(apiLogs.slice(-300)));
  } catch {
    // Best-effort cache for the detachable console.
  }
}

function clearApiLogsForFreshSession(): void {
  apiLogs.splice(0, apiLogs.length);
  try {
    localStorage.removeItem(API_LOG_STORAGE_KEY);
  } catch {
    // Best-effort cleanup only.
  }
}

function apiLogVisible(entry: ApiLogEntry, filter: string): boolean {
  if (filter === "all") return true;
  return entry.level === filter || entry.kind === filter;
}

function renderApiLogs(): void {
  const list = document.querySelector<HTMLElement>("#api-command-log");
  if (!list) return;
  const filter = document.querySelector<HTMLSelectElement>("#api-log-filter")?.value || "all";
  const visible = apiLogs.filter((entry) => apiLogVisible(entry, filter));
  setText("#api-log-count", `${apiLogs.length} event${apiLogs.length === 1 ? "" : "s"}`);
  list.innerHTML = "";
  if (visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "api-log-empty";
    empty.textContent = "No API events";
    list.appendChild(empty);
    return;
  }
  for (const entry of visible.slice(-120)) {
    const row = document.createElement("div");
    row.className = `api-log-row ${entry.level}`;
    const time = new Date(entry.time).toLocaleTimeString([], { hour12: false });
    row.innerHTML = `
      <span class="api-log-time">${time}</span>
      <span class="api-log-method">${escapeHtml(entry.method || entry.kind.toUpperCase())}</span>
      <span class="api-log-status">${entry.status || ""}</span>
      <span class="api-log-route">${escapeHtml(entry.route || "/api")}</span>
      <span>${entry.latencyMs ? `${entry.latencyMs}ms` : ""}</span>
      <span class="api-log-message">${escapeHtml(entry.message)}</span>
    `;
    list.appendChild(row);
  }
  list.scrollTop = list.scrollHeight;
}

function syncApiLogsFromStorage(): void {
  loadApiLogsFromStorage();
  renderApiLogs();
}

async function openApiCommandPopout(): Promise<void> {
  const existing = await WebviewWindow.getByLabel(API_COMMAND_POPOUT_LABEL);
  if (existing) {
    await existing.show();
    await existing.unminimize();
    await existing.setFocus();
    return;
  }
  const url = "index.html?popout=api-command-centre";
  const popout = new WebviewWindow(API_COMMAND_POPOUT_LABEL, {
    url,
    title: "Higgs API Command Centre",
    width: 900,
    height: 640,
    minWidth: 620,
    minHeight: 420,
    resizable: true,
    decorations: true,
  });
  popout.once("tauri://error", (event) => {
    showToast(`Command Centre popout failed: ${String(event.payload)}`, "error");
  });
}

function renderApiCommandPopoutShell(): void {
  document.body.className = "api-popout-body";
  document.body.innerHTML = `
    <main class="api-popout-shell">
      <header class="api-popout-head">
        <div>
          <h1>API Command Centre</h1>
          <span id="api-log-count" class="api-status-text">0 events</span>
        </div>
        <div class="api-actions">
          <select id="api-log-filter" class="select-input compact-select">
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warn">Warnings</option>
            <option value="error">Errors</option>
            <option value="request">Requests</option>
            <option value="job">Jobs</option>
          </select>
          <button id="api-copy-log-btn" class="compact-button" type="button">Copy</button>
          <button id="api-clear-log-btn" class="compact-button" type="button">Clear</button>
        </div>
      </header>
      <div id="api-command-log" class="api-command-log api-popout-log" aria-live="polite"></div>
    </main>
  `;
}

async function initApiCommandPopout(): Promise<void> {
  applySavedVisualShell();
  renderApiCommandPopoutShell();
  loadApiLogsFromStorage();
  renderApiLogs();
  el<HTMLSelectElement>("#api-log-filter").addEventListener("change", renderApiLogs);
  el("#api-copy-log-btn").addEventListener("click", () => {
    const text = apiLogs.map((entry) => `[${new Date(entry.time).toISOString()}] ${entry.level.toUpperCase()} ${entry.method || entry.kind} ${entry.status || ""} ${entry.route || "/api"} ${entry.message}`).join("\n");
    void copyText(text, "API log");
  });
  el("#api-clear-log-btn").addEventListener("click", () => {
    apiLogs.splice(0, apiLogs.length);
    persistApiLogs();
    renderApiLogs();
  });
  await listen<ApiLogEvent>("api-log", (event) => {
    appendApiLog(event.payload);
  });
  window.addEventListener("storage", (event) => {
    if (event.key === API_LOG_STORAGE_KEY) syncApiLogsFromStorage();
  });
}

async function refreshApiStatus(): Promise<void> {
  try {
    const status = await invoke<ApiServerStatus>("api_server_status");
    apiRunning = status.running;
    if (status.host) el<HTMLInputElement>("#api-host").value = status.host;
    if (status.port) el<HTMLInputElement>("#api-port").value = String(status.port);
    if (status.baseUrl) el<HTMLInputElement>("#api-base-url").value = status.baseUrl;
    setApiVisualStatus(status.running ? "running" : "stopped", status.running ? "Running" : "Stopped");
  } catch (e) {
    setApiVisualStatus("error", "API status error");
    appendApiLog({ level: "error", kind: "server", message: `Status failed: ${e}` });
  }
}

function initApiPanel(): void {
  loadApiLogsFromStorage();
  const savedKey = localStorage.getItem("higgsAudio.apiKey") || generateApiKey();
  el<HTMLInputElement>("#api-key").value = savedKey;
  localStorage.setItem("higgsAudio.apiKey", savedKey);
  el<HTMLInputElement>("#api-host").value = localStorage.getItem("higgsAudio.apiHost") || "127.0.0.1";
  el<HTMLInputElement>("#api-port").value = localStorage.getItem("higgsAudio.apiPort") || "7077";
  updateApiBaseUrl();
  renderApiLogs();
  renderApiExample();

  el("#api-status-button").addEventListener("click", () => switchMode("api"));
  el("#api-info-btn").addEventListener("click", () => {
    const panel = el<HTMLElement>("#api-docs-panel");
    const hidden = panel.classList.toggle("hidden");
    el<HTMLButtonElement>("#api-info-btn").classList.toggle("active", !hidden);
    renderApiExample();
  });
  for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-api-example]")) {
    tab.addEventListener("click", () => {
      selectedApiExample = (tab.dataset.apiExample || "curl") as ApiExampleKind;
      renderApiExample();
    });
  }
  for (const id of ["#api-host", "#api-port"]) {
    el<HTMLInputElement>(id).addEventListener("input", () => {
      localStorage.setItem("higgsAudio.apiHost", el<HTMLInputElement>("#api-host").value.trim() || "127.0.0.1");
      localStorage.setItem("higgsAudio.apiPort", el<HTMLInputElement>("#api-port").value.trim() || "7077");
      updateApiBaseUrl();
    });
  }
  el<HTMLInputElement>("#api-key").addEventListener("input", () => {
    localStorage.setItem("higgsAudio.apiKey", el<HTMLInputElement>("#api-key").value);
    renderApiExample();
  });
  el<HTMLInputElement>("#api-key").addEventListener("change", () => {
    appendApiLog({ kind: "server", message: "Custom local API key updated" });
  });
  el("#api-key-toggle").addEventListener("click", () => {
    const input = el<HTMLInputElement>("#api-key");
    const button = el<HTMLButtonElement>("#api-key-toggle");
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    button.textContent = show ? "◉" : "👁";
    button.setAttribute("aria-label", show ? "Hide API key" : "Show API key");
    button.title = show ? "Hide API key" : "Show API key";
  });
  el("#api-generate-key-btn").addEventListener("click", () => {
    const key = generateApiKey();
    el<HTMLInputElement>("#api-key").value = key;
    localStorage.setItem("higgsAudio.apiKey", key);
    renderApiExample();
    appendApiLog({ kind: "server", message: "Generated new local API key" });
  });
  el("#api-copy-base-btn").addEventListener("click", () => copyText(el<HTMLInputElement>("#api-base-url").value, "Base URL"));
  el("#api-copy-key-btn").addEventListener("click", () => copyText(el<HTMLInputElement>("#api-key").value, "API key"));
  el("#api-copy-curl-btn").addEventListener("click", () => {
    void copyText(buildCurrentApiExample("curl"), "curl examples");
  });
  el("#api-copy-example-btn").addEventListener("click", () => {
    void copyText(buildCurrentApiExample(selectedApiExample), "API example");
  });
  el("#api-start-btn").addEventListener("click", async () => {
    setApiVisualStatus("starting", "Starting");
    const host = el<HTMLInputElement>("#api-host").value.trim() || "127.0.0.1";
    const port = parseInt(el<HTMLInputElement>("#api-port").value, 10) || 7077;
    const apiKey = el<HTMLInputElement>("#api-key").value.trim() || generateApiKey();
    el<HTMLInputElement>("#api-key").value = apiKey;
    localStorage.setItem("higgsAudio.apiKey", apiKey);
    renderApiExample();
    try {
      for (const persona of speakerPersonas) {
        if (persona.refPath) await ensureSpeakerCachePath(persona);
      }
      const status = await invoke<ApiServerStatus>("api_server_start", {
        config: { host, port, apiKey, speakers: apiSpeakerPayload() },
      });
      apiRunning = true;
      if (status.baseUrl) el<HTMLInputElement>("#api-base-url").value = status.baseUrl;
      renderApiExample();
      setApiVisualStatus("running", "Running");
      appendApiLog({ kind: "server", method: "START", route: "/api", status: 200, message: "API server running" });
    } catch (e) {
      setApiVisualStatus("error", "Start failed");
      appendApiLog({ level: "error", kind: "server", method: "START", route: "/api", status: 500, message: String(e) });
    }
  });
  el("#api-stop-btn").addEventListener("click", async () => {
    try {
      await invoke("api_server_stop");
      apiRunning = false;
      setApiVisualStatus("stopped", "Stopped");
      appendApiLog({ kind: "server", method: "STOP", route: "/api", status: 200, message: "API server stopped" });
    } catch (e) {
      setApiVisualStatus("error", "Stop failed");
      appendApiLog({ level: "error", kind: "server", method: "STOP", route: "/api", status: 500, message: String(e) });
    }
  });
  el<HTMLSelectElement>("#api-log-filter").addEventListener("change", renderApiLogs);
  el("#api-popout-log-btn").addEventListener("click", () => {
    void openApiCommandPopout();
  });
  el("#api-clear-log-btn").addEventListener("click", () => {
    apiLogs.length = 0;
    persistApiLogs();
    renderApiLogs();
  });
  el("#api-copy-log-btn").addEventListener("click", () => {
    const text = apiLogs.map((entry) => {
      const time = new Date(entry.time).toISOString();
      return `${time} ${entry.level.toUpperCase()} ${entry.method || entry.kind} ${entry.route} ${entry.status} ${entry.latencyMs}ms ${entry.message}`;
    }).join("\n");
    void copyText(text, "API log");
  });
  window.addEventListener("storage", (event) => {
    if (event.key === API_LOG_STORAGE_KEY) syncApiLogsFromStorage();
  });
  void refreshApiStatus();
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode switching
// ═══════════════════════════════════════════════════════════════════════════

function switchMode(mode: Mode): void {
  currentMode = mode;
  if (mode === "api") {
    el<HTMLButtonElement>("#api-status-button").classList.remove("hidden");
  } else if (!apiRunning) {
    el<HTMLButtonElement>("#api-status-button").classList.add("hidden");
  }
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode-tab")) {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.classList.toggle("generating", tab.dataset.mode === activeGenerationJob?.mode);
    tab.setAttribute("aria-selected", String(active));
  }
  for (const content of document.querySelectorAll<HTMLElement>(".mode-content")) {
    content.classList.toggle("hidden", content.id !== `mode-${mode}`);
  }
  // Hide generation UI in utility views.
  const isUtility = mode === "history" || mode === "api" || mode === "speakers";
  el<HTMLElement>("#advanced-details").classList.toggle("hidden", isUtility);
  el<HTMLElement>("#action-row").classList.toggle("hidden", isUtility);
  el<HTMLElement>("#progress-section").classList.toggle("hidden", isUtility || !isGenerating || activeGenerationJob?.mode !== mode);
  for (const item of document.querySelectorAll<HTMLElement>(".multi-only-advanced")) {
    item.classList.toggle("hidden", mode !== "multi");
  }
  updateGenerationControls();

  // Show this mode's output if it has one, otherwise hide
  const modeOutput = outputByMode[mode];
  if (modeOutput && !isUtility) {
    lastResult = modeOutput;
    showOutput(modeOutput);
  } else {
    el<HTMLElement>("#output-section").classList.add("hidden");
  }
  if (mode === "speakers") refreshVisibleRefPreview("gallery");
}

function initModeTabs(): void {
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode-tab")) {
    tab.addEventListener("click", () => switchMode(tab.dataset.mode as Mode));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Model management
// ═══════════════════════════════════════════════════════════════════════════

function isRecommendedTtsModel(model: ModelListing): boolean {
  return model.name.toLowerCase().includes("q8_0");
}

function addLocalModelOption(path: string, label?: string): void {
  if (!path) return;
  const select = el<HTMLSelectElement>("#model-select");
  for (const opt of select.options) {
    if (opt.value === path) {
      select.value = path;
      return;
    }
  }
  const dirName = label || path.split(/[/\\]/).pop() || path;
  const opt = document.createElement("option");
  opt.value = path;
  opt.textContent = `${dirName} (local)`;
  select.appendChild(opt);
  select.value = path;
}

function modelDisplayName(model: ModelListing): string {
  const name = model.name.toLowerCase();
  if (name.includes("bf16")) return "Higgs Audio v3 BF16";
  if (name.includes("q8_0")) return "Higgs Audio v3 Q8_0";
  if (name.includes("q6_k")) return "Higgs Audio v3 Q6_K";
  if (name.includes("q5_k")) return "Higgs Audio v3 Q5_K";
  if (name.includes("q4_k")) return "Higgs Audio v3 Q4_K_M";
  return model.name;
}

function modelNameFromPath(path: string): string {
  const lower = path.toLowerCase();
  if (lower.includes("bf16")) return "Higgs Audio v3 BF16";
  if (lower.includes("q8_0") || lower.includes("q8-0")) return "Higgs Audio v3 Q8_0";
  if (lower.includes("q6_k") || lower.includes("q6-k")) return "Higgs Audio v3 Q6_K";
  if (lower.includes("q5_k") || lower.includes("q5-k")) return "Higgs Audio v3 Q5_K";
  if (lower.includes("q4_k") || lower.includes("q4-k")) return "Higgs Audio v3 Q4_K_M";
  return path.split(/[/\\]/).filter(Boolean).pop() || "Higgs model";
}

function selectedModelUiName(): string {
  const select = el<HTMLSelectElement>("#model-select");
  const value = select.value;
  const option = select.selectedOptions[0];
  if (value) {
    const inferred = modelNameFromPath(value);
    if (inferred !== "Higgs model" && !inferred.endsWith("model")) return inferred;
  }
  if (option?.textContent) {
    return option.textContent
      .replace(/^★\s*/, "")
      .replace(/\s*·\s*recommended/i, "")
      .replace(/\s*\([^)]*\)\s*$/, "")
      .trim();
  }
  return value ? modelNameFromPath(value) : "Higgs model";
}

function modelSortRank(model: ModelListing): number {
  const name = model.name.toLowerCase();
  if (name.includes("q8_0")) return 0;
  if (name.includes("q6_k")) return 1;
  if (name.includes("q5_k")) return 2;
  if (name.includes("q4_k")) return 3;
  if (name.includes("bf16")) return 4;
  return 10;
}

async function refreshModelList(): Promise<void> {
  try {
    const models = await invoke<ModelListing[]>("list_models");
    const sortedModels = [...models].sort((a, b) => {
      const rank = modelSortRank(a) - modelSortRank(b);
      return rank !== 0 ? rank : a.name.localeCompare(b.name);
    });
    const select = el<HTMLSelectElement>("#model-select");
    const currentVal = select.value || localStorage.getItem(MODEL_PATH_STORAGE_KEY) || "";
    select.innerHTML = '<option value="">Select a model…</option>';
    for (const m of sortedModels) {
      const opt = document.createElement("option");
      const recommended = isRecommendedTtsModel(m);
      const incomplete = m.hasConfig === false;
      opt.value = m.path;
      opt.textContent = `${recommended ? "★ " : ""}${modelDisplayName(m)}${recommended ? " · recommended" : ""} (${m.format}, ${formatBytes(m.sizeBytes)})${incomplete ? " · ⚠ incomplete" : ""}`;
      opt.title = incomplete
        ? `${m.name} — missing config.json / tokenizer files. Re-download the full model folder.`
        : m.name;
      select.appendChild(opt);
    }
    if (currentVal && models.some((m) => m.path === currentVal)) {
      select.value = currentVal;
    } else if (currentVal) {
      addLocalModelOption(currentVal);
    }
    el<HTMLButtonElement>("#load-model-btn").disabled = false;
  } catch (e) {
    showToast(`Failed to list models: ${e}`, "error");
  }
}

async function doLoadEngine(libraryPath?: string): Promise<void> {
  try {
    const savedEnginePath = localStorage.getItem(ENGINE_PATH_STORAGE_KEY) || "";
    const bundled = libraryPath ? null : await invoke<string | null>("bundled_engine_path");
    let libPath = libraryPath || savedEnginePath || bundled || undefined;
    let diagnostic = await diagnoseEngineForPath(libPath);
    if (!libraryPath && savedEnginePath && !diagnostic && bundled) {
      localStorage.removeItem(ENGINE_PATH_STORAGE_KEY);
      libPath = bundled;
      diagnostic = await diagnoseEngineForPath(libPath);
    }
    if (diagnostic && !diagnostic.ok) {
      showEngineDiagnosticModal(diagnostic);
      showToast("Engine dependency missing. Opened repair details.", "error");
      return;
    }
    const result = await invoke<{ success: boolean; version: string; supportsStreaming?: boolean }>("load_engine", {
      libraryPath: libPath,
    });
    if (result.success) {
      if (libPath) localStorage.setItem(ENGINE_PATH_STORAGE_KEY, libPath);
      engineSupportsStreaming = Boolean(result.supportsStreaming);
      setText("#engine-chip", t("Engine loaded"));
      el<HTMLElement>("#engine-chip").classList.add("active");
      showToast(t("Engine loaded"));
      await refreshModelList();
    }
  } catch (e) {
    const savedEnginePath = localStorage.getItem(ENGINE_PATH_STORAGE_KEY) || "";
    const bundled = await invoke<string | null>("bundled_engine_path").catch(() => null);
    const diagnostic = await diagnoseEngineForPath(libraryPath || savedEnginePath || bundled || undefined);
    if (diagnostic) {
      showEngineDiagnosticModal({ ...diagnostic, rawError: String(e) });
    }
    showToast(`Failed to load engine: ${e}`, "error");
  }
}

async function doBrowseEngine(): Promise<string | null> {
  const selected = await open({
    filters: [{ name: "Higgs Engine", extensions: ["dll", "so", "dylib"] }],
  });
  const path = Array.isArray(selected) ? selected[0] : selected;
  if (!path) return null;
  localStorage.setItem(ENGINE_PATH_STORAGE_KEY, path);
  showToast(`Selected engine: ${path.split(/[/\\]/).pop() || path}`);
  return path;
}

async function doLoadModel(): Promise<void> {
  const modelRoot = el<HTMLSelectElement>("#model-select").value;
  if (!modelRoot) {
    showToast("Select a model first", "warning");
    return;
  }
  try {
    el<HTMLButtonElement>("#load-model-btn").disabled = true;
    setText("#model-state", t("Loading…"));
    const result = await invoke<{ success: boolean; modelInfo: { family: string; displayName: string; weightType: string } }>(
      "load_model",
      {
        request: {
          modelRoot,
          backend: "cuda",
          device: 0,
          threads: 4,
          weightType: null,
          sessionOptions: null,
        },
      },
    );
    if (result.success) {
      const info = result.modelInfo;
      const uiName = selectedModelUiName();
      setText("#model-state", t("Loaded"));
      el("#model-state").classList.add("ok");
      setText("#model-chip", `${uiName} (${info.weightType || "default"})`);
      el("#model-chip").classList.remove("muted");
      el("#model-chip").classList.add("active");
      el<HTMLButtonElement>("#unload-model-btn").disabled = false;
      showToast(`Model loaded: ${uiName}`);
    }
  } catch (e) {
    setText("#model-state", t("Error"));
    showToast(`Failed to load model: ${e}`, "error");
  } finally {
    el<HTMLButtonElement>("#load-model-btn").disabled = false;
  }
}

async function doUnloadModel(): Promise<void> {
  try {
    await invoke("unload_model");
    setText("#model-state", t("Not loaded"));
    el("#model-state").classList.remove("ok");
    setText("#model-chip", t("No model"));
    el("#model-chip").classList.add("muted");
    el("#model-chip").classList.remove("active");
    el<HTMLButtonElement>("#unload-model-btn").disabled = true;
    showToast("Model unloaded");
  } catch (e) {
    showToast(`Failed to unload: ${e}`, "error");
  }
}

function initModelPanel(): void {
  const savedModelPath = localStorage.getItem(MODEL_PATH_STORAGE_KEY) || "";
  if (savedModelPath) addLocalModelOption(savedModelPath);
  el("#load-engine-btn").addEventListener("click", () => {
    void doLoadEngine();
  });
  el("#load-model-btn").addEventListener("click", doLoadModel);
  el("#unload-model-btn").addEventListener("click", doUnloadModel);
  el("#browse-model-btn").addEventListener("click", doBrowseModel);
  el<HTMLSelectElement>("#model-select").addEventListener("change", () => {
    const value = el<HTMLSelectElement>("#model-select").value;
    if (value) localStorage.setItem(MODEL_PATH_STORAGE_KEY, value);
    const popover = document.querySelector<HTMLDivElement>("#download-popover");
    if (!downloadActive && activeDownloadKind === "model" && popover && !popover.hidden) {
      setTtsDownloadPreset(inferTtsPresetFromModelSelection(), true);
    }
  });
}

async function doBrowseModel(): Promise<string | null> {
  const selected = await open({ directory: true, multiple: false });
  if (!selected) return null;

  const dirPath = Array.isArray(selected) ? selected[0] : selected;
  const dirName = dirPath.split(/[/\\]/).pop() || dirPath;
  localStorage.setItem(MODEL_PATH_STORAGE_KEY, dirPath);

  // Check for model files
  let hasWeights = false;
  let format = "";
  try {
    const models = await invoke<ModelListing[]>("list_models");
    const found = models.find((m) => m.path === dirPath);
    if (found) {
      hasWeights = true;
      format = found.format;
    }
  } catch {
    // list_models might not cover this dir; check via filesystem
  }

  // Add to dropdown regardless — the load will validate
  const select = el<HTMLSelectElement>("#model-select");
  for (const opt of select.options) {
    if (opt.value === dirPath) {
      select.value = dirPath;
      showToast(`Selected: ${dirName}`);
      return dirPath;
    }
  }
  const opt = document.createElement("option");
  opt.value = dirPath;
  opt.textContent = `${dirName} (local)`;
  select.appendChild(opt);
  select.value = dirPath;
  showToast(`Added local model: ${dirName}`);
  el<HTMLButtonElement>("#load-model-btn").disabled = false;
  return dirPath;
}

// ═══════════════════════════════════════════════════════════════════════════
// Dropzone
// ═══════════════════════════════════════════════════════════════════════════

function setupDropzone(
  dropzoneId: string,
  onFile: (path: string, name: string) => void | Promise<void>,
  onRemove?: () => void,
): void {
  const dz = el<HTMLElement>(dropzoneId);
  dz.dataset.emptyHtml = dz.innerHTML;
  dropzoneHandlers.push({ selector: dropzoneId, onFile });

  const handleFile = async (path: string, name: string) => {
    try {
      await onFile(path, name);
    } catch (e) {
      showToast(`Audio upload failed: ${e}`, "error");
    }
  };

  dz.addEventListener("click", async (event) => {
    const target = event.target as HTMLElement;
    if (target.closest(".dropzone-remove")) {
      event.preventDefault();
      event.stopPropagation();
      onRemove?.();
      return;
    }
    const selected = await open({
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "m4a", "ogg", "webm"] }],
    });
    if (selected) {
      const name = selected.split(/[/\\]/).pop() || selected;
      await handleFile(selected, name);
    }
  });

  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      const path = (file as any).path as string | undefined;
      if (!path) {
        return;
      }
      void handleFile(path, file.name);
    }
  });
}

function setDropzoneFile(dropzoneId: string, name: string): void {
  const dz = el<HTMLElement>(dropzoneId);
  dz.classList.add("has-file");
  dz.innerHTML = `
    <div class="dropzone-file">
      <span class="dropzone-file-icon">🎵</span>
      <span class="dropzone-file-name">${escapeHtml(name)}</span>
      <button class="dropzone-remove" type="button" aria-label="Remove audio">✕</button>
      <span class="dropzone-hint">Drop another audio file to replace it</span>
    </div>`;
}

function clearDropzone(dropzoneId: string): void {
  const dz = el<HTMLElement>(dropzoneId);
  dz.classList.remove("has-file", "drag-over");
  dz.innerHTML = dz.dataset.emptyHtml || "";
}

function clearAllDropzoneHover(): void {
  for (const item of document.querySelectorAll<HTMLElement>(".dropzone.drag-over")) {
    item.classList.remove("drag-over");
  }
}

async function routeDroppedAudioPath(target: HTMLElement | null, path: string): Promise<boolean> {
  const dz = target?.closest<HTMLElement>(".dropzone");
  if (!dz) return false;
  const name = path.split(/[/\\]/).pop() || path;
  const registered = dropzoneHandlers.find((handler) => dz.matches(handler.selector));
  if (registered) {
    await registered.onFile(path, name);
    return true;
  }
  if (dz.closest("#speaker-library")) {
    const speaker = speakerFromElement(dz);
    if (!speaker) return false;
    await setSpeakerAudioFromPath(speaker, path, name);
    return true;
  }
  if (dz.closest("#multi-lines")) {
    const line = lineFromElement(dz);
    if (!line) return false;
    await setLineAudioFromPath(line, path, name);
    return true;
  }
  return false;
}

function elementFromTauriDropPosition(position: { x: number; y: number }): HTMLElement | null {
  const scale = window.devicePixelRatio || 1;
  const attempts = [
    { x: position.x / scale, y: position.y / scale },
    { x: position.x, y: position.y },
  ];
  for (const point of attempts) {
    const target = document.elementFromPoint(point.x, point.y) as HTMLElement | null;
    if (target) return target;
  }
  return null;
}

function initTauriDropzones(): void {
  void getCurrentWebview().onDragDropEvent((event) => {
    const payload = event.payload as any;
    if (payload.type === "leave") {
      clearAllDropzoneHover();
      return;
    }
    if (!payload.position) return;
    const target = elementFromTauriDropPosition(payload.position);
    const dz = target?.closest<HTMLElement>(".dropzone");
    clearAllDropzoneHover();
    if (dz) dz.classList.add("drag-over");
    if (payload.type !== "drop") return;
    clearAllDropzoneHover();
    const firstPath = Array.isArray(payload.paths) ? payload.paths[0] : "";
    if (!firstPath) return;
    void routeDroppedAudioPath(target, firstPath).catch((e) => {
      showToast(`Audio drop failed: ${e}`, "error");
    });
  }).catch((e) => {
    console.warn("Tauri drag/drop unavailable", e);
  });
}

function drawRefPreview(kind: RefPreviewKind): void {
  const player = refPlayers[kind];
  if (!player) return;
  const ctx = player.canvas.getContext("2d");
  if (!ctx) return;
  const rect = player.canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (player.canvas.width !== Math.floor(width * dpr) || player.canvas.height !== Math.floor(height * dpr)) {
    player.canvas.width = Math.floor(width * dpr);
    player.canvas.height = Math.floor(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = cssVar("--bg-field", "#0f1215");
  ctx.fillRect(0, 0, width, height);
  const duration = player.audio.duration || 0;
  const progress = duration > 0 ? Math.min(1, player.audio.currentTime / duration) : 0;
  const accent = cssVar("--accent", "#25b8ab");
  const muted = cssVar("--text-muted", "#9ea8b3");
  if (player.waveformLoading) {
    ctx.fillStyle = muted;
    ctx.font = "12px Consolas, monospace";
    ctx.fillText("Loading waveform...", 10, Math.round(height / 2) + 4);
    return;
  }
  if (player.peaks.length === 0) {
    ctx.strokeStyle = `${muted}88`;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    return;
  }
  const bars = Math.max(28, Math.min(player.peaks.length, Math.floor(width / 6)));
  const gap = 3;
  const barW = Math.max(2, (width - gap * (bars - 1)) / bars);
  for (let i = 0; i < bars; i += 1) {
    const t = i / Math.max(1, bars - 1);
    const peakStart = Math.floor(i * player.peaks.length / bars);
    const peakEnd = Math.max(peakStart + 1, Math.floor((i + 1) * player.peaks.length / bars));
    let peak = 0;
    for (let p = peakStart; p < peakEnd; p++) peak = Math.max(peak, player.peaks[p] || 0);
    const barH = Math.max(2, peak * (height - 6));
    const x = i * (barW + gap);
    ctx.fillStyle = t <= progress ? accent : `${muted}55`;
    ctx.fillRect(x, (height - barH) / 2, barW, barH);
  }
}

async function loadRefWaveform(kind: RefPreviewKind, path: string): Promise<void> {
  const player = refPlayers[kind];
  if (!player) return;
  player.waveformPath = path;
  player.waveformLoading = true;
  player.peaks = [];
  drawRefPreview(kind);
  try {
    const result = await invoke<{ peaks: number[] }>("audio_waveform", { audioPath: path, points: 1400 });
    if (player.waveformPath !== path) return;
    player.peaks = result.peaks || [];
  } catch {
    if (player.waveformPath === path) player.peaks = [];
  } finally {
    if (player.waveformPath === path) {
      player.waveformLoading = false;
      drawRefPreview(kind);
    }
  }
}

function updateRefPlayback(kind: RefPreviewKind): void {
  const player = refPlayers[kind];
  if (!player) return;
  const duration = player.audio.duration || 0;
  const current = player.audio.currentTime || 0;
  player.seek.value = duration > 0 ? String(Math.round((current / duration) * 1000)) : "0";
  player.time.textContent = `${formatDuration(current)} / ${formatDuration(duration)}`;
  player.play.textContent = player.audio.paused ? "▶" : "⏸";
  drawRefPreview(kind);
}

function stopRefLoop(kind: RefPreviewKind): void {
  const player = refPlayers[kind];
  if (player?.raf) {
    cancelAnimationFrame(player.raf);
    player.raf = null;
  }
}

function startRefLoop(kind: RefPreviewKind): void {
  stopRefLoop(kind);
  const tick = () => {
    updateRefPlayback(kind);
    const player = refPlayers[kind];
    if (player && !player.audio.paused && !player.audio.ended) {
      player.raf = requestAnimationFrame(tick);
    }
  };
  refPlayers[kind]!.raf = requestAnimationFrame(tick);
}

function pauseOtherAudio(except?: RefPreviewKind): void {
  if (!audioPlayer.paused) audioPlayer.pause();
  el<HTMLButtonElement>("#play-btn").textContent = "▶";
  for (const kind of ["clone", "finish", "gallery"] as RefPreviewKind[]) {
    if (kind === except) continue;
    const player = refPlayers[kind];
    if (player && !player.audio.paused) {
      player.audio.pause();
      updateRefPlayback(kind);
    }
  }
}

function initRefPlayer(kind: RefPreviewKind): void {
  const player: RefPlayer = {
    audio: new Audio(),
    play: el<HTMLButtonElement>(`#${kind}-ref-play`),
    seek: el<HTMLInputElement>(`#${kind}-ref-seek`),
    time: el<HTMLElement>(`#${kind}-ref-time`),
    canvas: el<HTMLCanvasElement>(`#${kind}-ref-waveform`),
    peaks: [],
    waveformPath: "",
    waveformLoading: false,
    raf: null,
  };
  player.audio.preload = "metadata";
  player.play.addEventListener("click", async () => {
    if (!player.audio.src) return;
    if (player.audio.paused) {
      pauseOtherAudio(kind);
      try {
        await player.audio.play();
        startRefLoop(kind);
      } catch {
        stopRefLoop(kind);
        updateRefPlayback(kind);
        showToast("Could not play that audio file", "warning");
      }
    } else {
      player.audio.pause();
      stopRefLoop(kind);
      updateRefPlayback(kind);
    }
  });
  player.seek.addEventListener("input", () => {
    const duration = player.audio.duration || 0;
    if (duration > 0) {
      player.audio.currentTime = (parseInt(player.seek.value, 10) / 1000) * duration;
      updateRefPlayback(kind);
    }
  });
  player.audio.addEventListener("loadedmetadata", () => updateRefPlayback(kind));
  player.audio.addEventListener("timeupdate", () => updateRefPlayback(kind));
  player.audio.addEventListener("ended", () => {
    stopRefLoop(kind);
    updateRefPlayback(kind);
  });
  refPlayers[kind] = player;
  drawRefPreview(kind);
}

function showRefPreview(kind: RefPreviewKind, path: string): void {
  const preview = el<HTMLElement>(`#${kind}-ref-preview`);
  const player = refPlayers[kind];
  if (!player) return;
  preview.classList.remove("hidden");
  player.audio.pause();
  stopRefLoop(kind);
  player.audio.src = convertFileSrc(path);
  player.audio.load();
  player.seek.value = "0";
  void loadRefWaveform(kind, path);
  updateRefPlayback(kind);
}

function refreshVisibleRefPreview(kind: RefPreviewKind): void {
  const player = refPlayers[kind];
  const preview = document.querySelector<HTMLElement>(`#${kind}-ref-preview`);
  if (!player || !preview || preview.classList.contains("hidden") || !player.waveformPath) return;
  requestAnimationFrame(() => {
    drawRefPreview(kind);
    if (player.peaks.length === 0 && !player.waveformLoading) {
      void loadRefWaveform(kind, player.waveformPath);
    }
  });
}

function redrawAllRefPreviews(): void {
  for (const kind of ["clone", "finish", "gallery"] as RefPreviewKind[]) {
    refreshVisibleRefPreview(kind);
  }
}

function hideRefPreview(kind: RefPreviewKind): void {
  const player = refPlayers[kind];
  if (player) {
    player.audio.pause();
    stopRefLoop(kind);
    player.audio.removeAttribute("src");
    player.audio.load();
    player.peaks = [];
    player.waveformPath = "";
    player.waveformLoading = false;
    player.seek.value = "0";
    updateRefPlayback(kind);
  }
  el<HTMLElement>(`#${kind}-ref-preview`).classList.add("hidden");
}

function initDropzones(): void {
  initRefPlayer("clone");
  initRefPlayer("finish");
  initRefPlayer("gallery");

  const clearClone = () => {
    cloneRefPath = null;
    clonePersonaId = "";
    cloneRefName = "";
    clearDropzone("#clone-dropzone");
    hideRefPreview("clone");
  };
  const clearFinish = () => {
    finishRefPath = null;
    finishPersonaId = "";
    finishRefName = "";
    clearDropzone("#finish-dropzone");
    hideRefPreview("finish");
  };
  const clearGallery = () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.refPath = "";
    persona.refName = "";
    persona.cachePath = "";
    persona.updatedAt = Date.now();
    saveSpeakerPersonas();
    hideRefPreview("gallery");
  };

  setupDropzone("#clone-dropzone", async (path, name) => {
    const prepared = await prepareReferenceAudioFile(path, name);
    cloneRefPath = prepared.path;
    clonePersonaId = "";
    cloneRefName = prepared.name;
    setDropzoneFile("#clone-dropzone", prepared.name);
    showRefPreview("clone", prepared.path);
  }, clearClone);
  setupDropzone("#finish-dropzone", (path, name) => {
    finishRefPath = path;
    finishPersonaId = "";
    finishRefName = name;
    setDropzoneFile("#finish-dropzone", name);
    showRefPreview("finish", path);
  }, clearFinish);
  setupDropzone("#speaker-gallery-dropzone", (path, name) => {
    void (async () => {
      let persona = selectedGalleryPersona();
      if (!persona) {
        persona = createBlankPersona();
        speakerPersonas.push(persona);
        selectedGalleryPersonaId = persona.id;
      }
      const prepared = await prepareReferenceAudioFile(path, name);
      if (!persona.name || /^Speaker \d+$/.test(persona.name)) {
        persona.name = personaNameFromPath(name || path);
      }
      const stored = await storeSpeakerAsset(persona, prepared.path, "audio");
      persona.refPath = stored.path;
      persona.refName = prepared.name || stored.fileName;
      persona.cachePath = "";
      await ensureSpeakerCachePath(persona);
      persona.updatedAt = Date.now();
      saveSpeakerPersonas();
      showRefPreview("gallery", persona.refPath);
    })().catch((e) => showToast(`Reference copy failed: ${e}`, "error"));
  }, clearGallery);
}

async function setGalleryPhotoFromPath(imagePath: string): Promise<void> {
  let persona = selectedGalleryPersona();
  if (!persona) {
    persona = createBlankPersona();
    speakerPersonas.push(persona);
    selectedGalleryPersonaId = persona.id;
  }
  const stored = await storeSpeakerAsset(persona, imagePath, "image");
  persona.photoPath = stored.path;
  persona.updatedAt = Date.now();
  saveSpeakerPersonas();
}

// ═══════════════════════════════════════════════════════════════════════════
// Auto-transcribe
// ═══════════════════════════════════════════════════════════════════════════

async function doAutoTranscribe(refPath: string | null, textareaId: string): Promise<void> {
  if (!refPath) {
    showToast("Drop an audio file first", "warning");
    return;
  }
  const btn = refPath === cloneRefPath
    ? el<HTMLButtonElement>("#auto-transcribe-btn")
    : el<HTMLButtonElement>("#finish-auto-transcribe-btn");
  if (btn.disabled) return;
  btn.disabled = true;
  btn.classList.add("transcribing");
  try {
    const text = await transcribeAudioText(refPath);
    if (text === null) return;
    const textarea = el<HTMLTextAreaElement>(textareaId);
    textarea.value = text;
    textarea.classList.add("flash");
    setTimeout(() => textarea.classList.remove("flash"), 1000);
  } catch (e) {
    showToast(`Transcription failed: ${e}`, "error");
  } finally {
    btn.disabled = false;
    btn.classList.remove("transcribing");
  }
}

async function transcribeAudioText(refPath: string): Promise<string | null> {
  const whisperModelPath = localStorage.getItem("higgsAudio.whisperModel") || "";
  if (!whisperModelPath) {
    showToast("Select a whisper model in Settings first", "warning");
    return null;
  }
  const result = await invoke<{ text: string }>("transcribe_audio", {
    audioPath: refPath,
    whisperModelPath,
    language: null,
  });
  return result.text;
}

async function tryAutoTranscribeSilently(refPath: string): Promise<string> {
  const whisperModelPath = localStorage.getItem("higgsAudio.whisperModel") || "";
  if (!whisperModelPath) return "";
  try {
    const result = await invoke<{ text: string }>("transcribe_audio", {
      audioPath: refPath,
      whisperModelPath,
      language: null,
    });
    return result.text.trim();
  } catch {
    return "";
  }
}

async function pickAudioFile(): Promise<{ path: string; name: string } | null> {
  const selected = await open({
    filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "m4a", "ogg", "webm"] }],
  });
  if (!selected) return null;
  const path = Array.isArray(selected) ? selected[0] : selected;
  return { path, name: path.split(/[/\\]/).pop() || path };
}

async function prepareReferenceAudioFile(path: string, name?: string): Promise<{ path: string; name: string; cropped: boolean }> {
  const originalName = name || path.split(/[/\\]/).pop() || "reference audio";
  const result = await invoke<PreparedReferenceUpload>("prepare_reference_upload", {
    audioPath: path,
    maxSeconds: HIGGS_REFERENCE_MAX_SECONDS,
  });
  if (result.cropped) {
    showToast(`Reference cropped to first ${HIGGS_REFERENCE_MAX_SECONDS} seconds`, "warning");
  }
  return {
    path: result.path,
    name: result.cropped ? `${originalName} (first ${HIGGS_REFERENCE_MAX_SECONDS}s)` : originalName,
    cropped: result.cropped,
  };
}

async function pickReferenceAudioFile(): Promise<{ path: string; name: string; cropped: boolean } | null> {
  const file = await pickAudioFile();
  return file ? prepareReferenceAudioFile(file.path, file.name) : null;
}

async function pickImageFile(): Promise<string | null> {
  const selected = await open({
    filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp", "bmp"] }],
  });
  if (!selected) return null;
  return Array.isArray(selected) ? selected[0] : selected;
}

function cleanPersona(raw: any): SpeakerPersona | null {
  if (!raw || typeof raw !== "object") return null;
  const id = String(raw.id || "").trim() || nextId("persona");
  const name = String(raw.name || "Speaker").trim() || "Speaker";
  const refPath = String(raw.refPath || raw.ref_path || "").trim();
  return {
    id,
    name,
    refPath,
    refName: String(raw.refName || raw.ref_name || (refPath ? refPath.split(/[/\\]/).pop() : "") || ""),
    refText: String(raw.refText || raw.ref_text || ""),
    notes: String(raw.notes || ""),
    photoPath: String(raw.photoPath || raw.photo_path || ""),
    cachePath: String(raw.cachePath || raw.cache_path || ""),
    normalize: Boolean(raw.normalize || raw.normalizeReference || raw.normalize_reference),
    createdAt: Number(raw.createdAt || raw.created_at || Date.now()),
    updatedAt: Number(raw.updatedAt || raw.updated_at || Date.now()),
  };
}

function loadSpeakerPersonas(): void {
  try {
    const parsed = JSON.parse(localStorage.getItem(SPEAKER_PERSONA_STORAGE_KEY) || "[]");
    const rawList = Array.isArray(parsed) ? parsed : parsed.speakers;
    speakerPersonas = Array.isArray(rawList)
      ? rawList.map(cleanPersona).filter((item): item is SpeakerPersona => Boolean(item))
      : [];
  } catch {
    speakerPersonas = [];
  }
}

function persistSpeakerPersonas(): void {
  localStorage.setItem(SPEAKER_PERSONA_STORAGE_KEY, JSON.stringify(speakerPersonas));
}

function saveSpeakerPersonas(): void {
  speakerPersonas.sort((a, b) => a.name.localeCompare(b.name));
  persistSpeakerPersonas();
  for (const persona of speakerPersonas) scheduleSpeakerPersonaFolderSync(persona, 80);
  renderSpeakerPersonaUi();
  if (apiRunning) {
    void syncApiSpeakers("Speaker identities refreshed");
  }
}

async function syncApiSpeakers(message = "Speaker identities synced"): Promise<void> {
  try {
    for (const persona of speakerPersonas) {
      if (persona.refPath) await ensureSpeakerCachePath(persona);
    }
    const status = await invoke<{ running: boolean; speakerCount: number }>("api_update_speakers", {
      speakers: apiSpeakerPayload(),
    });
    if (status.running) {
      appendApiLog({
        kind: "server",
        method: "SYNC",
        route: "/api/speakers",
        status: 200,
        message: `${message}: ${status.speakerCount} saved speaker${status.speakerCount === 1 ? "" : "s"}`,
      });
    }
  } catch (e) {
    appendApiLog({
      level: "error",
      kind: "server",
      method: "SYNC",
      route: "/api/speakers",
      status: 500,
      message: `Speaker sync failed: ${e}`,
    });
  }
}

function findPersona(id: string | null | undefined): SpeakerPersona | undefined {
  if (!id) return undefined;
  return speakerPersonas.find((persona) => persona.id === id);
}

function personaNameFromPath(path: string): string {
  const file = path.split(/[/\\]/).pop() || "Speaker";
  return file.replace(/\.[^.]+$/, "") || "Speaker";
}

function personaOptions(selectedId: string, emptyLabel = "Select saved identity…"): string {
  const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`];
  for (const persona of speakerPersonas) {
    options.push(`<option value="${escapeHtml(persona.id)}" ${persona.id === selectedId ? "selected" : ""}>${escapeHtml(persona.name)}</option>`);
  }
  return options.join("");
}

function avatarMarkup(name: string, photoPath = "", extraClass = ""): string {
  if (photoPath) {
    return `<span class="speaker-avatar ${extraClass}"><img src="${escapeHtml(convertFileSrc(photoPath))}" alt="" /></span>`;
  }
  const initial = (name.trim()[0] || "?").toUpperCase();
  return `<span class="speaker-avatar ${extraClass}">${escapeHtml(initial)}</span>`;
}

function setGalleryAvatar(name: string, photoPath = ""): void {
  const avatar = document.querySelector<HTMLElement>("#speaker-gallery-avatar");
  if (!avatar) return;
  avatar.className = "speaker-avatar large";
  if (photoPath) {
    avatar.innerHTML = `<img src="${escapeHtml(convertFileSrc(photoPath))}" alt="" />`;
  } else {
    avatar.textContent = (name.trim()[0] || "?").toUpperCase();
  }
}

function upsertSpeakerPersona(persona: SpeakerPersona): void {
  persona.updatedAt = Date.now();
  if (!persona.createdAt) persona.createdAt = persona.updatedAt;
  const existing = speakerPersonas.findIndex((item) => item.id === persona.id);
  if (existing >= 0) speakerPersonas[existing] = persona;
  else speakerPersonas.push(persona);
  saveSpeakerPersonas();
}

async function storeSpeakerAsset(persona: SpeakerPersona, sourcePath: string, kind: "audio" | "image"): Promise<{ path: string; fileName: string }> {
  return invoke<{ path: string; fileName: string }>("store_speaker_asset", {
    speakerId: persona.id,
    speakerName: persona.name,
    sourcePath,
    assetKind: kind,
  });
}

async function restageSpeakerAssets(persona: SpeakerPersona): Promise<void> {
  if (persona.refPath) {
    const stored = await storeSpeakerAsset(persona, persona.refPath, "audio");
    persona.refPath = stored.path;
    persona.refName = persona.refName || stored.fileName;
    if (!persona.cachePath) await ensureSpeakerCachePath(persona);
  }
  if (persona.photoPath) {
    const stored = await storeSpeakerAsset(persona, persona.photoPath, "image");
    persona.photoPath = stored.path;
  }
}

async function syncSpeakerPersonaFolder(persona: SpeakerPersona): Promise<void> {
  const result = await invoke<{ cachePath?: string }>("write_speaker_metadata", { speaker: persona });
  if (result.cachePath) {
    const current = findPersona(persona.id);
    if (current && !current.cachePath) {
      current.cachePath = result.cachePath;
      persistSpeakerPersonas();
    }
  }
}

async function ensureSpeakerCachePath(persona: SpeakerPersona): Promise<string> {
  if (persona.cachePath) return persona.cachePath;
  const result = await invoke<{ path: string; exists: boolean }>("speaker_cache_path", {
    speakerId: persona.id,
    speakerName: persona.name,
  });
  persona.cachePath = result.path;
  persistSpeakerPersonas();
  scheduleSpeakerPersonaFolderSync(persona, 80);
  return result.path;
}

function scheduleSpeakerPersonaFolderSync(persona: SpeakerPersona, delay = 400): void {
  const existing = speakerSyncTimers.get(persona.id);
  if (existing) window.clearTimeout(existing);
  const snapshot = { ...persona };
  const timer = window.setTimeout(() => {
    speakerSyncTimers.delete(snapshot.id);
    void syncSpeakerPersonaFolder(snapshot).catch((e) => {
      console.warn("Speaker folder sync failed", e);
    });
  }, delay);
  speakerSyncTimers.set(persona.id, timer);
}

function selectedGalleryPersona(): SpeakerPersona | undefined {
  return findPersona(selectedGalleryPersonaId);
}

type VoicePackVoice = { name: string; audioPath: string; text: string };

// Download the standard Nerual Dreming voice pack and add every voice to the
// Speaker Gallery as a persona (reference audio + transcript). Idempotent:
// voices whose name already exists are skipped.
async function importStandardVoicePack(): Promise<void> {
  const btn = document.querySelector<HTMLButtonElement>("#speaker-voicepack-btn");
  if (btn) btn.disabled = true;
  showToast("Downloading standard voice pack…");
  try {
    const voices = await invoke<VoicePackVoice[]>("download_voicepack", { force: false });
    const existing = new Set(speakerPersonas.map((p) => p.name.trim().toLowerCase()));
    let added = 0;
    for (const voice of voices) {
      const key = voice.name.trim().toLowerCase();
      if (!key || existing.has(key)) continue;
      try {
        const persona = createBlankPersona();
        persona.name = voice.name;
        const prepared = await prepareReferenceAudioFile(voice.audioPath, voice.name);
        const stored = await storeSpeakerAsset(persona, prepared.path, "audio");
        persona.refPath = stored.path;
        persona.refName = prepared.name || stored.fileName;
        persona.refText = voice.text;
        persona.cachePath = "";
        persona.updatedAt = Date.now();
        speakerPersonas.push(persona);
        existing.add(key);
        added += 1;
      } catch (e) {
        console.warn(`Voice-pack import failed for ${voice.name}`, e);
      }
    }
    saveSpeakerPersonas();
    renderSpeakerPersonaUi();
    showToast(added > 0 ? `Standard voices added: ${added}` : "Standard voices already imported");
  } catch (e) {
    showToast(`Voice pack failed: ${e}`, "error");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function createBlankPersona(): SpeakerPersona {
  const now = Date.now();
  return {
    id: nextId("persona"),
    name: `Speaker ${speakerPersonas.length + 1}`,
    refPath: "",
    refName: "",
    refText: "",
    notes: "",
    photoPath: "",
    cachePath: "",
    normalize: false,
    createdAt: now,
    updatedAt: now,
  };
}

function renderSpeakerGalleryList(): void {
  const list = document.querySelector<HTMLElement>("#speaker-gallery-list");
  if (!list) return;
  list.innerHTML = "";
  if (speakerPersonas.length === 0) {
    const empty = document.createElement("div");
    empty.className = "speaker-gallery-empty";
    empty.textContent = t("No saved speakers yet");
    list.appendChild(empty);
    return;
  }
  for (const persona of speakerPersonas) {
    const item = document.createElement("button");
    item.className = `speaker-gallery-item ${persona.id === selectedGalleryPersonaId ? "active" : ""}`;
    item.type = "button";
    item.dataset.personaId = persona.id;
    item.innerHTML = `
      ${avatarMarkup(persona.name, persona.photoPath)}
      <span class="speaker-gallery-item-main">
        <strong>${escapeHtml(persona.name)}</strong>
        <span>${persona.refName ? escapeHtml(persona.refName) : "No reference audio"}</span>
      </span>
    `;
    list.appendChild(item);
  }
  translateStaticDom(list);
}

function renderSpeakerGalleryEditor(): void {
  const editor = document.querySelector<HTMLElement>("#speaker-gallery-editor");
  if (!editor) return;
  if (!selectedGalleryPersonaId && speakerPersonas.length > 0) {
    selectedGalleryPersonaId = speakerPersonas[0].id;
  }
  const persona = selectedGalleryPersona();
  editor.classList.toggle("hidden", !persona);
  if (!persona) {
    hideRefPreview("gallery");
    return;
  }

  el<HTMLInputElement>("#speaker-gallery-name").value = persona.name;
  el<HTMLTextAreaElement>("#speaker-gallery-transcript").value = persona.refText;
  el<HTMLTextAreaElement>("#speaker-gallery-notes").value = persona.notes;
  el<HTMLInputElement>("#speaker-gallery-normalize").checked = persona.normalize;
  setGalleryAvatar(persona.name, persona.photoPath);

  if (persona.refPath) {
    setDropzoneFile("#speaker-gallery-dropzone", persona.refName || persona.refPath.split(/[/\\]/).pop() || persona.name);
    showRefPreview("gallery", persona.refPath);
  } else {
    clearDropzone("#speaker-gallery-dropzone");
    hideRefPreview("gallery");
  }
}

function renderSpeakerPersonaUi(): void {
  const countText = `${speakerPersonas.length} saved`;
  const count = document.querySelector<HTMLElement>("#speaker-identity-count");
  if (count) count.textContent = countText;

  const cloneSelect = document.querySelector<HTMLSelectElement>("#clone-persona-select");
  if (cloneSelect) {
    const current = cloneSelect.value;
    cloneSelect.innerHTML = personaOptions(current);
    cloneSelect.value = findPersona(current) ? current : "";
  }

  const finishSelect = document.querySelector<HTMLSelectElement>("#finish-persona-select");
  if (finishSelect) {
    const current = finishSelect.value;
    finishSelect.innerHTML = personaOptions(current);
    finishSelect.value = findPersona(current) ? current : "";
  }

  renderSpeakerGalleryList();
  renderSpeakerGalleryEditor();
  if (multiSpeakers.length > 0 && document.querySelector("#speaker-library")) renderMultiWorkflow();
  if (document.querySelector("#api-example-code")) renderApiExample();
}

function applyPersonaToClone(personaId: string): void {
  const persona = findPersona(personaId);
  if (!persona) {
    showToast("Select a saved speaker identity first", "warning");
    return;
  }
  clonePersonaId = persona.id;
  void ensureSpeakerCachePath(persona).catch((e) => console.warn("Speaker cache path failed", e));
  el<HTMLTextAreaElement>("#clone-ref-text").value = persona.refText;
  el<HTMLInputElement>("#clone-normalize-ref").checked = persona.normalize;
  if (persona.refPath) {
    cloneRefPath = persona.refPath;
    cloneRefName = persona.refName || persona.refPath.split(/[/\\]/).pop() || persona.name;
    setDropzoneFile("#clone-dropzone", cloneRefName);
    showRefPreview("clone", persona.refPath);
  }
  showToast(`Loaded speaker identity: ${persona.name}`);
}

function applyPersonaToFinish(personaId: string): void {
  const persona = findPersona(personaId);
  if (!persona) {
    showToast("Select a saved speaker identity first", "warning");
    return;
  }
  if (!persona.refPath) {
    showToast("That speaker identity has no reference audio", "warning");
    return;
  }
  finishPersonaId = persona.id;
  void ensureSpeakerCachePath(persona).catch((e) => console.warn("Speaker cache path failed", e));
  finishRefPath = persona.refPath;
  finishRefName = persona.refName || persona.refPath.split(/[/\\]/).pop() || persona.name;
  setDropzoneFile("#finish-dropzone", finishRefName);
  showRefPreview("finish", persona.refPath);
  el<HTMLTextAreaElement>("#finish-transcript").value = persona.refText;
  el<HTMLInputElement>("#finish-normalize-ref").checked = persona.normalize;
  showToast(`Loaded continuation source: ${persona.name}`);
}

function selectedExportPersonas(): SpeakerPersona[] {
  return Array.from(document.querySelectorAll<HTMLInputElement>("#speaker-export-list input[data-export-id]:checked"))
    .map((input) => findPersona(input.dataset.exportId))
    .filter((persona): persona is SpeakerPersona => Boolean(persona));
}

function updateSpeakerExportCount(): void {
  const selected = selectedExportPersonas().length;
  setText("#speaker-export-count", `${selected} selected`);
  el<HTMLButtonElement>("#speaker-export-confirm").disabled = selected === 0;
}

function renderSpeakerExportList(): void {
  const list = el<HTMLElement>("#speaker-export-list");
  list.innerHTML = "";
  for (const persona of speakerPersonas) {
    const label = document.createElement("label");
    label.className = "speaker-export-item";
    label.innerHTML = `
      <input type="checkbox" data-export-id="${escapeHtml(persona.id)}" checked />
      ${avatarMarkup(persona.name, persona.photoPath)}
      <span class="speaker-export-item-main">
        <strong>${escapeHtml(persona.name)}</strong>
        <span>${persona.refName ? escapeHtml(persona.refName) : "No reference audio"}</span>
      </span>
    `;
    list.appendChild(label);
  }
  updateSpeakerExportCount();
}

function openSpeakerExportPicker(): void {
  if (speakerPersonas.length === 0) {
    showToast("No saved speakers to export", "warning");
    return;
  }
  renderSpeakerExportList();
  el<HTMLElement>("#speaker-export-modal").classList.remove("hidden");
}

function closeSpeakerExportPicker(): void {
  el<HTMLElement>("#speaker-export-modal").classList.add("hidden");
}

async function exportSpeakerPersonas(personas: SpeakerPersona[]): Promise<void> {
  if (personas.length === 0) {
    showToast("Choose at least one speaker to export", "warning");
    return;
  }
  const path = await save({
    defaultPath: `higgs-speaker-gallery-v${APP_VERSION}.zip`,
    filters: [{ name: "Speaker Gallery ZIP", extensions: ["zip"] }],
  });
  if (!path) return;
  await invoke("export_speaker_zip", { path, speakers: personas });
  showToast(`Exported ${personas.length} speaker${personas.length === 1 ? "" : "s"}`);
}

function importSpeakerPersonasFromJsonText(text: string): number {
  const parsed = JSON.parse(text);
  const rawList = Array.isArray(parsed) ? parsed : parsed.speakers;
  if (!Array.isArray(rawList)) throw new Error("No speakers array found");
  let imported = 0;
  for (const raw of rawList) {
    const persona = cleanPersona(raw);
    if (!persona) continue;
    const existing = speakerPersonas.findIndex((item) => item.id === persona.id);
    if (existing >= 0) speakerPersonas[existing] = persona;
    else speakerPersonas.push(persona);
    if (!selectedGalleryPersonaId) selectedGalleryPersonaId = persona.id;
    imported += 1;
  }
  saveSpeakerPersonas();
  return imported;
}

async function importSpeakerPersonasFromPath(path: string): Promise<void> {
  let imported = 0;
  if (path.toLowerCase().endsWith(".zip")) {
    const result = await invoke<{ speakers: SpeakerPersona[] }>("import_speaker_zip", { path });
    for (const raw of result.speakers || []) {
      const persona = cleanPersona(raw);
      if (!persona) continue;
      const existing = speakerPersonas.findIndex((item) => item.id === persona.id);
      if (existing >= 0) speakerPersonas[existing] = persona;
      else speakerPersonas.push(persona);
      if (!selectedGalleryPersonaId) selectedGalleryPersonaId = persona.id;
      imported += 1;
    }
    saveSpeakerPersonas();
  } else {
    const text = await invoke<string>("read_text_file", { path });
    imported = importSpeakerPersonasFromJsonText(text);
  }
  showToast(`Imported ${imported} speaker identit${imported === 1 ? "y" : "ies"}`);
}

function deleteSelectedPersona(): void {
  const id = selectedGalleryPersonaId;
  const persona = findPersona(id);
  if (!persona) {
    showToast("Select a speaker identity to delete", "warning");
    return;
  }
  if (!window.confirm(`Delete speaker identity "${persona.name}"?`)) return;
  speakerPersonas = speakerPersonas.filter((item) => item.id !== id);
  for (const speaker of multiSpeakers) {
    if (speaker.personaId === id) speaker.personaId = "";
  }
  selectedGalleryPersonaId = speakerPersonas[0]?.id || "";
  saveSpeakerPersonas();
  showToast("Speaker identity deleted");
}

function initSpeakerPersonas(): void {
  loadSpeakerPersonas();
  selectedGalleryPersonaId = speakerPersonas[0]?.id || "";
  renderSpeakerPersonaUi();

  const openImport = async () => {
    const selected = await open({
      filters: [{ name: "Speaker Gallery", extensions: ["zip", "json"] }],
    });
    const path = Array.isArray(selected) ? selected[0] : selected;
    if (!path) return;
    await importSpeakerPersonasFromPath(path);
  };
  el("#speaker-add-btn").addEventListener("click", () => {
    const persona = createBlankPersona();
    speakerPersonas.push(persona);
    selectedGalleryPersonaId = persona.id;
    saveSpeakerPersonas();
    switchMode("speakers");
  });
  el("#speaker-voicepack-btn").addEventListener("click", () => {
    void importStandardVoicePack();
  });
  el("#speaker-import-btn").addEventListener("click", () => {
    void openImport().catch((e) => showToast(`Import failed: ${e}`, "error"));
  });
  el("#multi-import-identities-btn").addEventListener("click", () => {
    void openImport().catch((e) => showToast(`Import failed: ${e}`, "error"));
  });
  el("#speaker-export-btn").addEventListener("click", () => {
    openSpeakerExportPicker();
  });
  el("#multi-export-identities-btn").addEventListener("click", () => {
    openSpeakerExportPicker();
  });
  el("#speaker-delete-btn").addEventListener("click", deleteSelectedPersona);
  el("#speaker-export-close").addEventListener("click", closeSpeakerExportPicker);
  el("#speaker-export-cancel").addEventListener("click", closeSpeakerExportPicker);
  el("#speaker-export-list").addEventListener("change", updateSpeakerExportCount);
  el("#speaker-export-select-all").addEventListener("click", () => {
    for (const input of document.querySelectorAll<HTMLInputElement>("#speaker-export-list input[data-export-id]")) {
      input.checked = true;
    }
    updateSpeakerExportCount();
  });
  el("#speaker-export-select-none").addEventListener("click", () => {
    for (const input of document.querySelectorAll<HTMLInputElement>("#speaker-export-list input[data-export-id]")) {
      input.checked = false;
    }
    updateSpeakerExportCount();
  });
  el("#speaker-export-confirm").addEventListener("click", () => {
    const selected = selectedExportPersonas();
    void exportSpeakerPersonas(selected)
      .then(closeSpeakerExportPicker)
      .catch((e) => showToast(`Export failed: ${e}`, "error"));
  });

  el("#speaker-gallery-list").addEventListener("click", (event) => {
    const item = (event.target as HTMLElement).closest<HTMLElement>("[data-persona-id]");
    if (!item?.dataset.personaId) return;
    selectedGalleryPersonaId = item.dataset.personaId;
    renderSpeakerPersonaUi();
  });
  el<HTMLInputElement>("#speaker-gallery-name").addEventListener("input", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.name = el<HTMLInputElement>("#speaker-gallery-name").value || "Speaker";
    persona.updatedAt = Date.now();
    persistSpeakerPersonas();
    scheduleSpeakerPersonaFolderSync(persona);
    setGalleryAvatar(persona.name, persona.photoPath);
  });
  el<HTMLInputElement>("#speaker-gallery-name").addEventListener("change", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    void restageSpeakerAssets(persona)
      .catch((e) => showToast(`Speaker asset refresh failed: ${e}`, "warning"))
      .finally(() => saveSpeakerPersonas());
  });
  el<HTMLTextAreaElement>("#speaker-gallery-transcript").addEventListener("input", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.refText = el<HTMLTextAreaElement>("#speaker-gallery-transcript").value;
    persona.updatedAt = Date.now();
    persistSpeakerPersonas();
    scheduleSpeakerPersonaFolderSync(persona);
  });
  el<HTMLTextAreaElement>("#speaker-gallery-notes").addEventListener("input", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.notes = el<HTMLTextAreaElement>("#speaker-gallery-notes").value;
    persona.updatedAt = Date.now();
    persistSpeakerPersonas();
    scheduleSpeakerPersonaFolderSync(persona);
  });
  el<HTMLInputElement>("#speaker-gallery-normalize").addEventListener("change", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.normalize = el<HTMLInputElement>("#speaker-gallery-normalize").checked;
    persona.updatedAt = Date.now();
    saveSpeakerPersonas();
  });
  el("#speaker-gallery-photo-btn").addEventListener("click", async () => {
    const persona = selectedGalleryPersona();
    if (!persona) {
      showToast("Create a speaker first", "warning");
      return;
    }
    const imagePath = await pickImageFile();
    if (!imagePath) return;
    await setGalleryPhotoFromPath(imagePath);
  });
  el("#speaker-gallery-photo-delete").addEventListener("click", () => {
    const persona = selectedGalleryPersona();
    if (!persona) return;
    persona.photoPath = "";
    persona.updatedAt = Date.now();
    saveSpeakerPersonas();
  });
  el("#speaker-gallery-auto").addEventListener("click", async () => {
    const persona = selectedGalleryPersona();
    if (!persona?.refPath) {
      showToast("Drop a reference voice first", "warning");
      return;
    }
    const btn = el<HTMLButtonElement>("#speaker-gallery-auto");
    btn.disabled = true;
    btn.classList.add("transcribing");
    try {
      const text = await transcribeAudioText(persona.refPath);
      if (text !== null) {
        persona.refText = text;
        persona.updatedAt = Date.now();
        persistSpeakerPersonas();
        scheduleSpeakerPersonaFolderSync(persona);
        el<HTMLTextAreaElement>("#speaker-gallery-transcript").value = text;
      }
    } finally {
      btn.disabled = false;
      btn.classList.remove("transcribing");
    }
  });
  el<HTMLSelectElement>("#clone-persona-select").addEventListener("change", () => {
    const value = el<HTMLSelectElement>("#clone-persona-select").value;
    if (value) applyPersonaToClone(value);
  });
  el<HTMLSelectElement>("#finish-persona-select").addEventListener("change", () => {
    const value = el<HTMLSelectElement>("#finish-persona-select").value;
    if (value) applyPersonaToFinish(value);
  });
}

function createSpeaker(name?: string): MultiSpeaker {
  const count = multiSpeakers.length + 1;
  return {
    id: nextId("speaker"),
    personaId: "",
    name: name || `Speaker ${count}`,
    refPath: null,
    refName: "",
    refText: "",
    notes: "",
    photoPath: "",
    cachePath: "",
    normalize: false,
    open: true,
  };
}

function createLine(speakerId?: string): MultiLine {
  return {
    id: nextId("line"),
    speakerId: speakerId || multiSpeakers[0]?.id || "",
    text: "",
    overridePath: null,
    overrideName: "",
    overrideText: "",
    open: false,
  };
}

function resetMultiLines(): void {
  multiLines.splice(0, multiLines.length);
  multiLines.push(createLine(multiSpeakers[0]?.id));
  multiLines.push(createLine(multiSpeakers[1]?.id || multiSpeakers[0]?.id));
  renderMultiLines();
}

function ensureMultiDefaults(): void {
  while (multiSpeakers.length < 2) {
    multiSpeakers.push(createSpeaker());
  }
  while (multiLines.length < 2) {
    multiLines.push(createLine(multiSpeakers[multiLines.length]?.id || multiSpeakers[0]?.id));
  }
  const fallbackSpeaker = multiSpeakers[0]?.id || "";
  for (const line of multiLines) {
    if (!multiSpeakers.some((speaker) => speaker.id === line.speakerId)) {
      line.speakerId = fallbackSpeaker;
    }
  }
}

function findMultiSpeakerPreflightIssues(speakers: MultiSpeaker[], lines: MultiLine[]): string[] {
  const issues: string[] = [];
  const seen = new Set<string>();
  lines.forEach((line, index) => {
    const lineNo = index + 1;
    const text = line.text.trim();
    if (!text) {
      const message = `Line ${lineNo} needs text`;
      if (!seen.has(message)) {
        seen.add(message);
        issues.push(message);
      }
    }

    const speaker = speakers.find((item) => item.id === line.speakerId);
    if (!speaker) {
      const message = `Line ${lineNo} uses a missing speaker. Pick an existing speaker before generating.`;
      if (!seen.has(message)) {
        seen.add(message);
        issues.push(message);
      }
      return;
    }

    if (!line.overridePath && !speaker.refPath) {
      const speakerName = speaker.name.trim() || `Speaker ${speakers.indexOf(speaker) + 1}`;
      const message = `Line ${lineNo} uses "${speakerName}", but that speaker has no reference voice. Add a reference voice or choose a saved speaker identity.`;
      if (!seen.has(message)) {
        seen.add(message);
        issues.push(message);
      }
    }
  });
  return issues;
}

function assertMultiSpeakerReady(speakers: MultiSpeaker[], lines: MultiLine[]): void {
  const issues = findMultiSpeakerPreflightIssues(speakers, lines);
  if (issues.length > 0) {
    throw new Error(issues.slice(0, 4).join(" "));
  }
}

function multiDropzoneMarkup(fileName: string, emptyText: string): string {
  if (fileName) {
    return `
      <div class="dropzone-file">
        <span class="dropzone-file-icon">🎵</span>
        <span class="dropzone-file-name">${escapeHtml(fileName)}</span>
        <button class="dropzone-remove" type="button" data-action="clear-audio" aria-label="Remove audio">✕</button>
        <span class="dropzone-hint">Drop another audio file to replace it</span>
      </div>`;
  }
  return `
    <div class="dropzone-empty">
      <p>⤒ ${emptyText}</p>
      <p class="dropzone-hint">mp3 · wav · flac · m4a</p>
    </div>`;
}

function speakerOptions(selectedId: string): string {
  return multiSpeakers
    .map((speaker) => `<option value="${speaker.id}" ${speaker.id === selectedId ? "selected" : ""}>${escapeHtml(speaker.name)}</option>`)
    .join("");
}

function renderMultiSpeakers(): void {
  ensureMultiDefaults();
  const list = el<HTMLElement>("#speaker-library");
  list.innerHTML = "";
  for (const speaker of multiSpeakers) {
    const card = document.createElement("article");
    card.className = "speaker-card";
    card.dataset.speakerId = speaker.id;
    card.innerHTML = `
      <div class="speaker-card-head">
        ${avatarMarkup(speaker.name, speaker.photoPath)}
        <input class="text-input speaker-name-input" data-field="speaker-name" value="${escapeHtml(speaker.name)}" />
        <button class="compact-button" data-action="auto-speaker" type="button">✦ Auto-transcribe</button>
        <button class="icon-button speaker-toggle" data-action="toggle-speaker" type="button" aria-label="${speaker.open ? "Collapse speaker" : "Expand speaker"}">${speaker.open ? "▴" : "▾"}</button>
        <button class="compact-button" data-action="remove-speaker" type="button" ${multiSpeakers.length <= 2 ? "disabled" : ""}>−</button>
      </div>
      <div class="speaker-card-body ${speaker.open ? "" : "hidden"}">
        <div class="identity-select-row compact">
          <select class="select-input" data-field="speaker-persona">${personaOptions(speaker.personaId, "Use saved identity…")}</select>
        </div>
        <div class="dropzone mini-dropzone ${speaker.refName ? "has-file" : ""}" data-action="pick-speaker-audio">
          ${multiDropzoneMarkup(speaker.refName, "Drop reference voice, or click to browse")}
        </div>
        <label class="field-label transcript-label">Reference transcript</label>
        <textarea class="text-area" data-field="speaker-transcript" rows="2" placeholder="Optional. Auto-filled with Whisper when available.">${escapeHtml(speaker.refText)}</textarea>
        <label class="inline-toggle reference-normalize-toggle">
          <span>Normalize reference</span>
          <span class="toggle-switch"><input type="checkbox" data-field="speaker-normalize" ${speaker.normalize ? "checked" : ""} /><span class="toggle-slider"></span></span>
        </label>
        <label class="field-label transcript-label">Source / consent notes</label>
        <textarea class="text-area" data-field="speaker-notes" rows="2" placeholder="Optional notes for this identity">${escapeHtml(speaker.notes)}</textarea>
      </div>`;
    list.appendChild(card);
  }
  translateStaticDom(list);
}

function renderMultiLines(): void {
  ensureMultiDefaults();
  const list = el<HTMLElement>("#multi-lines");
  list.innerHTML = "";
  multiLines.forEach((line, index) => {
    const item = document.createElement("article");
    item.className = "dialogue-line";
    item.dataset.lineId = line.id;
    item.innerHTML = `
      <div class="line-main">
        <span class="line-grip" role="button" tabindex="0" aria-label="Drag line">⋮⋮</span>
        <span class="line-number">${index + 1}</span>
        <select class="select-input speaker-select" data-field="line-speaker">${speakerOptions(line.speakerId)}</select>
        <button class="compact-button" data-action="toggle-line-ref" type="button">${line.open ? "Hide reference" : "Reference"}</button>
        <button class="compact-button" data-action="remove-line" type="button" ${multiLines.length <= 2 ? "disabled" : ""}>−</button>
      </div>
      <textarea class="text-area line-text" data-field="line-text" rows="3" placeholder="Text to speak for this line...">${escapeHtml(line.text)}</textarea>
      <div class="line-reference ${line.open ? "" : "hidden"}">
        <div class="dropzone mini-dropzone ${line.overrideName ? "has-file" : ""}" data-action="pick-line-audio">
          ${multiDropzoneMarkup(line.overrideName, "Optional line-specific reference voice")}
        </div>
        <div class="label-row">
          <label class="field-label">Reference transcript override</label>
          <button class="link-button" data-action="auto-line" type="button">✦ Auto-transcribe</button>
        </div>
        <textarea class="text-area" data-field="line-transcript" rows="2" placeholder="Leave blank to use the selected speaker transcript.">${escapeHtml(line.overrideText)}</textarea>
      </div>`;
    list.appendChild(item);
  });
  translateStaticDom(list);
}

function renderMultiWorkflow(): void {
  renderMultiSpeakers();
  renderMultiLines();
}

function updateSpeakerTagsFromInput(): void {
  const names = el<HTMLInputElement>("#speaker-tags-input").value
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  while (names.length < 2) names.push(`Speaker ${names.length + 1}`);
  while (multiSpeakers.length < names.length) multiSpeakers.push(createSpeaker());
  while (multiSpeakers.length > Math.max(2, names.length)) {
    const removed = multiSpeakers.pop();
    if (removed) {
      for (const line of multiLines) {
        if (line.speakerId === removed.id) line.speakerId = multiSpeakers[0]?.id || "";
      }
    }
  }
  names.forEach((name, index) => {
    multiSpeakers[index].name = name;
  });
  renderMultiWorkflow();
}

function speakerFromElement(target: HTMLElement): MultiSpeaker | undefined {
  const card = target.closest<HTMLElement>("[data-speaker-id]");
  return card ? multiSpeakers.find((speaker) => speaker.id === card.dataset.speakerId) : undefined;
}

function lineFromElement(target: HTMLElement): MultiLine | undefined {
  const lineEl = target.closest<HTMLElement>("[data-line-id]");
  return lineEl ? multiLines.find((line) => line.id === lineEl.dataset.lineId) : undefined;
}

function clearLineDragStyles(lineList = el<HTMLElement>("#multi-lines")): void {
  for (const item of lineList.querySelectorAll(".dialogue-line")) {
    item.classList.remove("dragging", "drag-target");
  }
}

function lineElementFromPoint(clientX: number, clientY: number): HTMLElement | null {
  return (document.elementFromPoint(clientX, clientY) as HTMLElement | null)?.closest<HTMLElement>(".dialogue-line") ?? null;
}

function scrollMultiModeDuringDrag(clientY: number): void {
  const mode = el<HTMLElement>("#mode-multi");
  const rect = mode.getBoundingClientRect();
  const edge = 42;
  if (clientY < rect.top + edge) mode.scrollTop -= 18;
  else if (clientY > rect.bottom - edge) mode.scrollTop += 18;
}

async function setSpeakerAudioFromPath(speaker: MultiSpeaker, path: string, name?: string): Promise<void> {
  const prepared = await prepareReferenceAudioFile(path, name);
  speaker.personaId = "";
  speaker.refPath = prepared.path;
  speaker.refName = prepared.name;
  speaker.cachePath = "";
  renderMultiSpeakers();
}

async function setSpeakerAudioFromDrop(speaker: MultiSpeaker, files: FileList | null): Promise<void> {
  const file = files?.[0];
  if (!file) return;
  const path = (file as any).path as string | undefined;
  if (!path) {
    return;
  }
  await setSpeakerAudioFromPath(speaker, path, file.name);
}

function applyPersonaToMultiSpeaker(speaker: MultiSpeaker, personaId: string): void {
  const persona = findPersona(personaId);
  if (!persona) {
    showToast("Select a saved speaker identity first", "warning");
    return;
  }
  speaker.personaId = persona.id;
  speaker.name = persona.name;
  speaker.refPath = persona.refPath || null;
  speaker.refName = persona.refName || (persona.refPath ? persona.refPath.split(/[/\\]/).pop() || persona.name : "");
  speaker.refText = persona.refText;
  speaker.notes = persona.notes;
  speaker.photoPath = persona.photoPath;
  speaker.cachePath = persona.cachePath;
  speaker.normalize = persona.normalize;
  void ensureSpeakerCachePath(persona).then((path) => {
    speaker.cachePath = path;
  }).catch((e) => console.warn("Speaker cache path failed", e));
  renderMultiWorkflow();
  showToast(`Loaded speaker identity: ${persona.name}`);
}

async function savePersonaFromMultiSpeaker(speaker: MultiSpeaker): Promise<void> {
  if (!speaker.refPath) {
    showToast("Drop a reference voice before saving this identity", "warning");
    return;
  }
  const existing = findPersona(speaker.personaId);
  const persona: SpeakerPersona = {
    id: existing?.id || nextId("persona"),
    name: speaker.name.trim() || existing?.name || personaNameFromPath(speaker.refPath),
    refPath: speaker.refPath,
    refName: speaker.refName || speaker.refPath.split(/[/\\]/).pop() || speaker.name,
    refText: speaker.refText.trim(),
    notes: speaker.notes.trim(),
    photoPath: speaker.photoPath || existing?.photoPath || "",
    cachePath: existing?.cachePath || speaker.cachePath || "",
    normalize: speaker.normalize,
    createdAt: existing?.createdAt || Date.now(),
    updatedAt: Date.now(),
  };
  const storedAudio = await storeSpeakerAsset(persona, speaker.refPath, "audio");
  persona.refPath = storedAudio.path;
  persona.refName = speaker.refName || storedAudio.fileName;
  if (!persona.cachePath) await ensureSpeakerCachePath(persona);
  if (speaker.photoPath) {
    const storedPhoto = await storeSpeakerAsset(persona, speaker.photoPath, "image");
    persona.photoPath = storedPhoto.path;
  }
  upsertSpeakerPersona(persona);
  speaker.personaId = persona.id;
  speaker.refPath = persona.refPath;
  speaker.refName = persona.refName;
  speaker.photoPath = persona.photoPath;
  speaker.cachePath = persona.cachePath;
  renderMultiWorkflow();
  showToast(`Saved speaker identity: ${persona.name}`);
}

async function setLineAudioFromPath(line: MultiLine, path: string, name?: string): Promise<void> {
  const prepared = await prepareReferenceAudioFile(path, name);
  line.overridePath = prepared.path;
  line.overrideName = prepared.name;
  renderMultiLines();
}

async function setLineAudioFromDrop(line: MultiLine, files: FileList | null): Promise<void> {
  const file = files?.[0];
  if (!file) return;
  const path = (file as any).path as string | undefined;
  if (!path) {
    return;
  }
  await setLineAudioFromPath(line, path, file.name);
}

function applyTaggedScriptImport(): void {
  const script = el<HTMLTextAreaElement>("#multi-script-input").value;
  const rows = script.split(/\r?\n/);
  const parsed: { tag: string; text: string }[] = [];
  for (const raw of rows) {
    const line = raw.trim();
    if (!line) continue;
    const match = line.match(/^\[([^\]]+)\]\s*(.+)$/);
    if (match) {
      parsed.push({ tag: match[1].trim(), text: match[2].trim() });
    } else if (parsed.length > 0) {
      parsed[parsed.length - 1].text = `${parsed[parsed.length - 1].text}\n${line}`;
    }
  }
  if (parsed.length === 0) {
    showToast("Use lines like [Speaker1] text to import a script", "warning");
    return;
  }

  const unresolvedTags: string[] = [];
  for (const row of parsed) {
    let speaker = multiSpeakers.find((item) => item.name.toLowerCase() === row.tag.toLowerCase());
    if (!speaker) {
      speaker = createSpeaker(row.tag);
      const persona = speakerPersonas.find((item) => item.name.toLowerCase() === row.tag.toLowerCase());
      if (persona) {
        speaker.personaId = persona.id;
        speaker.refPath = persona.refPath || null;
        speaker.refName = persona.refName;
        speaker.refText = persona.refText;
        speaker.notes = persona.notes;
        speaker.photoPath = persona.photoPath;
        speaker.cachePath = persona.cachePath;
        speaker.normalize = persona.normalize;
        void ensureSpeakerCachePath(persona).then((path) => {
          if (speaker) speaker.cachePath = path;
        }).catch((e) => console.warn("Speaker cache path failed", e));
      } else if (!unresolvedTags.some((tag) => tag.toLowerCase() === row.tag.toLowerCase())) {
        unresolvedTags.push(row.tag);
      }
      multiSpeakers.push(speaker);
    }
  }
  while (multiSpeakers.length < 2) multiSpeakers.push(createSpeaker());

  multiLines.splice(0, multiLines.length);
  for (const row of parsed) {
    const speaker = multiSpeakers.find((item) => item.name.toLowerCase() === row.tag.toLowerCase()) || multiSpeakers[0];
    const line = createLine(speaker.id);
    line.text = row.text;
    multiLines.push(line);
  }
  while (multiLines.length < 2) multiLines.push(createLine(multiSpeakers[multiLines.length]?.id || multiSpeakers[0]?.id));
  renderMultiWorkflow();
  if (unresolvedTags.length > 0) {
    showToast(`Imported ${parsed.length} lines. Add reference voices for: ${unresolvedTags.join(", ")}`, "warning");
  } else {
    showToast(`Imported ${parsed.length} tagged script line${parsed.length === 1 ? "" : "s"}`);
  }
}

function reorderMultiLine(targetLineId: string, clientY: number): void {
  if (!draggedLineId || draggedLineId === targetLineId) return;
  const from = multiLines.findIndex((line) => line.id === draggedLineId);
  const to = multiLines.findIndex((line) => line.id === targetLineId);
  if (from < 0 || to < 0) return;
  const targetEl = document.querySelector<HTMLElement>(`[data-line-id="${targetLineId}"]`);
  const rect = targetEl?.getBoundingClientRect();
  const insertAfter = rect ? clientY > rect.top + rect.height / 2 : false;
  const [line] = multiLines.splice(from, 1);
  let nextIndex = to;
  if (from < to) nextIndex -= 1;
  if (insertAfter) nextIndex += 1;
  multiLines.splice(Math.max(0, Math.min(multiLines.length, nextIndex)), 0, line);
  renderMultiLines();
}

function moveDraggedLineToEnd(): void {
  if (!draggedLineId) return;
  const from = multiLines.findIndex((line) => line.id === draggedLineId);
  if (from < 0 || from === multiLines.length - 1) return;
  const [line] = multiLines.splice(from, 1);
  multiLines.push(line);
  renderMultiLines();
}

function initMultiSpeakerWorkflow(): void {
  ensureMultiDefaults();
  renderMultiWorkflow();

  el("#apply-speaker-tags").addEventListener("click", updateSpeakerTagsFromInput);
  el("#add-speaker-btn").addEventListener("click", () => {
    multiSpeakers.push(createSpeaker());
    renderMultiWorkflow();
  });
  el("#add-line-btn").addEventListener("click", () => {
    multiLines.push(createLine(multiSpeakers[0]?.id));
    renderMultiLines();
  });
  el("#clear-lines-btn").addEventListener("click", resetMultiLines);
  el("#multi-script-import-btn").addEventListener("click", applyTaggedScriptImport);

  const speakerList = el<HTMLElement>("#speaker-library");
  speakerList.addEventListener("input", (event) => {
    const target = event.target as HTMLElement;
    const speaker = speakerFromElement(target);
    if (!speaker) return;
    if ((target as HTMLInputElement).dataset.field === "speaker-name") {
      speaker.name = (target as HTMLInputElement).value || "Speaker";
      renderMultiLines();
    } else if ((target as HTMLTextAreaElement).dataset.field === "speaker-transcript") {
      speaker.refText = (target as HTMLTextAreaElement).value;
    } else if ((target as HTMLTextAreaElement).dataset.field === "speaker-notes") {
      speaker.notes = (target as HTMLTextAreaElement).value;
    }
  });
  speakerList.addEventListener("change", (event) => {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const speaker = speakerFromElement(target);
    if (speaker && target.dataset.field === "speaker-persona") {
      speaker.personaId = target.value;
      if (target.value) applyPersonaToMultiSpeaker(speaker, target.value);
    } else if (speaker && target.dataset.field === "speaker-normalize") {
      speaker.normalize = (target as HTMLInputElement).checked;
    }
  });
  speakerList.addEventListener("click", async (event) => {
    const target = event.target as HTMLElement;
    const action = target.dataset.action || target.closest<HTMLElement>("[data-action]")?.dataset.action;
    const speaker = speakerFromElement(target);
    if (!speaker || !action) return;
    if (action === "toggle-speaker") {
      speaker.open = !speaker.open;
      renderMultiSpeakers();
    } else if (action === "remove-speaker" && multiSpeakers.length > 2) {
      const idx = multiSpeakers.findIndex((item) => item.id === speaker.id);
      multiSpeakers.splice(idx, 1);
      for (const line of multiLines) {
        if (line.speakerId === speaker.id) line.speakerId = multiSpeakers[0]?.id || "";
      }
      renderMultiWorkflow();
    } else if (action === "pick-speaker-audio") {
      const file = await pickReferenceAudioFile();
      if (file) {
        speaker.refPath = file.path;
        speaker.refName = file.name;
        speaker.personaId = "";
        speaker.cachePath = "";
        renderMultiSpeakers();
      }
    } else if (action === "clear-audio") {
      speaker.refPath = null;
      speaker.refName = "";
      renderMultiSpeakers();
    } else if (action === "auto-speaker") {
      if (!speaker.refPath) {
        showToast("Drop a reference voice first", "warning");
        return;
      }
      target.classList.add("transcribing");
      const text = await transcribeAudioText(speaker.refPath);
      target.classList.remove("transcribing");
      if (text !== null) {
        speaker.refText = text;
        renderMultiSpeakers();
      }
    }
  });
  speakerList.addEventListener("dragover", (event) => {
    const dz = (event.target as HTMLElement).closest<HTMLElement>(".dropzone");
    if (!dz) return;
    event.preventDefault();
    dz.classList.add("drag-over");
  });
  speakerList.addEventListener("dragleave", (event) => {
    (event.target as HTMLElement).closest<HTMLElement>(".dropzone")?.classList.remove("drag-over");
  });
  speakerList.addEventListener("drop", async (event) => {
    const target = event.target as HTMLElement;
    const dz = target.closest<HTMLElement>(".dropzone");
    const speaker = speakerFromElement(target);
    if (!dz || !speaker) return;
    event.preventDefault();
    dz.classList.remove("drag-over");
    await setSpeakerAudioFromDrop(speaker, event.dataTransfer?.files || null);
  });

  const lineList = el<HTMLElement>("#multi-lines");
  lineList.addEventListener("input", (event) => {
    const target = event.target as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement;
    const line = lineFromElement(target);
    if (!line) return;
    if (target.dataset.field === "line-text") line.text = target.value;
    else if (target.dataset.field === "line-transcript") line.overrideText = target.value;
    else if (target.dataset.field === "line-speaker") line.speakerId = target.value;
  });
  lineList.addEventListener("change", (event) => {
    const target = event.target as HTMLSelectElement;
    const line = lineFromElement(target);
    if (line && target.dataset.field === "line-speaker") line.speakerId = target.value;
  });
  lineList.addEventListener("click", async (event) => {
    const target = event.target as HTMLElement;
    const action = target.dataset.action || target.closest<HTMLElement>("[data-action]")?.dataset.action;
    const line = lineFromElement(target);
    if (!line || !action) return;
    if (action === "toggle-line-ref") {
      line.open = !line.open;
      renderMultiLines();
    } else if (action === "remove-line" && multiLines.length > 2) {
      const idx = multiLines.findIndex((item) => item.id === line.id);
      multiLines.splice(idx, 1);
      renderMultiLines();
    } else if (action === "pick-line-audio") {
      const file = await pickReferenceAudioFile();
      if (file) {
        line.overridePath = file.path;
        line.overrideName = file.name;
        renderMultiLines();
      }
    } else if (action === "clear-audio") {
      line.overridePath = null;
      line.overrideName = "";
      renderMultiLines();
    } else if (action === "auto-line") {
      const path = line.overridePath || multiSpeakers.find((speaker) => speaker.id === line.speakerId)?.refPath;
      if (!path) {
        showToast("Drop a reference voice first", "warning");
        return;
      }
      target.classList.add("transcribing");
      const text = await transcribeAudioText(path);
      target.classList.remove("transcribing");
      if (text !== null) {
        line.overrideText = text;
        line.open = true;
        renderMultiLines();
      }
    }
  });
  lineList.addEventListener("pointerdown", (event) => {
    const grip = (event.target as HTMLElement).closest<HTMLElement>(".line-grip");
    const line = grip ? lineFromElement(grip) : undefined;
    if (!grip || !line || event.button !== 0) return;
    event.preventDefault();
    linePointerDrag = { id: line.id, pointerId: event.pointerId, grip, active: false };
    draggedLineId = line.id;
    grip.setPointerCapture(event.pointerId);
    grip.closest(".dialogue-line")?.classList.add("dragging");
  });
  const handleLinePointerMove = (event: PointerEvent) => {
    if (!linePointerDrag || linePointerDrag.pointerId !== event.pointerId) return;
    event.preventDefault();
    linePointerDrag.active = true;
    scrollMultiModeDuringDrag(event.clientY);
    const item = lineElementFromPoint(event.clientX, event.clientY);
    clearLineDragStyles(lineList);
    lineList.querySelector<HTMLElement>(`[data-line-id="${linePointerDrag.id}"]`)?.classList.add("dragging");
    if (item?.dataset.lineId && item.dataset.lineId !== linePointerDrag.id) item.classList.add("drag-target");
  };
  const finishLinePointerDrag = (event: PointerEvent) => {
    if (!linePointerDrag || linePointerDrag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const activeDrag = linePointerDrag;
    const item = lineElementFromPoint(event.clientX, event.clientY);
    const listRect = lineList.getBoundingClientRect();
    const inList =
      event.clientX >= listRect.left &&
      event.clientX <= listRect.right &&
      event.clientY >= listRect.top &&
      event.clientY <= listRect.bottom;
    if (activeDrag.active) {
      if (item?.dataset.lineId && item.dataset.lineId !== activeDrag.id) reorderMultiLine(item.dataset.lineId, event.clientY);
      else if (!item && inList) moveDraggedLineToEnd();
    }
    try {
      activeDrag.grip.releasePointerCapture(activeDrag.pointerId);
    } catch {
      // Pointer capture may already be released if the browser cancels the gesture.
    }
    linePointerDrag = null;
    draggedLineId = null;
    clearLineDragStyles(lineList);
  };
  window.addEventListener("pointermove", handleLinePointerMove);
  window.addEventListener("pointerup", finishLinePointerDrag);
  window.addEventListener("pointercancel", finishLinePointerDrag);
  lineList.addEventListener("dragstart", (event) => {
    const grip = (event.target as HTMLElement).closest<HTMLElement>(".line-grip");
    const line = grip ? lineFromElement(grip) : undefined;
    if (!line || !grip) {
      event.preventDefault();
      return;
    }
    draggedLineId = line.id;
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", line.id);
      event.dataTransfer.setDragImage(grip, 10, 10);
    }
    grip.closest(".dialogue-line")?.classList.add("dragging");
  });
  lineList.addEventListener("dragover", (event) => {
    if (!draggedLineId || (event.target as HTMLElement).closest(".line-reference .dropzone")) return;
    const item = (event.target as HTMLElement).closest<HTMLElement>(".dialogue-line");
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
    clearLineDragStyles(lineList);
    lineList.querySelector<HTMLElement>(`[data-line-id="${draggedLineId}"]`)?.classList.add("dragging");
    item?.classList.add("drag-target");
  });
  lineList.addEventListener("dragleave", (event) => {
    (event.target as HTMLElement).closest<HTMLElement>(".dialogue-line")?.classList.remove("drag-target");
  });
  lineList.addEventListener("drop", (event) => {
    if (!draggedLineId || (event.target as HTMLElement).closest(".line-reference .dropzone")) return;
    const item = (event.target as HTMLElement).closest<HTMLElement>(".dialogue-line");
    event.preventDefault();
    event.stopPropagation();
    if (item?.dataset.lineId) reorderMultiLine(item.dataset.lineId, event.clientY);
    else moveDraggedLineToEnd();
  });
  lineList.addEventListener("dragend", () => {
    draggedLineId = null;
    clearLineDragStyles(lineList);
  });
  lineList.addEventListener("dragover", (event) => {
    if (draggedLineId) return;
    const dz = (event.target as HTMLElement).closest<HTMLElement>(".line-reference .dropzone");
    if (!dz) return;
    event.preventDefault();
    dz.classList.add("drag-over");
  });
  lineList.addEventListener("drop", async (event) => {
    if (draggedLineId) return;
    const target = event.target as HTMLElement;
    const dz = target.closest<HTMLElement>(".line-reference .dropzone");
    const line = lineFromElement(target);
    if (!dz || !line) return;
    event.preventDefault();
    dz.classList.remove("drag-over");
    await setLineAudioFromDrop(line, event.dataTransfer?.files || null);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Advanced options
// ═══════════════════════════════════════════════════════════════════════════

function initAdvancedOptions(): void {
  const sliders: [string, string, (v: number) => string][] = [
    ["#opt-temperature", "#val-temperature", (v) => v.toFixed(2)],
    ["#opt-top-k", "#val-top-k", (v) => String(Math.round(v))],
    ["#opt-top-p", "#val-top-p", (v) => v.toFixed(2)],
    ["#opt-pause", "#val-pause", (v) => v.toFixed(2)],
    ["#opt-speaker-pause", "#val-speaker-pause", (v) => v.toFixed(2)],
  ];
  for (const [inputId, valueId, fmt] of sliders) {
    const input = el<HTMLInputElement>(inputId);
    const update = () => setText(valueId, fmt(parseFloat(input.value)));
    input.addEventListener("input", update);
    update();
  }

  // Longform chunking toggle — show/hide chunk size + pause fields
  const chunkToggle = el<HTMLInputElement>("#opt-chunk-enabled");
  const updateChunkFields = () => {
    const enabled = chunkToggle.checked;
    el<HTMLInputElement>("#opt-chunk-size").disabled = !enabled;
    el<HTMLInputElement>("#opt-pause").disabled = !enabled;
    el<HTMLElement>("#chunk-size-row").classList.toggle("disabled-row", !enabled);
    el<HTMLElement>("#pause-row").classList.toggle("disabled-row", !enabled);
    el<HTMLElement>("#val-pause").style.opacity = enabled ? "1" : "0.4";
  };
  chunkToggle.addEventListener("change", updateChunkFields);
  updateChunkFields();

  el("#seed-randomize").addEventListener("click", () => {
    el<HTMLInputElement>("#opt-seed").value = String(Math.floor(Math.random() * 2147483647));
  });
}

function gatherOptions(): Record<string, number | string | boolean> {
  const temp = parseFloat(el<HTMLInputElement>("#opt-temperature").value);
  const topK = parseInt(el<HTMLInputElement>("#opt-top-k").value, 10);
  const topP = parseFloat(el<HTMLInputElement>("#opt-top-p").value);
  let seed = parseInt(el<HTMLInputElement>("#opt-seed").value, 10);
  const seedMode = el<HTMLSelectElement>("#opt-seed-mode").value;
  const maxTokens = parseInt(el<HTMLInputElement>("#opt-max-tokens").value, 10);
  const chunkEnabled = el<HTMLInputElement>("#opt-chunk-enabled").checked;
  const chunkSize = parseInt(el<HTMLInputElement>("#opt-chunk-size").value, 10);
  const pause = parseFloat(el<HTMLInputElement>("#opt-pause").value);
  const speakerPause = parseFloat(el<HTMLInputElement>("#opt-speaker-pause").value);

  // Apply seed mode
  if (seedMode === "random") {
    seed = Math.floor(Math.random() * 2147483647);
    el<HTMLInputElement>("#opt-seed").value = String(seed);
  }

  const opts: Record<string, number | string | boolean> = {
    temperature: temp,
    top_k: topK,
    top_p: topP,
    seed,
    max_tokens: maxTokens,
    stream_playback: streamPlayback,
  };

  if (chunkEnabled) {
    opts.text_chunk_size = chunkSize;
    opts.pause_between_chunks = pause;
  }

  if (currentMode === "multi") {
    opts.pause_between_speakers = speakerPause;
  }

  return opts;
}

function deliveryControlPrefix(): string {
  const emotion = el<HTMLSelectElement>("#opt-emotion").value;
  const style = el<HTMLSelectElement>("#opt-style").value;
  const speed = el<HTMLSelectElement>("#opt-speed").value;
  const pitch = el<HTMLSelectElement>("#opt-pitch").value;
  const expressive = el<HTMLSelectElement>("#opt-expressive").value;
  const parts: string[] = [];
  if (emotion) parts.push(`<|emotion:${emotion}|>`);
  if (style) parts.push(`<|style:${style}|>`);
  for (const prosody of [speed, pitch, expressive]) {
    if (prosody) parts.push(`<|prosody:${prosody}|>`);
  }
  return parts.join("");
}

function applyDeliveryControls(text: string): string {
  return `${deliveryControlPrefix()}${text}`;
}

function advanceSeedAfterGeneration(): void {
  const seedMode = el<HTMLSelectElement>("#opt-seed-mode").value;
  const seedEl = el<HTMLInputElement>("#opt-seed");
  const cur = parseInt(seedEl.value, 10) || 0;
  if (seedMode === "increment") {
    seedEl.value = String(cur + 1);
  } else if (seedMode === "decrement") {
    seedEl.value = String(cur - 1);
  } else if (seedMode === "random") {
    seedEl.value = String(Math.floor(Math.random() * 2147483647));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Generate / Cancel
// ═══════════════════════════════════════════════════════════════════════════

function modeLabel(mode: GenerationMode): string {
  if (mode === "tts") return "TTS";
  if (mode === "clone") return "Voice Clone";
  if (mode === "finish") return "Continue Speech";
  return "Multi Speaker";
}

function updateGeneratingTabs(): void {
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode-tab")) {
    tab.classList.toggle("generating", tab.dataset.mode === activeGenerationJob?.mode);
  }
}

function updateGenerationControls(): void {
  const modeCanGenerate = isGenerationMode(currentMode);
  const isUtility = currentMode === "history" || currentMode === "api" || currentMode === "speakers";
  const generateBtn = el<HTMLButtonElement>("#generate-btn");
  const batchBtn = el<HTMLButtonElement>("#batch-btn");
  const queueBtn = el<HTMLButtonElement>("#queue-btn");
  const cancelBtn = el<HTMLButtonElement>("#cancel-btn");
  const modeSupportsBatch = currentMode === "tts" || currentMode === "clone";
  el<HTMLElement>("#action-row").classList.toggle("hidden", isUtility || !modeCanGenerate);
  generateBtn.classList.toggle("hidden", isGenerating || !modeCanGenerate);
  batchBtn.classList.toggle("hidden", isGenerating || !modeSupportsBatch);
  queueBtn.classList.toggle("hidden", !isGenerating || !modeCanGenerate);
  cancelBtn.classList.toggle("hidden", !isGenerating || activeGenerationJob?.mode !== currentMode);
  el<HTMLElement>("#progress-section").classList.toggle(
    "hidden",
    isUtility || !isGenerating || activeGenerationJob?.mode !== currentMode,
  );
  updateGeneratingTabs();
}

function isLiveStudioJob(job: StudioJobEvent): boolean {
  return job.status === "queued" || job.status === "generating";
}

function externalWorkflowLabel(workflow: string): string {
  if (workflow === "tts") return "TTS";
  if (workflow === "voice_clone") return "Voice Clone";
  if (workflow === "finish") return "Continue Speech";
  if (workflow === "multi") return "Multi Speaker";
  return workflow || "API";
}

function renderQueuePanel(): void {
  const panel = el<HTMLElement>("#queue-panel");
  const externalJobs = Array.from(externalStudioJobs.values()).filter(isLiveStudioJob);
  const externalActive = externalJobs.find((job) => job.status === "generating") || null;
  const externalQueuedCount = externalJobs.filter((job) => job.status === "queued").length;
  const waitingCount = generationQueue.length + externalQueuedCount;
  const hasItems = Boolean(activeGenerationJob) || generationQueue.length > 0 || externalJobs.length > 0;
  panel.classList.toggle("hidden", !hasItems);
  setText("#queue-status", activeGenerationJob || externalActive
    ? `${waitingCount} queued`
    : waitingCount > 0
      ? `${waitingCount} waiting`
      : "Idle");
  const active = el<HTMLElement>("#queue-active");
  if (activeGenerationJob) {
    active.textContent = `Generating: ${modeLabel(activeGenerationJob.mode)} - ${activeGenerationJob.label}`;
  } else if (externalActive) {
    active.textContent = `API: ${externalWorkflowLabel(externalActive.workflow)} - ${externalActive.label || externalActive.phase}`;
  } else {
    active.textContent = "Nothing generating";
  }
  const list = el<HTMLElement>("#queue-list");
  list.innerHTML = "";
  if (generationQueue.length === 0 && externalJobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = t("No queued generations");
    list.appendChild(empty);
  } else {
    generationQueue.forEach((job, index) => {
      const item = document.createElement("div");
      item.className = "queue-item";
      item.dataset.queueId = job.id;
      item.innerHTML = `
        <span class="queue-number">${index + 1}</span>
        <span class="queue-main">
          <strong>${escapeHtml(modeLabel(job.mode))}</strong>
          <span>${escapeHtml(job.label)}</span>
        </span>
        <span class="queue-actions">
          <button class="compact-button" data-queue-action="edit" type="button">Edit</button>
          <button class="compact-button danger" data-queue-action="remove" type="button">Delete</button>
        </span>
      `;
      list.appendChild(item);
    });
    externalJobs.forEach((job) => {
      const progress = job.total && job.total > 1 && typeof job.current === "number"
        ? `${job.current}/${job.total}`
        : job.phase || job.status;
      const item = document.createElement("div");
      item.className = `queue-item external ${job.status}`;
      item.innerHTML = `
        <span class="queue-number">API</span>
        <span class="queue-main">
          <strong>${escapeHtml(externalWorkflowLabel(job.workflow))}</strong>
          <span>${escapeHtml(job.label || job.message || "API request")}</span>
        </span>
        <span class="queue-actions">
          <span class="queue-pill">${escapeHtml(progress)}</span>
        </span>
      `;
      list.appendChild(item);
    });
  }
  translateStaticDom(list);
  el<HTMLButtonElement>("#queue-clear-btn").disabled = generationQueue.length === 0;
}

function jobLabel(job: GenerationJob): string {
  const payload = job.payload;
  if (payload.kind === "tts") return payload.text.slice(0, 48) || "Plain TTS";
  if (payload.kind === "clone") return payload.text.slice(0, 48) || payload.refName || "Voice clone";
  if (payload.kind === "finish") return payload.text.slice(0, 48) || payload.refName || "Continue speech";
  return payload.lines.map((line) => line.text.trim()).filter(Boolean).join(" / ").slice(0, 48) || "Multi speaker";
}

function captureCurrentGenerationJob(overrideText?: string): GenerationJob {
  if (!isGenerationMode(currentMode)) {
    throw new Error("This tab cannot generate audio");
  }
  const options = gatherOptions();
  const deliveryPrefix = deliveryControlPrefix();
  let payload: GenerationJob["payload"];
  if (currentMode === "tts") {
    const text = (overrideText ?? el<HTMLTextAreaElement>("#tts-text").value).trim();
    if (!text) throw new Error("Please enter text to speak");
    payload = { kind: "tts", text };
  } else if (currentMode === "clone") {
    const text = (overrideText ?? el<HTMLTextAreaElement>("#clone-text").value).trim();
    if (!text) throw new Error("Please enter text to speak");
    if (!cloneRefPath) throw new Error("Please provide a reference voice");
    payload = {
      kind: "clone",
      text,
      refPath: cloneRefPath,
      refName: cloneRefName,
      refText: el<HTMLTextAreaElement>("#clone-ref-text").value.trim() || undefined,
      normalize: el<HTMLInputElement>("#clone-normalize-ref").checked,
      personaId: clonePersonaId,
    };
  } else if (currentMode === "finish") {
    if (!finishRefPath) throw new Error("Please provide audio to continue");
    const text = el<HTMLTextAreaElement>("#finish-text").value.trim();
    if (!text) throw new Error("Continuation text is required");
    payload = {
      kind: "finish",
      text,
      refPath: finishRefPath,
      refName: finishRefName,
      transcript: el<HTMLTextAreaElement>("#finish-transcript").value.trim(),
      normalize: el<HTMLInputElement>("#finish-normalize-ref").checked,
      includeSource: el<HTMLInputElement>("#finish-include-source").checked,
      personaId: finishPersonaId,
    };
  } else {
    ensureMultiDefaults();
    assertMultiSpeakerReady(multiSpeakers, multiLines);
    payload = {
      kind: "multi",
      speakers: multiSpeakers.map(cloneMultiSpeaker),
      lines: multiLines.map(cloneMultiLine),
    };
  }
  const job: GenerationJob = {
    id: nextId("queue"),
    mode: currentMode,
    label: "",
    createdAt: Date.now(),
    options,
    deliveryPrefix,
    payload,
  };
  job.label = jobLabel(job);
  return job;
}

function enqueueCurrentGenerationJob(): void {
  try {
    const job = captureCurrentGenerationJob();
    generationQueue.push(job);
    renderQueuePanel();
    updateGenerationControls();
    showToast(`Queued: ${job.label}`);
  } catch (e) {
    showToast(String(e), "warning");
  }
}

function applyOptionsToUi(options: Record<string, number | string | boolean>): void {
  const setValue = (selector: string, value: unknown) => {
    const input = document.querySelector<HTMLInputElement>(selector);
    if (!input || value === undefined) return;
    input.value = String(value);
    input.dispatchEvent(new Event("input"));
  };
  setValue("#opt-temperature", options.temperature);
  setValue("#opt-top-k", options.top_k);
  setValue("#opt-top-p", options.top_p);
  setValue("#opt-seed", options.seed);
  setValue("#opt-max-tokens", options.max_tokens);
  const chunk = Boolean(options.text_chunk_size || options.pause_between_chunks);
  el<HTMLInputElement>("#opt-chunk-enabled").checked = chunk;
  el<HTMLInputElement>("#opt-chunk-enabled").dispatchEvent(new Event("change"));
  setValue("#opt-chunk-size", options.text_chunk_size);
  setValue("#opt-pause", options.pause_between_chunks);
  setValue("#opt-speaker-pause", options.pause_between_speakers);
}

function loadQueuedJobForEdit(job: GenerationJob): void {
  switchMode(job.mode);
  applyOptionsToUi(job.options);
  const payload = job.payload;
  if (payload.kind === "tts") {
    el<HTMLTextAreaElement>("#tts-text").value = payload.text;
  } else if (payload.kind === "clone") {
    el<HTMLTextAreaElement>("#clone-text").value = payload.text;
    cloneRefPath = payload.refPath;
    clonePersonaId = payload.personaId || "";
    cloneRefName = payload.refName;
    setDropzoneFile("#clone-dropzone", payload.refName || payload.refPath.split(/[/\\]/).pop() || "reference");
    showRefPreview("clone", payload.refPath);
    el<HTMLTextAreaElement>("#clone-ref-text").value = payload.refText || "";
    el<HTMLInputElement>("#clone-normalize-ref").checked = payload.normalize;
  } else if (payload.kind === "finish") {
    el<HTMLTextAreaElement>("#finish-text").value = payload.text;
    finishRefPath = payload.refPath;
    finishPersonaId = payload.personaId || "";
    finishRefName = payload.refName;
    setDropzoneFile("#finish-dropzone", payload.refName || payload.refPath.split(/[/\\]/).pop() || "reference");
    showRefPreview("finish", payload.refPath);
    el<HTMLTextAreaElement>("#finish-transcript").value = payload.transcript;
    el<HTMLInputElement>("#finish-normalize-ref").checked = payload.normalize;
    el<HTMLInputElement>("#finish-include-source").checked = payload.includeSource;
  } else {
    multiSpeakers.splice(0, multiSpeakers.length, ...payload.speakers.map(cloneMultiSpeaker));
    multiLines.splice(0, multiLines.length, ...payload.lines.map(cloneMultiLine));
    renderMultiWorkflow();
  }
  showToast("Loaded queued item for editing");
}

function removeQueuedJob(id: string): GenerationJob | null {
  const index = generationQueue.findIndex((job) => job.id === id);
  if (index < 0) return null;
  const [job] = generationQueue.splice(index, 1);
  renderQueuePanel();
  updateGenerationControls();
  return job;
}

function renderGenerationSteps(labels: string[], activeIndex = 0, completedThrough = -1): void {
  currentProgressLabels = labels;
  const steps = el<HTMLElement>("#gen-progress-steps");
  steps.innerHTML = "";
  labels.forEach((label, index) => {
    const item = document.createElement("span");
    item.className = "progress-step";
    if (index <= completedThrough) item.classList.add("complete");
    if (index === activeIndex) item.classList.add("active");
    item.textContent = label;
    steps.appendChild(item);
  });
}

function updateGenerationStep(activeIndex: number, completedThrough = activeIndex - 1): void {
  if (currentProgressLabels.length === 0) return;
  renderGenerationSteps(currentProgressLabels, activeIndex, completedThrough);
}

function generationStepForPhase(phase: string): number {
  const p = phase.toLowerCase();
  if (p.includes("output") || p.includes("save") || p.includes("complete") || p.includes("done")) return 4;
  if (p.includes("decode") || p.includes("audio") || p.includes("process")) return 3;
  if (p.includes("generate") || p.includes("sample") || p.includes("token")) return 2;
  if (p.includes("reference") || p.includes("voice") || p.includes("load")) return 1;
  return 0;
}

function generationElapsedLabel(prefix = "Elapsed"): string {
  const elapsed = genStartedAt > 0 ? ((performance.now() - genStartedAt) / 1000).toFixed(1) : "0.0";
  return `${prefix} ${elapsed}s`;
}

function setGenerationElapsedLabel(prefix = generationProgressPrefix): void {
  generationProgressPrefix = prefix;
  setText("#gen-progress-text", generationElapsedLabel(prefix));
}

function beginGeneration(job: GenerationJob): void {
  isGenerating = true;
  cancelRequested = false;
  activeGenerationJob = job;
  generationProgressPrefix = "Elapsed";
  genStartedAt = performance.now();
  el<HTMLElement>("#output-section").classList.toggle("hidden", !streamPlayback);
  if (streamPlayback) startLiveStreamPreview();
  if (genTimer) clearInterval(genTimer);
  genTimer = window.setInterval(() => {
    if (!isGenerating || !activeGenerationJob) return;
    setGenerationElapsedLabel(generationProgressPrefix);
  }, 350);
  const cancelBtn = el<HTMLButtonElement>("#cancel-btn");
  cancelBtn.disabled = false;
  cancelBtn.textContent = "⏹ Cancel";
  updateGenerationControls();
  if (!streamPlayback) el<HTMLElement>("#output-section").classList.add("hidden");
  const bar = el<HTMLElement>("#gen-progress-bar");
  bar.classList.add("indeterminate");
  bar.style.width = "";
  renderGenerationSteps([]);
  setGenerationElapsedLabel();
  renderQueuePanel();
}

function finishGeneration(success: boolean, message: string, tone?: "success" | "warning" | "error"): void {
  isGenerating = false;
  if (!success || !liveStream) stopLiveStreamPreview();
  if (genTimer) {
    clearInterval(genTimer);
    genTimer = null;
  }
  const cancelBtn = el<HTMLButtonElement>("#cancel-btn");
  cancelBtn.disabled = false;
  cancelBtn.textContent = "⏹ Cancel";
  const bar = el<HTMLElement>("#gen-progress-bar");
  bar.classList.remove("indeterminate");
  if (success) {
    setProgress("#gen-progress-bar", 1, 1);
    renderGenerationSteps([]);
    setGenerationElapsedLabel("Complete in");
  }
  el<HTMLElement>("#progress-section").classList.add("hidden");
  activeGenerationJob = null;
  updateGenerationControls();
  renderQueuePanel();
  if (message) showToast(message, tone || (success ? "success" : "error"));
}

async function resolveMultiLineReference(
  line: MultiLine,
  speakers: MultiSpeaker[],
  lines: MultiLine[],
): Promise<{ refPath: string; refText?: string; speakerName: string; normalize: boolean; cachePath?: string }> {
  const speaker = speakers.find((item) => item.id === line.speakerId);
  const refPath = line.overridePath || speaker?.refPath || "";
  if (!refPath) {
    throw new Error(`Line ${lines.indexOf(line) + 1} needs a reference voice`);
  }

  let refText = (line.overrideText || speaker?.refText || "").trim();
  if (!refText) {
    const text = await tryAutoTranscribeSilently(refPath);
    if (text) {
      refText = text;
      if (line.overridePath) line.overrideText = text;
      else if (speaker) speaker.refText = text;
    }
  }
  let cachePath = "";
  if (!line.overridePath && speaker?.personaId) {
    const persona = findPersona(speaker.personaId);
    if (persona) {
      cachePath = await ensureSpeakerCachePath(persona);
      speaker.cachePath = cachePath;
    } else {
      cachePath = speaker.cachePath || "";
    }
  } else if (!line.overridePath) {
    cachePath = speaker?.cachePath || "";
  }

  return {
    refPath,
    refText: refText || undefined,
    speakerName: speaker?.name || "Speaker",
    normalize: speaker?.normalize || false,
    cachePath: cachePath || undefined,
  };
}

async function generateMultiSpeakerJob(job: GenerationJob & { payload: { kind: "multi"; speakers: MultiSpeaker[]; lines: MultiLine[] } }): Promise<GenerationResult> {
  const options = job.options;
  const lines = job.payload.lines;
  const speakers = job.payload.speakers;
  const speakerPause = typeof options.pause_between_speakers === "number" ? options.pause_between_speakers : 0.15;
  const lineOptions = { ...options };
  delete lineOptions.pause_between_speakers;
  assertMultiSpeakerReady(speakers, lines);

  renderGenerationSteps([]);

  const outputs: GenerationResult[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    setProgress("#gen-progress-bar", i, lines.length);
    const resolved = await resolveMultiLineReference(line, speakers, lines);
    setGenerationElapsedLabel(`Line ${i + 1}/${lines.length} ·`);
    const perLineOptions: Record<string, number | string | boolean> = { ...lineOptions, normalize_reference: resolved.normalize };
    if (resolved.cachePath) perLineOptions.reference_cache_path = resolved.cachePath;
    const result = await invoke<GenerationResult>("generate_voice_clone", {
      request: {
        text: `${job.deliveryPrefix}${line.text.trim()}`,
        refAudioPath: resolved.refPath,
        refText: resolved.refText,
        options: perLineOptions,
      },
    });
    outputs.push(result);
    setProgress("#gen-progress-bar", i + 1, lines.length);
  }
  return concatenateWavResults(outputs, speakerPause);
}

async function generateJobAudio(job: GenerationJob): Promise<GenerationResult> {
  const payload = job.payload;
  if (payload.kind === "tts") {
    return invoke<GenerationResult>("generate_tts", {
      request: { text: `${job.deliveryPrefix}${payload.text}`, options: job.options },
    });
  }
  if (payload.kind === "clone") {
    const options: Record<string, number | string | boolean> = { ...job.options, normalize_reference: payload.normalize };
    if (payload.personaId) {
      const persona = findPersona(payload.personaId);
      if (persona) options.reference_cache_path = await ensureSpeakerCachePath(persona);
    }
    return invoke<GenerationResult>("generate_voice_clone", {
      request: {
        text: `${job.deliveryPrefix}${payload.text}`,
        refAudioPath: payload.refPath,
        refText: payload.refText,
        options,
      },
    });
  }
  if (payload.kind === "finish") {
    const options: Record<string, number | string | boolean> = {
      ...job.options,
      reference_text: payload.transcript,
      normalize_reference: payload.normalize,
    };
    if (payload.personaId) {
      const persona = findPersona(payload.personaId);
      if (persona) options.reference_cache_path = await ensureSpeakerCachePath(persona);
    }
    let result = await invoke<GenerationResult>("generate_finish_sentence", {
      request: {
        audioPath: payload.refPath,
        continuationText: `${job.deliveryPrefix}${payload.text}`,
        options,
      },
    });
    if (payload.includeSource) {
      setGenerationElapsedLabel();
      const source = await invoke<GenerationResult>("read_audio_as_wav", {
        audioPath: payload.refPath,
        targetSampleRate: result.sampleRate,
      });
      result = concatenateWavResults([source, result]);
    }
    return result;
  }
  return generateMultiSpeakerJob(job as GenerationJob & { payload: { kind: "multi"; speakers: MultiSpeaker[]; lines: MultiLine[] } });
}

async function runGenerationJob(job: GenerationJob): Promise<void> {
  let startNext = true;
  try {
    beginGeneration(job);
    const result = await generateJobAudio(job);
    if (cancelRequested) {
      cancelRequested = false;
      finishGeneration(false, "Generation cancelled", "warning");
      return;
    }
    setGenerationElapsedLabel();
    lastResult = result;
    showOutput(result, job.mode);
    setGenerationElapsedLabel();
    addHistory(job.mode, job.label || "Untitled", result);
    finishGeneration(true, "Generation complete");
    advanceSeedAfterGeneration();
  } catch (e) {
    const message = String(e);
    if (cancelRequested || message.toLowerCase().includes("cancel")) {
      cancelRequested = false;
      finishGeneration(false, "Generation cancelled", "warning");
    } else {
      startNext = false;
      finishGeneration(false, `Generation failed: ${e}`);
    }
  } finally {
    renderQueuePanel();
    updateGenerationControls();
    if (startNext && generationQueue.length > 0) {
      const next = generationQueue.shift()!;
      renderQueuePanel();
      void runGenerationJob(next);
    }
  }
}

async function doGenerate(): Promise<void> {
  if (isGenerating) {
    enqueueCurrentGenerationJob();
    return;
  }
  try {
    const job = captureCurrentGenerationJob();
    await runGenerationJob(job);
  } catch (e) {
    showToast(String(e), "warning");
  }
}

// Batch mode: split the current text field into lines and generate each line
// as a separate clip via the existing generation queue. Works in Text-to-Speech
// and Voice Clone modes (one voice/settings, many lines). Each clip lands in
// History with its own player and save button.
function batchTextForMode(): string | null {
  if (currentMode === "tts") return el<HTMLTextAreaElement>("#tts-text").value;
  if (currentMode === "clone") return el<HTMLTextAreaElement>("#clone-text").value;
  return null;
}

async function doBatchGenerate(): Promise<void> {
  const raw = batchTextForMode();
  if (raw === null) {
    showToast("Batch works in the Text to Speech and Voice Clone tabs", "warning");
    return;
  }
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) {
    showToast("Please enter text to speak (one clip per line)", "warning");
    return;
  }
  if (lines.length === 1) {
    await doGenerate();
    return;
  }
  let jobs: GenerationJob[];
  try {
    jobs = lines.map((line) => captureCurrentGenerationJob(line));
  } catch (e) {
    showToast(String(e), "warning");
    return;
  }
  if (isGenerating) {
    generationQueue.push(...jobs);
    renderQueuePanel();
    updateGenerationControls();
    showToast(`Batch queued: ${jobs.length} clips`);
    return;
  }
  const [first, ...rest] = jobs;
  generationQueue.push(...rest);
  renderQueuePanel();
  updateGenerationControls();
  showToast(`Batch: generating ${jobs.length} clips`);
  await runGenerationJob(first);
}

async function doCancel(): Promise<void> {
  if (!isGenerating || !activeGenerationJob) return;
  cancelRequested = true;
  const cancelBtn = el<HTMLButtonElement>("#cancel-btn");
  cancelBtn.disabled = true;
  cancelBtn.textContent = "Cancelling...";
  setGenerationElapsedLabel("Cancelling ·");
  try {
    await invoke("cancel_generation");
  } catch (e) {
    cancelRequested = false;
    showToast(`Cancel failed: ${e}`, "error");
    cancelBtn.disabled = false;
    cancelBtn.textContent = "⏹ Cancel";
  }
}

function initGenerate(): void {
  el("#generate-btn").addEventListener("click", doGenerate);
  el("#batch-btn").addEventListener("click", () => void doBatchGenerate());
  el("#queue-btn").addEventListener("click", enqueueCurrentGenerationJob);
  el("#cancel-btn").addEventListener("click", doCancel);
  el("#queue-clear-btn").addEventListener("click", () => {
    generationQueue.splice(0, generationQueue.length);
    renderQueuePanel();
    updateGenerationControls();
    showToast("Queue cleared");
  });
  el("#queue-list").addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const action = target.dataset.queueAction;
    const item = target.closest<HTMLElement>("[data-queue-id]");
    if (!action || !item?.dataset.queueId) return;
    const job = removeQueuedJob(item.dataset.queueId);
    if (!job) return;
    if (action === "edit") {
      loadQueuedJobForEdit(job);
    } else {
      showToast("Queued item deleted");
    }
  });
  el("#auto-transcribe-btn").addEventListener("click", () => doAutoTranscribe(cloneRefPath, "#clone-ref-text"));
  const finishTranscribe = document.querySelector<HTMLElement>("#finish-auto-transcribe-btn");
  if (finishTranscribe) finishTranscribe.addEventListener("click", () => doAutoTranscribe(finishRefPath, "#finish-transcript"));
  el("#history-clear-all").addEventListener("click", () => {
    history = [];
    renderHistory();
    showToast("History cleared");
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio output — waveform + playback
// ═══════════════════════════════════════════════════════════════════════════

function stopLiveStreamPreview(): void {
  if (!liveStream) return;
  liveStream.processor.onaudioprocess = null;
  try { liveStream.processor.disconnect(); } catch { /* already disconnected */ }
  if (liveStream.finalObjectUrl) URL.revokeObjectURL(liveStream.finalObjectUrl);
  void liveStream.context.close().catch(() => {});
  liveStream = null;
  el<HTMLElement>("#output-section").classList.remove("streaming");
  el<HTMLButtonElement>("#play-btn").textContent = "▶";
}

function startLiveStreamPreview(): void {
  stopLiveStreamPreview();
  const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextCtor) return;
  const context = new AudioContextCtor();
  const processor = context.createScriptProcessor(1024, 0, 2);
  processor.onaudioprocess = processLiveStreamAudio;
  processor.connect(context.destination);
  void context.resume().catch(() => {});
  lastResult = null;
  waveformBuildToken += 1;
  if (!audioPlayer.paused) audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  waveformSamples = new Float32Array(0);
  el<HTMLElement>("#output-section").classList.remove("hidden");
  el<HTMLElement>("#output-section").classList.add("streaming");
  el<HTMLButtonElement>("#play-btn").textContent = "⏸";
  setText("#output-time", "00:00 / 00:00");
  drawWaveform();
  liveStream = {
    context,
    processor,
    sampleRate: 0,
    channels: 0,
    receivedFrames: 0,
    playbackFrame: 0,
    playing: true,
    pcm: new Float32Array(0),
    finalized: false,
    finalObjectUrl: null,
    finalDuration: 0,
  };
  startWaveLoop();
}

function processLiveStreamAudio(event: AudioProcessingEvent): void {
  const stream = liveStream;
  const output = event.outputBuffer;
  const outputs = Array.from({ length: output.numberOfChannels }, (_, channel) => output.getChannelData(channel));
  for (const channel of outputs) channel.fill(0);
  if (!stream || !stream.playing || !stream.sampleRate || !stream.channels || stream.receivedFrames <= 0) return;

  const rateRatio = stream.sampleRate / stream.context.sampleRate;
  const frameCount = output.length;
  for (let outFrame = 0; outFrame < frameCount; outFrame++) {
    if (stream.playbackFrame >= stream.receivedFrames) {
      if (stream.finalized && stream.finalObjectUrl) {
        stream.playing = false;
        requestAnimationFrame(() => completeLiveStreamToFinalPlayer(true));
      }
      break;
    }

    const baseFrame = Math.floor(stream.playbackFrame);
    const frac = stream.playbackFrame - baseFrame;
    const nextFrame = Math.min(baseFrame + 1, stream.receivedFrames - 1);
    for (let outChannel = 0; outChannel < outputs.length; outChannel++) {
      const sourceChannel = stream.channels === 1 ? 0 : Math.min(outChannel, stream.channels - 1);
      const a = stream.pcm[baseFrame * stream.channels + sourceChannel] || 0;
      const b = stream.pcm[nextFrame * stream.channels + sourceChannel] || a;
      outputs[outChannel][outFrame] = (a + (b - a) * frac) * outputVolume;
    }
    stream.playbackFrame += rateRatio;
  }
}

function scheduleLiveAudioChunk(event: GenerationAudioChunkEvent): void {
  if (!isGenerating || !liveStream) return;
  try {
    const pcm = parseWavPcm(event.wavBase64);
    if (pcm.samples.length === 0) return;
    const frames = Math.floor(pcm.samples.length / pcm.channels);
    if (frames <= 0) return;
    if (!liveStream.sampleRate || !liveStream.channels) {
      liveStream.sampleRate = pcm.sampleRate;
      liveStream.channels = pcm.channels;
      setGenerationElapsedLabel();
    }
    if (liveStream.sampleRate !== pcm.sampleRate || liveStream.channels !== pcm.channels) return;
    const hintedStart = Number.isFinite(event.startSample) && event.startSample >= 0
      ? Math.floor(event.startSample / pcm.channels)
      : liveStream.receivedFrames;
    placeLivePcmChunk(pcm, Math.max(0, hintedStart));
    updateOutputTimeLabel();
    drawWaveform();
    startWaveLoop();
  } catch {
    // Streaming preview is best-effort; final output still arrives normally.
  }
}

function placeLivePcmChunk(pcm: WavPcm, startFrame: number): void {
  if (!liveStream) return;
  const frames = Math.floor(pcm.samples.length / pcm.channels);
  if (frames <= 0) return;
  const incoming = new Float32Array(frames * pcm.channels);
  for (let frame = 0; frame < frames; frame++) {
    for (let channel = 0; channel < pcm.channels; channel++) {
      incoming[frame * pcm.channels + channel] = pcm.samples[frame * pcm.channels + channel] / 32768;
    }
  }

  const fadeFrames = Math.min(96, frames);
  if (fadeFrames > 0) {
    for (let channel = 0; channel < pcm.channels; channel++) {
      const previous = startFrame > 0 && (startFrame - 1) < liveStream.receivedFrames
        ? liveStream.pcm[(startFrame - 1) * pcm.channels + channel] || 0
        : 0;
      for (let frame = 0; frame < fadeFrames; frame++) {
        const index = frame * pcm.channels + channel;
        const target = incoming[index];
        const weight = (frame + 1) / fadeFrames;
        incoming[index] = previous + (target - previous) * weight;
      }
    }
  }

  const neededSamples = (startFrame + frames) * pcm.channels;
  if (liveStream.pcm.length < neededSamples) {
    const next = new Float32Array(Math.max(neededSamples, Math.ceil(liveStream.pcm.length * 1.5) + 48000));
    next.set(liveStream.pcm);
    liveStream.pcm = next;
  }
  liveStream.pcm.set(incoming, startFrame * pcm.channels);
  liveStream.receivedFrames = Math.max(liveStream.receivedFrames, startFrame + frames);
  refreshLiveWaveformFromBuffer();
}

function refreshLiveWaveformFromBuffer(): void {
  if (!liveStream || !liveStream.channels || liveStream.receivedFrames <= 0) {
    waveformSamples = new Float32Array(0);
    return;
  }
  const mono = new Float32Array(liveStream.receivedFrames);
  for (let frame = 0; frame < liveStream.receivedFrames; frame++) {
    mono[frame] = liveStream.pcm[frame * liveStream.channels] || 0;
  }
  waveformSamples = mono;
}

function liveStreamDuration(): number {
  if (!liveStream || !liveStream.sampleRate) return 0;
  return liveStream.receivedFrames / liveStream.sampleRate;
}

function liveStreamCurrentTime(): number {
  if (!liveStream || !liveStream.sampleRate) return 0;
  return Math.min(liveStream.playbackFrame / liveStream.sampleRate, liveStreamFinalDuration());
}

function liveStreamFinalDuration(): number {
  if (!liveStream) return 0;
  return Math.max(liveStreamDuration(), liveStream.finalDuration);
}

function seekLiveStream(timeSeconds: number): void {
  if (!liveStream || !liveStream.sampleRate) return;
  const maxTime = liveStreamFinalDuration();
  const clamped = Math.max(0, Math.min(timeSeconds, maxTime));
  liveStream.playbackFrame = clamped * liveStream.sampleRate;
  updateOutputTimeLabel();
  drawWaveform();
}

function updateOutputTimeLabel(): void {
  if (liveStream) {
    setText("#output-time", `${formatTime(liveStreamCurrentTime())} / ${formatTime(liveStreamFinalDuration())}`);
    return;
  }
  setText("#output-time", `${formatTime(audioPlayer.currentTime)} / ${formatTime(audioPlayer.duration || 0)}`);
}

function completeLiveStreamToFinalPlayer(resumePlayback: boolean): void {
  if (!liveStream || !liveStream.finalObjectUrl) return;
  const finalUrl = liveStream.finalObjectUrl;
  const currentTime = liveStreamCurrentTime();
  liveStream.finalObjectUrl = null;
  liveStream.processor.onaudioprocess = null;
  try { liveStream.processor.disconnect(); } catch { /* already disconnected */ }
  void liveStream.context.close().catch(() => {});
  liveStream = null;
  el<HTMLElement>("#output-section").classList.remove("streaming");
  if (outputObjectUrl && outputObjectUrl !== finalUrl) URL.revokeObjectURL(outputObjectUrl);
  outputObjectUrl = finalUrl;
  audioPlayer.src = finalUrl;
  audioPlayer.load();
  const finalDuration = streamSafeDurationFromWaveform();
  el<HTMLButtonElement>("#play-btn").textContent = "▶";
  audioPlayer.onloadedmetadata = () => {
    if (audioPlayer.duration > 0) {
      audioPlayer.currentTime = Math.min(currentTime, audioPlayer.duration);
    } else if (finalDuration > 0) {
      audioPlayer.currentTime = Math.min(currentTime, finalDuration);
    }
    if (resumePlayback && audioPlayer.currentTime < (audioPlayer.duration || finalDuration) - 0.05) {
      void audioPlayer.play().then(() => {
        el<HTMLButtonElement>("#play-btn").textContent = "⏸";
        startWaveLoop();
      }).catch(() => {});
    }
    updateOutputTimeLabel();
    drawWaveform();
  };
  setText("#output-time", `${formatTime(Math.min(currentTime, finalDuration))} / ${formatTime(finalDuration)}`);
  drawWaveform();
}

function streamSafeDurationFromWaveform(): number {
  if (lastResult) {
    return lastResult.sampleCount / lastResult.sampleRate / lastResult.channels;
  }
  return waveformSamples ? waveformSamples.length / 48000 : 0;
}

function finalizeLiveStreamOutput(result: GenerationResult, mode: Mode, token: number): void {
  if (currentMode === mode && isGenerationMode(mode)) {
    el<HTMLElement>("#output-section").classList.remove("hidden");
  }
  const duration = result.sampleCount / result.sampleRate / result.channels;
  if (liveStream) {
    liveStream.finalized = true;
    liveStream.finalDuration = duration;
  }
  updateOutputTimeLabel();
  void base64ToBlobAsync(result.wavBase64, "audio/wav").then((blob) => {
    if (token !== waveformBuildToken) return;
    const url = URL.createObjectURL(blob);
    if (liveStream) {
      if (liveStream.finalObjectUrl) URL.revokeObjectURL(liveStream.finalObjectUrl);
      liveStream.finalObjectUrl = url;
      if (liveStream.finalized && !liveStream.playing) {
        completeLiveStreamToFinalPlayer(false);
      }
    } else {
      if (outputObjectUrl) URL.revokeObjectURL(outputObjectUrl);
      outputObjectUrl = url;
      audioPlayer.src = outputObjectUrl;
      audioPlayer.load();
    }
  }).catch(() => {
    if (token === waveformBuildToken) showToast("Could not prepare audio preview", "warning");
  });
  void drawWaveformFromBase64(result.wavBase64, token);
}

function showOutput(result: GenerationResult, mode: Mode = currentMode): void {
  const token = ++waveformBuildToken;
  // Store for this mode
  outputByMode[mode] = result;
  lastResult = result;

  if (liveStream) {
    finalizeLiveStreamOutput(result, mode, token);
    return;
  }

  // Stop any current playback
  if (!audioPlayer.paused) audioPlayer.pause();
  el<HTMLButtonElement>("#play-btn").textContent = "▶";

  audioPlayer.removeAttribute("src");
  void base64ToBlobAsync(result.wavBase64, "audio/wav").then((blob) => {
    if (token !== waveformBuildToken) return;
    if (outputObjectUrl) URL.revokeObjectURL(outputObjectUrl);
    outputObjectUrl = URL.createObjectURL(blob);
    audioPlayer.src = outputObjectUrl;
    audioPlayer.load();
  }).catch(() => {
    if (token === waveformBuildToken) showToast("Could not prepare audio preview", "warning");
  });

  if (currentMode === mode && isGenerationMode(mode)) {
    el<HTMLElement>("#output-section").classList.remove("hidden");
  }

  const duration = result.sampleCount / result.sampleRate / result.channels;
  setText("#output-time", `00:00 / ${formatTime(duration)}`);

  audioPlayer.onloadedmetadata = () => {
    updateOutputTimeLabel();
  };
  audioPlayer.onended = () => {
    el<HTMLButtonElement>("#play-btn").textContent = "▶";
    if (waveRAF) { cancelAnimationFrame(waveRAF); waveRAF = null; }
    drawWaveform();
  };

  waveformSamples = null;
  drawWaveform();
  void drawWaveformFromBase64(result.wavBase64, token);
}

let waveformSamples: Float32Array | null = null;
let waveRAF: number | null = null;
let waveformBuildToken = 0;
let outputObjectUrl: string | null = null;

function startWaveLoop(): void {
  const tick = () => {
    updateOutputTimeLabel();
    drawWaveform();
    if (liveStream) {
      if (liveStream.playing || !liveStream.finalized) {
        waveRAF = requestAnimationFrame(tick);
      } else {
        waveRAF = null;
      }
    } else if (!audioPlayer.paused && !audioPlayer.ended) {
      waveRAF = requestAnimationFrame(tick);
    } else {
      waveRAF = null;
    }
  };
  if (waveRAF) cancelAnimationFrame(waveRAF);
  waveRAF = requestAnimationFrame(tick);
}

async function drawWaveformFromBase64(base64: string, token = waveformBuildToken): Promise<void> {
  try {
    const buf = await base64ToBytesAsync(base64);
    if (token !== waveformBuildToken) return;
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const numChannels = dv.getUint16(22, true);
    const dataOffset = 44;
    const totalSamples = Math.floor((buf.length - dataOffset) / 2 / numChannels);
    const samples = new Float32Array(Math.min(totalSamples, 48000 * 120));
    for (let i = 0; i < samples.length; i++) {
      const offset = dataOffset + i * numChannels * 2;
      if (offset + 1 >= buf.length) break;
      samples[i] = dv.getInt16(offset, true) / 32768.0;
      if ((i & 0x3fff) === 0) {
        await nextFrame();
        if (token !== waveformBuildToken) return;
      }
    }
    waveformSamples = samples;
    drawWaveform();
  } catch {
    // ignore decode errors
  }
}

function drawWaveform(): void {
  const canvas = el<HTMLCanvasElement>("#waveform-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d")!;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.floor(width * dpr)) {
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = cssVar("--bg-inner", "#0d1013");
  ctx.fillRect(0, 0, width, height);
  if (!waveformSamples) return;

  const accent = cssVar("--accent", "#25b8ab");
  const muted = cssVar("--text-muted", "#9ea8b3");
  const barGap = 2;
  const barMinW = 3;
  const barCount = Math.max(60, Math.min(300, Math.floor(width / (barMinW + barGap))));
  const barW = (width - barCount * barGap) / barCount;
  const step = Math.max(1, Math.floor(waveformSamples.length / barCount));
  const mid = height / 2;

  const currentTime = liveStream ? liveStreamCurrentTime() : audioPlayer.currentTime;
  const duration = liveStream ? liveStreamFinalDuration() : audioPlayer.duration;
  const progress = duration > 0 ? currentTime / duration : 0;
  const playheadX = progress * width;

  for (let i = 0; i < barCount; i++) {
    let sum = 0;
    for (let j = 0; j < step; j++) {
      const idx = i * step + j;
      if (idx >= waveformSamples.length) break;
      sum += waveformSamples[idx] ** 2;
    }
    const rms = Math.sqrt(sum / step);
    const barH = Math.max(2, rms * height * 2.5);
    const x = i * (barW + barGap);
    ctx.fillStyle = x + barW / 2 <= playheadX ? accent : muted + "50";
    ctx.fillRect(x, mid - barH / 2, barW, barH);
  }

  // Playhead line
  if (progress > 0 && progress < 1) {
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();
  }
}

function setSaveFormat(format: SaveFormat): void {
  selectedSaveFormat = format === "mp3" ? "mp3" : "wav";
  localStorage.setItem("higgsAudio.saveFormat", selectedSaveFormat);
  for (const btn of document.querySelectorAll<HTMLButtonElement>(".save-format")) {
    const active = btn.dataset.format === selectedSaveFormat;
    btn.classList.toggle("active", active);
    btn.textContent = `${active ? "✓ " : ""}${(btn.dataset.format || "wav").toUpperCase()}`;
  }
}

function initAudioPlayer(): void {
  const playBtn = el<HTMLButtonElement>("#play-btn");

  const volumeSlider = el<HTMLInputElement>("#output-volume");
  const volumeIcon = document.querySelector<HTMLElement>("#output-volume-icon");
  const applyVolume = (value: number): void => {
    outputVolume = Math.min(1, Math.max(0, value));
    audioPlayer.volume = outputVolume;
    if (volumeIcon) {
      volumeIcon.textContent = outputVolume === 0 ? "🔇" : outputVolume < 0.5 ? "🔉" : "🔊";
    }
  };
  volumeSlider.value = String(Math.round(outputVolume * 100));
  applyVolume(outputVolume);
  volumeSlider.addEventListener("input", () => {
    applyVolume(Number(volumeSlider.value) / 100);
    localStorage.setItem(OUTPUT_VOLUME_STORAGE_KEY, String(outputVolume));
  });

  playBtn.addEventListener("click", async () => {
    if (liveStream) {
      if (liveStream.playing) {
        liveStream.playing = false;
        playBtn.textContent = "▶";
        if (waveRAF) { cancelAnimationFrame(waveRAF); waveRAF = null; }
        updateOutputTimeLabel();
        drawWaveform();
        if (liveStream.finalized && liveStream.finalObjectUrl) completeLiveStreamToFinalPlayer(false);
      } else {
        if (liveStream.finalized && liveStream.finalObjectUrl) {
          completeLiveStreamToFinalPlayer(true);
          return;
        }
        await liveStream.context.resume().catch(() => {});
        liveStream.playing = true;
        playBtn.textContent = "⏸";
        startWaveLoop();
      }
      return;
    }
    if (audioPlayer.paused) {
      if (audioPlayer.duration > 0 && audioPlayer.currentTime >= audioPlayer.duration - 0.05) {
        audioPlayer.currentTime = 0;
      }
      await audioPlayer.play().catch(() => {});
      playBtn.textContent = "⏸";
      startWaveLoop();
    } else {
      audioPlayer.pause();
      playBtn.textContent = "▶";
      if (waveRAF) { cancelAnimationFrame(waveRAF); waveRAF = null; }
      drawWaveform();
    }
  });

  // Click/drag-to-seek on waveform, including while streaming.
  const canvas = el<HTMLCanvasElement>("#waveform-canvas");
  canvas.style.cursor = "pointer";
  const seekFromPointer = (e: PointerEvent | MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    if (liveStream) {
      seekLiveStream(pct * liveStreamFinalDuration());
      if (liveStream.playing) startWaveLoop();
      return;
    }
    if (!audioPlayer.duration) return;
    audioPlayer.currentTime = pct * audioPlayer.duration;
    drawWaveform();
    if (!audioPlayer.paused) startWaveLoop();
  };
  let outputWaveDragging = false;
  canvas.addEventListener("pointerdown", (e) => {
    outputWaveDragging = true;
    canvas.setPointerCapture(e.pointerId);
    seekFromPointer(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (outputWaveDragging) seekFromPointer(e);
  });
  canvas.addEventListener("pointerup", (e) => {
    outputWaveDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  });
  canvas.addEventListener("pointercancel", () => {
    outputWaveDragging = false;
  });

  setSaveFormat(selectedSaveFormat);
  for (const btn of document.querySelectorAll<HTMLButtonElement>(".save-format")) {
    btn.addEventListener("click", () => setSaveFormat((btn.dataset.format as SaveFormat) || "wav"));
  }

  el("#download-output-btn").addEventListener("click", async () => {
    if (!lastResult) return;
    const format = selectedSaveFormat;
    try {
      const path = await save({
        defaultPath: `higgs_output_${Date.now()}.${format}`,
        filters: [
          format === "wav"
            ? { name: "WAV Audio", extensions: ["wav"] }
            : { name: "MP3 Audio", extensions: ["mp3"] },
        ],
      });
      if (path) {
        const base64Audio = format === "wav"
          ? lastResult.wavBase64
          : bytesToBase64(await encodeMp3FromWav(lastResult.wavBase64));
        await invoke("save_binary_file", { path, base64Data: base64Audio });
        showToast(`Saved ${format.toUpperCase()} file`);
      }
    } catch (e) {
      showToast(`Save failed: ${e}`, "error");
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Generation history
// ═══════════════════════════════════════════════════════════════════════════

function addHistory(mode: Mode, label: string, result: GenerationResult): void {
  const entry: HistoryEntry = {
    id: `gen_${Date.now()}`,
    mode,
    label,
    timestamp: Date.now(),
    wavBase64: result.wavBase64,
    sampleRate: result.sampleRate,
    channels: result.channels,
  };
  history.unshift(entry);
  if (history.length > 10) history.length = 10; // keep last 10
  renderHistory();
}

function renderHistory(): void {
  const list = el<HTMLElement>("#recent-list");
  if (history.length === 0) {
    list.innerHTML = `<p class="empty-state">${t("No generations yet")}</p>`;
    el<HTMLElement>("#history-clear-all").classList.add("hidden");
    return;
  }
  el<HTMLElement>("#history-clear-all").classList.remove("hidden");
  list.innerHTML = "";
  for (const entry of history) {
    const item = document.createElement("div");
    item.className = "recent-item";
    const time = new Date(entry.timestamp).toLocaleTimeString();
    item.innerHTML = `<span class="recent-mode">${entry.mode}</span><span class="recent-label">${entry.label}</span><span class="recent-time">${time}</span><button class="recent-delete" data-id="${entry.id}" title="Delete">✕</button>`;

    // Click on the item (not the delete button) plays it
    item.addEventListener("click", (e) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains("recent-delete")) return;
      // Stop current playback
      if (!audioPlayer.paused) audioPlayer.pause();
      el<HTMLButtonElement>("#play-btn").textContent = "▶";
      // Load this entry
      const fakeResult: GenerationResult = {
        sampleRate: entry.sampleRate,
        channels: entry.channels,
        sampleCount: 0,
        wavBase64: entry.wavBase64,
      };
      // Switch to the entry's mode so it shows in the right tab
      switchMode(entry.mode as Mode);
      showOutput(fakeResult);
    });

    list.appendChild(item);
  }

  // Wire delete buttons
  for (const btn of list.querySelectorAll<HTMLButtonElement>(".recent-delete")) {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      history = history.filter((h) => h.id !== id);
      renderHistory();
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Download flow
// ═══════════════════════════════════════════════════════════════════════════

function modelDownloadTarget(url: string): { destDir: string; filename: string | null } {
  try {
    const parsed = new URL(url);
    const match = parsed.pathname.match(/\/models\/([^/]+)\/([^/]+)$/);
    if (match) {
      return {
        destDir: `models/${decodeURIComponent(match[1])}`,
        filename: decodeURIComponent(match[2]),
      };
    }
  } catch {
    // The backend will report the invalid URL. Keep the fallback target simple.
  }
  return { destDir: "models", filename: null };
}

// Base URL (folder) of a file URL — used to fetch sibling config/tokenizer
// assets that live next to a pasted .gguf weight file.
function modelBaseUrl(url: string): string | null {
  const clean = url.split("?")[0];
  const idx = clean.lastIndexOf("/");
  return idx > 0 ? clean.slice(0, idx) : null;
}

function updateDownloadIndicator(status: DownloadProgressEvent["status"] = downloadPaused ? "paused" : "running"): void {
  const button = document.querySelector<HTMLButtonElement>("#download-status-button");
  if (button) {
    button.classList.toggle("hidden", !downloadActive);
    button.classList.toggle("downloading", downloadActive && status === "running");
    button.classList.toggle("paused", downloadActive && status === "paused");
    button.title = status === "paused" ? "Download paused. Click to reopen." : "Download running. Click to reopen.";
  }

  const fetchButton = document.querySelector<HTMLButtonElement>("#download-fetch-btn");
  const pauseButton = document.querySelector<HTMLButtonElement>("#download-pause-btn");
  const resumeButton = document.querySelector<HTMLButtonElement>("#download-resume-btn");
  const stopButton = document.querySelector<HTMLButtonElement>("#download-stop-btn");
  if (fetchButton) fetchButton.disabled = downloadActive;
  if (pauseButton) pauseButton.disabled = !downloadActive || status === "paused";
  if (resumeButton) resumeButton.disabled = !downloadActive || status !== "paused";
  if (stopButton) stopButton.disabled = !downloadActive;
}

function initDownload(): void {
  const trigger = el<HTMLButtonElement>("#download-trigger");
  const whisperTrigger = el<HTMLButtonElement>("#whisper-download-trigger");
  const engineTrigger = el<HTMLButtonElement>("#download-engine-btn");
  const statusButton = el<HTMLButtonElement>("#download-status-button");
  const popover = el<HTMLDivElement>("#download-popover");
  const urlInput = el<HTMLInputElement>("#download-url-input");
  const title = el<HTMLElement>("#download-title");
  const presetRow = el<HTMLElement>("#download-model-preset-row");
  const presetSelect = el<HTMLSelectElement>("#download-model-preset");
  const progressContainer = el<HTMLElement>("#download-progress-container");
  const fetchButton = el<HTMLButtonElement>("#download-fetch-btn");
  const pauseButton = el<HTMLButtonElement>("#download-pause-btn");
  const resumeButton = el<HTMLButtonElement>("#download-resume-btn");
  const stopButton = el<HTMLButtonElement>("#download-stop-btn");
  populateTtsDownloadPresetSelect();

  const setOpen = (open: boolean, kind: DownloadKind = activeDownloadKind) => {
    if (downloadActive) {
      kind = activeDownloadKind;
    }
    activeDownloadKind = kind;
    const whisperPreset = selectedWhisperPreset();
    title.textContent = kind === "whisper"
      ? t("Download Whisper Model")
      : kind === "engine"
        ? t("Download Engine DLLs")
        : t("Download Model");
    urlInput.placeholder = kind === "whisper"
      ? t("Paste whisper.cpp ggml .bin URL…")
      : kind === "engine"
        ? t("Engine package URL…")
        : t("Paste HuggingFace GGUF URL…");
    presetRow.classList.toggle("hidden", kind !== "model");
    if (kind === "whisper") {
      urlInput.value = whisperPresetUrl(whisperPreset);
      urlInput.title = `${whisperPreset.id} (${whisperPreset.size})`;
    } else if (kind === "engine") {
      urlInput.value = ENGINE_PACKAGE_URL;
      urlInput.title = "Downloads audiocpp_engine.dll plus required CUDA/MSVC runtime DLLs";
    } else {
      setTtsDownloadPreset(inferTtsPresetFromModelSelection(), !downloadActive);
    }
    popover.hidden = !open;
    updateDownloadIndicator();
  };

  const startDownload = async (kind: DownloadKind, url: string) => {
    if (downloadActive) {
      showToast("A download is already running. Use Pause, Resume, or Stop.", "warning");
      setOpen(true, activeDownloadKind);
      return;
    }
    activeDownloadKind = kind;
    downloadActive = true;
    downloadPaused = false;
    progressContainer.classList.remove("hidden");
    setProgress("#download-progress-bar", 0, 1);
    setText("#download-size-text", "0 / 0");
    setText("#download-speed-text", "0 MB/s");
    activeDownloadFileLabel = "";
    updateDownloadIndicator("running");

    try {
      if (kind === "engine") {
        const result = await invoke<{ path: string; size: number }>("download_engine_dll", { url });
        showToast(`Engine DLLs downloaded: ${result.path}`);
      } else if (kind === "whisper") {
        const result = await invoke<{ path: string; size: number }>("download_model", {
          request: { url, destDir: "models/whisper", filename: null },
        });
        localStorage.setItem("higgsAudio.whisperPreset", selectedWhisperPreset().id);
        setWhisperModelPath(result.path);
        showToast("Whisper model downloaded");
      } else {
        const selectedPreset = ttsPresetById(presetSelect.value);
        const preset = ttsPresetFromModelDownloadUrl(url) ||
          (url === ttsPresetUrl(selectedPreset) ? selectedPreset : null);
        if (preset) {
          const entries = ttsPresetPackageEntries(preset);
          for (let index = 0; index < entries.length; index += 1) {
            const entry = entries[index];
            activeDownloadFileLabel = `File ${index + 1}/${entries.length}: ${entry.filename}`;
            setText("#download-speed-text", `${activeDownloadFileLabel} · 0 MB/s`);
            await invoke<{ path: string; size: number }>("download_model", {
              request: { url: entry.url, destDir: entry.destDir, filename: entry.filename },
            });
          }
          showToast(`${preset.label} folder downloaded`);
        } else {
          const target = modelDownloadTarget(url);
          // Always fetch the WHOLE model folder, not just the .gguf. The engine
          // needs config.json + the tokenizer files next to the weights or it
          // fails to load with "missing config". A pasted GGUF URL that doesn't
          // match a built-in preset used to download only the .gguf — this is
          // the root cause of the "downloaded but won't load" bug.
          const base = modelBaseUrl(url);
          const looksLikeGguf = /\.gguf(?:$|\?)/i.test(url);
          const entries: Array<{ url: string; destDir: string; filename: string | null }> = [
            { url, destDir: target.destDir, filename: target.filename },
          ];
          if (base && looksLikeGguf && target.destDir.startsWith("models/")) {
            for (const asset of HIGGS_MODEL_ASSET_FILES) {
              entries.push({ url: `${base}/${asset}`, destDir: target.destDir, filename: asset });
            }
          }
          for (let index = 0; index < entries.length; index += 1) {
            const entry = entries[index];
            activeDownloadFileLabel = `File ${index + 1}/${entries.length}: ${entry.filename ?? ""}`;
            setText("#download-speed-text", `${activeDownloadFileLabel} · 0 MB/s`);
            try {
              await invoke<{ path: string; size: number }>("download_model", {
                request: { url: entry.url, destDir: entry.destDir, filename: entry.filename },
              });
            } catch (assetErr) {
              // The weights (index 0) are mandatory; a missing sidecar asset
              // (some repos omit one) should warn, not abort the whole folder.
              if (index === 0) throw assetErr;
              console.warn(`Optional model asset failed: ${entry.url}`, assetErr);
            }
          }
          showToast(entries.length > 1 ? "Model folder downloaded" : "Download complete");
        }
        await refreshModelList();
      }
    } catch (e) {
      const message = String(e);
      if (message.toLowerCase().includes("cancel")) {
        showToast("Download stopped", "warning");
      } else {
        showToast(`Download failed: ${e}`, "error");
      }
    } finally {
      downloadActive = false;
      downloadPaused = false;
      activeDownloadFileLabel = "";
      updateDownloadIndicator("complete");
    }
  };

  trigger.addEventListener("click", (e) => { e.stopPropagation(); setOpen(popover.hidden, "model"); });
  whisperTrigger.addEventListener("click", (e) => { e.stopPropagation(); setOpen(popover.hidden, "whisper"); });
  engineTrigger.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(true, "engine");
    void startDownload("engine", ENGINE_PACKAGE_URL);
  });
  statusButton.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(true, activeDownloadKind);
  });

  presetSelect.addEventListener("change", () => {
    setTtsDownloadPreset(ttsPresetById(presetSelect.value), true);
  });

  const dlClose = document.querySelector<HTMLElement>("#download-close");
  if (dlClose) dlClose.addEventListener("click", (e) => { e.stopPropagation(); setOpen(false); });
  document.addEventListener("pointerdown", (event) => {
    const target = event.target as Node;
    if (!popover.hidden && !popover.contains(target) && !trigger.contains(target) && !whisperTrigger.contains(target) && !engineTrigger.contains(target) && !statusButton.contains(target)) {
      setOpen(false);
    }
  });

  pauseButton.addEventListener("click", async () => {
    await invoke("download_control", { action: "pause" });
    downloadPaused = true;
    updateDownloadIndicator("paused");
  });
  resumeButton.addEventListener("click", async () => {
    await invoke("download_control", { action: "resume" });
    downloadPaused = false;
    updateDownloadIndicator("running");
  });
  stopButton.addEventListener("click", async () => {
    await invoke("download_control", { action: "cancel" });
    downloadPaused = false;
    updateDownloadIndicator("cancelled");
  });

  fetchButton.addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showToast(activeDownloadKind === "whisper" ? "Enter a whisper.cpp model URL" : "Enter a HuggingFace URL", "warning");
      return;
    }
    await startDownload(activeDownloadKind, url);
  });

  updateDownloadIndicator("complete");
}

// ═══════════════════════════════════════════════════════════════════════════
// Hardware monitor (port from SAM3DBody)
// ═══════════════════════════════════════════════════════════════════════════

const hardwareCanvas = document.querySelector<HTMLCanvasElement>("#hardware-graph")!;
const hardwareCtx = hardwareCanvas.getContext("2d")!;

function setMeter(barId: string, textId: string, current: number, total: number, text: string): void {
  setProgress(barId, current, total || current || 1);
  setText(textId, text);
}

function updateHardware(snapshot: HardwareSnapshot): void {
  hardwareHistory.push(snapshot);
  if (hardwareHistory.length > hardwareHistoryLimit) hardwareHistory.shift();

  setText("#hardware-detail", `${snapshot.gpuName} | ${snapshot.temperature || "−"} C | app RAM ${formatBytes(snapshot.processRam)}`);

  setMeter("#hw-vram-bar", "#hw-vram-text",
    snapshot.usedVram, snapshot.totalVram,
    `${formatBytes(snapshot.usedVram)} / ${formatBytes(snapshot.totalVram)}`);

  setMeter("#hw-gpu-bar", "#hw-gpu-text",
    snapshot.gpuUtilization, 100,
    `${snapshot.gpuUtilization || 0}%`);

  setMeter("#hw-power-bar", "#hw-power-text",
    snapshot.powerDraw, snapshot.powerLimit,
    snapshot.powerLimit ? `${snapshot.powerDraw.toFixed(0)} / ${snapshot.powerLimit.toFixed(0)} W` : "−");

  setMeter("#hw-ram-bar", "#hw-ram-text",
    snapshot.usedRam, snapshot.totalRam,
    `${formatBytes(snapshot.usedRam)} / ${formatBytes(snapshot.totalRam)}`);

  drawHardwareGraph();
}

function drawHardwareGraph(): void {
  const rect = hardwareCanvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (hardwareCanvas.width !== Math.floor(width * dpr)) {
    hardwareCanvas.width = Math.floor(width * dpr);
    hardwareCanvas.height = Math.floor(height * dpr);
  }
  hardwareCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  hardwareCtx.clearRect(0, 0, width, height);
  hardwareCtx.fillStyle = cssVar("--bg-inner", "#0d1013");
  hardwareCtx.fillRect(0, 0, width, height);

  const labelWidth = 38;
  const plotX = labelWidth;
  const plotWidth = Math.max(1, width - labelWidth - 4);

  // Grid lines + labels
  hardwareCtx.font = "10px Inter, sans-serif";
  hardwareCtx.fillStyle = cssVar("--text-muted", "#9ea8b3");
  hardwareCtx.textAlign = "right";
  hardwareCtx.textBaseline = "middle";
  for (const [pct, label] of [[1, "100%"], [0.5, "50%"], [0, "0%"]] as const) {
    const y = Math.round(height - pct * height) + 0.5;
    hardwareCtx.fillText(label, labelWidth - 7, Math.max(8, Math.min(height - 8, y)));
  }
  hardwareCtx.strokeStyle = cssVar("--border", "#1f262d");
  hardwareCtx.lineWidth = 1;
  for (const pct of [0, 0.25, 0.5, 0.75, 1]) {
    const y = Math.round(height * pct) + 0.5;
    hardwareCtx.beginPath();
    hardwareCtx.moveTo(plotX, y);
    hardwareCtx.lineTo(width, y);
    hardwareCtx.stroke();
  }

  // Determine visible window based on viewOffset
  const totalLen = hardwareHistory.length;
  const visibleStart = hardwareFollowLive
    ? Math.max(0, totalLen - hardwareGraphPoints)
    : Math.max(0, Math.min(totalLen - hardwareGraphPoints, totalLen - hardwareGraphPoints - hardwareViewOffset));
  const visibleEnd = Math.min(totalLen, visibleStart + hardwareGraphPoints);

  drawHardwareLine(plotX, plotWidth, height, "#e0a12b", (s) => s.totalVram ? s.usedVram / s.totalVram : 0, visibleStart, visibleEnd);
  drawHardwareLine(plotX, plotWidth, height, "#25b8ab", (s) => s.gpuUtilization / 100, visibleStart, visibleEnd);
  drawHardwareLine(plotX, plotWidth, height, "#c56cf0", (s) => s.powerLimit ? s.powerDraw / s.powerLimit : 0, visibleStart, visibleEnd);
  drawHardwareLine(plotX, plotWidth, height, "#6aa6ff", (s) => s.totalRam ? s.usedRam / s.totalRam : 0, visibleStart, visibleEnd);

  // Hover crosshair
  if (hardwareHover && !hardwareScrubDrag) {
    const hoverIdx = visibleStart + hardwareHover.idx;
    if (hoverIdx >= 0 && hoverIdx < totalLen) {
      const x = plotX + (hardwareHover.idx / Math.max(1, hardwareGraphPoints - 1)) * plotWidth;
      hardwareCtx.strokeStyle = cssVar("--text-muted", "#9ea8b3");
      hardwareCtx.lineWidth = 1;
      hardwareCtx.setLineDash([3, 3]);
      hardwareCtx.beginPath();
      hardwareCtx.moveTo(x, 0);
      hardwareCtx.lineTo(x, height);
      hardwareCtx.stroke();
      hardwareCtx.setLineDash([]);

      // Dots at each line's value for the hovered sample
      const snap = hardwareHistory[hoverIdx];
      const drawDot = (color: string, val: number) => {
        const y = height - Math.min(1, Math.max(0, val)) * height;
        hardwareCtx.fillStyle = color;
        hardwareCtx.beginPath();
        hardwareCtx.arc(x, y, 3, 0, Math.PI * 2);
        hardwareCtx.fill();
      };
      drawDot("#e0a12b", snap.totalVram ? snap.usedVram / snap.totalVram : 0);
      drawDot("#25b8ab", snap.gpuUtilization / 100);
      drawDot("#c56cf0", snap.powerLimit ? snap.powerDraw / snap.powerLimit : 0);
      drawDot("#6aa6ff", snap.totalRam ? snap.usedRam / snap.totalRam : 0);

      // Tooltip text in the detail line
      setText("#hardware-detail",
        `${formatBytes(snap.usedVram)} VRAM · ${snap.gpuUtilization.toFixed(0)}% GPU · ${snap.powerDraw.toFixed(0)}W` +
        (hardwareFollowLive ? "" : "  ◀ scrubbed"));
    }
  } else if (!hardwareFollowLive) {
    setText("#hardware-detail", `◀ Scrubbed view — drag to navigate, release at right edge to resume live`);
  }
}

function drawHardwareLine(
  start: number, width: number, height: number, color: string,
  valueFor: (sample: HardwareSnapshot) => number,
  visStart: number, visEnd: number,
): void {
  const count = visEnd - visStart;
  if (count < 2) return;
  const step = width / Math.max(1, hardwareGraphPoints - 1);
  hardwareCtx.beginPath();
  hardwareCtx.strokeStyle = color;
  hardwareCtx.lineWidth = 1.8;
  for (let i = 0; i < count; i++) {
    const sample = hardwareHistory[visStart + i];
    const value = Math.min(1, Math.max(0, valueFor(sample)));
    const x = start + step * i;
    const y = height - value * height;
    if (i === 0) hardwareCtx.moveTo(x, y);
    else hardwareCtx.lineTo(x, y);
  }
  hardwareCtx.stroke();
}

function scheduleGraphRedraw(): void {
  requestAnimationFrame(() => drawHardwareGraph());
}

function initHardwareScrubbing(): void {
  hardwareCanvas.style.cursor = "crosshair";

  hardwareCanvas.addEventListener("mousemove", (e) => {
    const rect = hardwareCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const plotX = 38;
    const plotWidth = Math.max(1, rect.width - plotX - 4);
    const slotIdx = Math.round(((x - plotX) / plotWidth) * (hardwareGraphPoints - 1));

    if (hardwareScrubDrag) {
      // Dragging — adjust viewOffset
      const dxPx = e.clientX - hardwareScrubDrag.startX;
      const ptsPerPx = hardwareGraphPoints / Math.max(1, plotWidth);
      const maxOffset = Math.max(0, hardwareHistory.length - hardwareGraphPoints);
      hardwareViewOffset = Math.max(0, Math.min(maxOffset,
        Math.round(hardwareScrubDrag.startOffset + dxPx * ptsPerPx)));
      scheduleGraphRedraw();
    } else {
      // Hovering — show crosshair
      const visStart = hardwareFollowLive
        ? Math.max(0, hardwareHistory.length - hardwareGraphPoints)
        : Math.max(0, Math.min(hardwareHistory.length - hardwareGraphPoints, hardwareHistory.length - hardwareGraphPoints - hardwareViewOffset));
      hardwareHover = { x, idx: Math.max(0, Math.min(hardwareGraphPoints - 1, slotIdx)) };
      scheduleGraphRedraw();
    }
  });

  hardwareCanvas.addEventListener("mouseleave", () => {
    if (!hardwareScrubDrag) {
      hardwareHover = null;
      scheduleGraphRedraw();
    }
  });

  hardwareCanvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    hardwareScrubDrag = { startX: e.clientX, startOffset: hardwareViewOffset };
    hardwareFollowLive = false;
    hardwareHover = null;
    hardwareCanvas.style.cursor = "grabbing";
  });

  document.addEventListener("mousemove", (e) => {
    if (!hardwareScrubDrag) return;
    const rect = hardwareCanvas.getBoundingClientRect();
    const plotWidth = Math.max(1, rect.width - 38 - 4);
    const dxPx = e.clientX - hardwareScrubDrag.startX;
    const ptsPerPx = hardwareGraphPoints / Math.max(1, plotWidth);
    const maxOffset = Math.max(0, hardwareHistory.length - hardwareGraphPoints);
    hardwareViewOffset = Math.max(0, Math.min(maxOffset,
      Math.round(hardwareScrubDrag.startOffset + dxPx * ptsPerPx)));
    scheduleGraphRedraw();
  });

  document.addEventListener("mouseup", () => {
    if (!hardwareScrubDrag) return;
    hardwareScrubDrag = null;
    hardwareCanvas.style.cursor = "crosshair";
    // Release near right edge → resume live
    if (hardwareViewOffset <= 1) {
      hardwareViewOffset = 0;
      hardwareFollowLive = true;
    }
    drawHardwareGraph();
  });
}

async function pollHardware(): Promise<void> {
  try {
    const snapshot = await invoke<HardwareSnapshot>("hardware_snapshot");
    updateHardware(snapshot);
  } catch {
    // ignore
  } finally {
    setTimeout(pollHardware, hardwarePollMs);
  }
}

function initHardwarePollRate(): void {
  const select = el<HTMLSelectElement>("#hardware-poll-rate");
  if (![250, 500, 1000, 1500].includes(hardwarePollMs)) hardwarePollMs = 1000;
  select.value = String(hardwarePollMs);
  select.addEventListener("change", () => {
    hardwarePollMs = parseInt(select.value, 10) || 1000;
    localStorage.setItem("higgsAudio.hardwarePollMs", String(hardwarePollMs));
  });
}

function initHardwareCollapse(): void {
  const panel = el<HTMLElement>("#hardware-panel");
  const toggle = el<HTMLButtonElement>("#hardware-toggle");
  toggle.addEventListener("click", () => {
    const collapsed = panel.classList.toggle("collapsed");
    toggle.textContent = collapsed ? "+" : "−";
    toggle.setAttribute("aria-label", collapsed ? "Expand hardware" : "Collapse hardware");
    drawHardwareGraph();
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Event listeners (Tauri events from backend)
// ═══════════════════════════════════════════════════════════════════════════

async function initEventListeners(): Promise<void> {
  await listen<ModelStatusEvent>("model-status", (event) => {
    const status = event.payload;
    engineSupportsStreaming = Boolean(status.supportsStreaming);
    setText("#engine-chip", status.engineLoaded ? t("Engine loaded") : t("Engine unloaded"));
    el<HTMLElement>("#engine-chip").classList.toggle("active", status.engineLoaded);

    if (!status.modelLoaded) {
      setText("#model-state", t("Not loaded"));
      el("#model-state").classList.remove("ok");
      setText("#model-chip", t("No model"));
      el("#model-chip").classList.add("muted");
      el("#model-chip").classList.remove("active");
      el<HTMLButtonElement>("#unload-model-btn").disabled = true;
      return;
    }

    if (status.displayName) {
      const uiName = selectedModelUiName();
      setText("#model-state", t("Loaded"));
      el("#model-state").classList.add("ok");
      setText("#model-chip", `${uiName} (${status.weightType || "default"})`);
      el("#model-chip").classList.remove("muted");
      el("#model-chip").classList.add("active");
      el<HTMLButtonElement>("#unload-model-btn").disabled = false;
    }
  });

  await listen<ProgressEvent>("generation-progress", (event) => {
    const p = event.payload;
    const bar = el<HTMLElement>("#gen-progress-bar");
    const phase = p.phase.toLowerCase();
    const step = generationStepForPhase(phase);
    updateGenerationStep(step, step - 1);
    if (p.total > 1) {
      bar.classList.remove("indeterminate");
      setProgress("#gen-progress-bar", p.current, p.total);
    } else {
      bar.classList.add("indeterminate");
    }
    setGenerationElapsedLabel();
  });

  await listen<GenerationAudioChunkEvent>("generation-audio-chunk", (event) => {
    scheduleLiveAudioChunk(event.payload);
  });

  await listen<StudioJobEvent>("studio-job", (event) => {
    const job = event.payload;
    externalStudioJobs.set(job.id, job);
    renderQueuePanel();
    if (!isLiveStudioJob(job)) {
      window.setTimeout(() => {
        const latest = externalStudioJobs.get(job.id);
        if (latest && latest.status === job.status) {
          externalStudioJobs.delete(job.id);
          renderQueuePanel();
        }
      }, 5000);
    }
  });

  await listen<DownloadProgressEvent>("download-progress", (event) => {
    const p = event.payload;
    setProgress("#download-progress-bar", p.downloaded, p.total);
    setText("#download-size-text", `${formatBytes(p.downloaded)} / ${formatBytes(p.total)}`);
    downloadPaused = p.status === "paused";
    if (p.status === "paused") {
      setText("#download-speed-text", activeDownloadFileLabel ? `${activeDownloadFileLabel} · Paused` : "Paused");
    } else if (p.status === "cancelled") {
      setText("#download-speed-text", "Stopped");
    } else if (p.status === "complete") {
      setText("#download-speed-text", activeDownloadFileLabel ? `${activeDownloadFileLabel} · Complete` : "Complete");
    } else {
      const speed = `${p.speedMbps.toFixed(1)} MB/s`;
      setText("#download-speed-text", activeDownloadFileLabel ? `${activeDownloadFileLabel} · ${speed}` : speed);
    }
    updateDownloadIndicator(p.status);
  });

  await listen<ApiLogEvent>("api-log", (event) => {
    appendApiLog(event.payload);
    if (!apiRunning && event.payload.kind === "server" && event.payload.method === "START") {
      apiRunning = true;
      setApiVisualStatus("running", "Running");
    } else if (event.payload.kind === "server" && event.payload.method === "STOP") {
      apiRunning = false;
      setApiVisualStatus("stopped", "Stopped");
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Text input char counting
// ═══════════════════════════════════════════════════════════════════════════

function initTextCounting(): void {
  const ttsText = el<HTMLTextAreaElement>("#tts-text");
  const ttsCount = el<HTMLElement>("#tts-count");
  ttsText.addEventListener("input", () => {
    ttsCount.textContent = String(ttsText.value.length);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  clearApiLogsForFreshSession();
  initTooltips();
  initSettings();
  initExternalLinks();
  initEngineDiagnosticModal();
  initWhisperPanel();
  initModeTabs();
  initApiPanel();
  initModelPanel();
  initDropzones();
  initSpeakerPersonas();
  initMultiSpeakerWorkflow();
  initTauriDropzones();
  initAdvancedOptions();
  initGenerate();
  initAudioPlayer();
  initDownload();
  initSetupWizard();
  initTextCounting();
  initHardwarePollRate();
  initHardwareCollapse();
  initHardwareScrubbing();
  await initEventListeners();

  renderHistory();
  pollHardware();

  // Auto-load engine on startup if bundled
  const bundled = await invoke<string | null>("bundled_engine_path");
  if (bundled) {
    await doLoadEngine();
  }
  window.setTimeout(() => void maybeShowSetupWizard(), 500);

  window.addEventListener("resize", () => {
    drawHardwareGraph();
    drawWaveform();
    redrawAllRefPreviews();
  });
}

document.addEventListener("contextmenu", (e) => e.preventDefault());

// Localize the static markup before any init runs (default language: Russian).
document.documentElement.lang = getLang();
translateStaticDom();

if (new URLSearchParams(window.location.search).get("popout") === "api-command-centre") {
  void initApiCommandPopout();
} else {
  main();
}
