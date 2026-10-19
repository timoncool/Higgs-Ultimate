import { invoke } from "@tauri-apps/api/core";

export type ReferenceTrimContext =
  | "clone"
  | "finish"
  | "gallery"
  | `speaker:${string}`
  | `line:${string}`;

export type ReferenceTrimSource = {
  path: string;
  name: string;
};

export type ReferenceTrimResult = {
  path: string;
  durationSeconds: number;
  cropped: boolean;
};

type TrimmerOptions = {
  getReference: (context: ReferenceTrimContext) => ReferenceTrimSource | null;
  applyTrim: (context: ReferenceTrimContext, result: ReferenceTrimResult) => void | Promise<void>;
  showToast: (message: string, tone?: "success" | "warning" | "error") => void;
  maxSeconds: number;
};

type AudioPreview = {
  wavBase64: string;
  sampleRate: number;
  channels: number;
  sampleCount: number;
};

let options: TrimmerOptions | null = null;
let activeContext: ReferenceTrimContext | null = null;
let activeSource: ReferenceTrimSource | null = null;
let durationSeconds = 0;
let previewUrl: string | null = null;
let waveformPeaks: number[] = [];
let selectionPlayback = false;

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing reference trimmer element: ${id}`);
  return element as T;
}

function formatTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

async function base64ToBlob(base64: string, mime: string): Promise<Blob> {
  const response = await fetch(`data:${mime};base64,${base64}`);
  return response.blob();
}

function startValue(): number {
  return Number(byId<HTMLInputElement>("reference-trim-start").value) || 0;
}

function endValue(): number {
  return Number(byId<HTMLInputElement>("reference-trim-end").value) || 0;
}

function setSelection(start: number, end: number, changed: "start" | "end" | "both" = "both"): void {
  if (!options) return;
  const minimum = Math.min(0.1, durationSeconds);
  let nextStart = Math.max(0, Math.min(start, durationSeconds));
  let nextEnd = Math.max(0, Math.min(end, durationSeconds));

  if (nextEnd - nextStart < minimum) {
    if (changed === "start") nextStart = Math.max(0, nextEnd - minimum);
    else nextEnd = Math.min(durationSeconds, nextStart + minimum);
  }
  if (nextEnd - nextStart > options.maxSeconds) {
    if (changed === "start") nextEnd = Math.min(durationSeconds, nextStart + options.maxSeconds);
    else nextStart = Math.max(0, nextEnd - options.maxSeconds);
  }

  byId<HTMLInputElement>("reference-trim-start").value = nextStart.toFixed(2);
  byId<HTMLInputElement>("reference-trim-end").value = nextEnd.toFixed(2);
  byId<HTMLInputElement>("reference-trim-start-range").value = String(nextStart);
  byId<HTMLInputElement>("reference-trim-end-range").value = String(nextEnd);
  byId("reference-trim-selection").textContent = `${formatTime(nextStart)} to ${formatTime(nextEnd)} · ${(nextEnd - nextStart).toFixed(2)}s selected`;
  drawWaveform();
}

function drawWaveform(): void {
  const canvas = byId<HTMLCanvasElement>("reference-trim-waveform");
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * window.devicePixelRatio));
  const height = Math.max(1, Math.round(rect.height * window.devicePixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  const styles = getComputedStyle(document.documentElement);
  const background = styles.getPropertyValue("--bg-inner").trim() || "#111827";
  const waveform = styles.getPropertyValue("--accent").trim() || "#14b8a6";
  const muted = styles.getPropertyValue("--text-muted").trim() || "#64748b";
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);

  if (waveformPeaks.length > 0) {
    const middle = height / 2;
    context.strokeStyle = waveform;
    context.lineWidth = Math.max(1, window.devicePixelRatio);
    context.beginPath();
    for (let x = 0; x < width; x += Math.max(1, Math.round(window.devicePixelRatio))) {
      const index = Math.min(waveformPeaks.length - 1, Math.floor(x * waveformPeaks.length / width));
      const amplitude = Math.max(1, waveformPeaks[index] * height * 0.45);
      context.moveTo(x + 0.5, middle - amplitude);
      context.lineTo(x + 0.5, middle + amplitude);
    }
    context.stroke();
  }

  if (durationSeconds > 0) {
    const startX = startValue() / durationSeconds * width;
    const endX = endValue() / durationSeconds * width;
    context.fillStyle = `${muted}66`;
    context.fillRect(0, 0, startX, height);
    context.fillRect(endX, 0, Math.max(0, width - endX), height);
    context.strokeStyle = waveform;
    context.lineWidth = Math.max(2, 2 * window.devicePixelRatio);
    context.strokeRect(startX, 1, Math.max(1, endX - startX), height - 2);
  }
}

function closeReferenceTrimmer(): void {
  const modal = document.getElementById("reference-trim-modal");
  modal?.classList.add("hidden");
  const audio = document.getElementById("reference-trim-audio") as HTMLAudioElement | null;
  audio?.pause();
  if (audio) audio.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  activeContext = null;
  activeSource = null;
  waveformPeaks = [];
  selectionPlayback = false;
}

async function applyCurrentTrim(): Promise<void> {
  if (!options || !activeContext || !activeSource) return;
  const applyButton = byId<HTMLButtonElement>("reference-trim-apply");
  applyButton.disabled = true;
  applyButton.textContent = "Applying…";
  try {
    const result = await invoke<ReferenceTrimResult>("trim_reference_audio", {
      audioPath: activeSource.path,
      startSeconds: startValue(),
      endSeconds: endValue(),
      maxSeconds: options.maxSeconds,
    });
    await options.applyTrim(activeContext, result);
    options.showToast(`Reference trimmed to ${result.durationSeconds.toFixed(2)} seconds`);
    closeReferenceTrimmer();
  } catch (error) {
    options.showToast(`Could not trim reference: ${error}`, "error");
  } finally {
    applyButton.disabled = false;
    applyButton.textContent = "Apply trim";
  }
}

async function loadReference(context: ReferenceTrimContext): Promise<void> {
  if (!options) return;
  const source = options.getReference(context);
  if (!source?.path) {
    options.showToast("Add a reference voice before trimming", "warning");
    return;
  }

  activeContext = context;
  activeSource = source;
  const modal = byId("reference-trim-modal");
  modal.classList.remove("hidden");
  byId("reference-trim-file-name").textContent = source.name || source.path.split(/[/\\]/).pop() || "Reference audio";
  byId("reference-trim-selection").textContent = "Loading audio…";
  byId<HTMLButtonElement>("reference-trim-apply").disabled = true;

  try {
    const [preview, waveform] = await Promise.all([
      invoke<AudioPreview>("read_audio_as_wav", { audioPath: source.path, targetSampleRate: null }),
      invoke<{ peaks: number[] }>("audio_waveform", { audioPath: source.path, points: 1400 }),
    ]);
    if (activeContext !== context) return;
    durationSeconds = preview.sampleCount / Math.max(1, preview.sampleRate * preview.channels);
    waveformPeaks = waveform.peaks || [];
    const audio = byId<HTMLAudioElement>("reference-trim-audio");
    const blob = await base64ToBlob(preview.wavBase64, "audio/wav");
    if (activeContext !== context) return;
    previewUrl = URL.createObjectURL(blob);
    audio.src = previewUrl;
    audio.load();
    for (const id of ["reference-trim-start-range", "reference-trim-end-range"]) {
      byId<HTMLInputElement>(id).max = String(durationSeconds);
    }
    byId<HTMLInputElement>("reference-trim-start").max = String(durationSeconds);
    byId<HTMLInputElement>("reference-trim-end").max = String(durationSeconds);
    setSelection(0, Math.min(durationSeconds, options.maxSeconds));
    byId<HTMLButtonElement>("reference-trim-apply").disabled = false;
  } catch (error) {
    options.showToast(`Could not open reference editor: ${error}`, "error");
    closeReferenceTrimmer();
  }
}

function renderTrimmer(): void {
  document.body.insertAdjacentHTML("beforeend", `
    <div id="reference-trim-modal" class="modal-backdrop hidden" role="dialog" aria-modal="true" aria-labelledby="reference-trim-title">
      <section class="modal-panel reference-trim-panel">
        <div class="modal-head">
          <div>
            <h3 id="reference-trim-title">Trim reference audio</h3>
            <span id="reference-trim-file-name" class="api-status-text"></span>
          </div>
          <button id="reference-trim-close" class="popover-close" type="button" aria-label="Close">✕</button>
        </div>
        <audio id="reference-trim-audio" class="reference-trim-audio" controls preload="metadata"></audio>
        <canvas id="reference-trim-waveform" class="reference-trim-waveform"></canvas>
        <div class="reference-trim-ranges">
          <label>
            <span>Start</span>
            <input id="reference-trim-start-range" type="range" min="0" max="30" step="0.01" value="0" />
            <input id="reference-trim-start" class="text-input" type="number" min="0" max="30" step="0.01" value="0" />
          </label>
          <label>
            <span>End</span>
            <input id="reference-trim-end-range" type="range" min="0" max="30" step="0.01" value="30" />
            <input id="reference-trim-end" class="text-input" type="number" min="0" max="30" step="0.01" value="30" />
          </label>
        </div>
        <div class="reference-trim-tools">
          <button id="reference-trim-set-start" class="compact-button" type="button">Set start to playhead</button>
          <button id="reference-trim-preview" class="compact-button" type="button">Play selection</button>
          <button id="reference-trim-set-end" class="compact-button" type="button">Set end to playhead</button>
        </div>
        <p id="reference-trim-selection" class="reference-trim-selection">0.00s selected</p>
        <p class="input-hint">The saved trim is the exact reference sent to Higgs and Whisper. Selections must be 30 seconds or shorter.</p>
        <div class="modal-actions">
          <button id="reference-trim-cancel" class="compact-button" type="button">Cancel</button>
          <button id="reference-trim-apply" class="compact-button primary" type="button">Apply trim</button>
        </div>
      </section>
    </div>`);
}

export function initReferenceTrimmer(value: TrimmerOptions): void {
  options = value;
  renderTrimmer();
  const audio = byId<HTMLAudioElement>("reference-trim-audio");
  const startRange = byId<HTMLInputElement>("reference-trim-start-range");
  const endRange = byId<HTMLInputElement>("reference-trim-end-range");
  const startInput = byId<HTMLInputElement>("reference-trim-start");
  const endInput = byId<HTMLInputElement>("reference-trim-end");

  startRange.addEventListener("input", () => setSelection(Number(startRange.value), endValue(), "start"));
  endRange.addEventListener("input", () => setSelection(startValue(), Number(endRange.value), "end"));
  startInput.addEventListener("change", () => setSelection(Number(startInput.value), endValue(), "start"));
  endInput.addEventListener("change", () => setSelection(startValue(), Number(endInput.value), "end"));
  byId("reference-trim-set-start").addEventListener("click", () => setSelection(audio.currentTime, endValue(), "start"));
  byId("reference-trim-set-end").addEventListener("click", () => setSelection(startValue(), audio.currentTime, "end"));
  byId("reference-trim-preview").addEventListener("click", () => {
    audio.currentTime = startValue();
    selectionPlayback = true;
    void audio.play();
  });
  audio.addEventListener("timeupdate", () => {
    if (selectionPlayback && audio.currentTime >= endValue()) {
      audio.pause();
      audio.currentTime = startValue();
      selectionPlayback = false;
    }
  });
  audio.addEventListener("pause", () => { selectionPlayback = false; });
  byId("reference-trim-close").addEventListener("click", closeReferenceTrimmer);
  byId("reference-trim-cancel").addEventListener("click", closeReferenceTrimmer);
  byId("reference-trim-apply").addEventListener("click", () => { void applyCurrentTrim(); });
  byId("reference-trim-modal").addEventListener("click", (event) => {
    if (event.target === byId("reference-trim-modal")) closeReferenceTrimmer();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !byId("reference-trim-modal").classList.contains("hidden")) closeReferenceTrimmer();
  });
  window.addEventListener("resize", drawWaveform);
}

export function openReferenceTrimmer(context: ReferenceTrimContext): void {
  void loadReference(context);
}
