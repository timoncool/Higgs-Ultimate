import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { MICROPHONE_DEVICE_STORAGE_KEY } from "./config";
import { cssVar, escapeHtml, formatDuration } from "./dom";
import type {
  MicrophoneDevice,
  RecordingMeterEvent,
  RecordingResult,
  RecordingStarted,
} from "./types";

export type ReferenceRecordingContext = "clone" | "finish" | "gallery" | `speaker:${string}`;

type RecorderBindings = {
  referenceExists: (context: ReferenceRecordingContext) => boolean;
  removeReference: (context: ReferenceRecordingContext) => void;
  applyRecording: (context: ReferenceRecordingContext, result: RecordingResult) => Promise<void>;
  showToast: (message: string, tone?: "success" | "warning" | "error") => void;
};

type ActiveReferenceRecording = {
  context: ReferenceRecordingContext;
  sessionId: string;
  deviceName: string;
  maxSeconds: number;
  elapsedSeconds: number;
  slices: Array<{ low: number; high: number }>;
  stopping: boolean;
};

let bindings: RecorderBindings | null = null;
let microphoneDevices: MicrophoneDevice[] = [];
let selectedMicrophoneDevice = localStorage.getItem(MICROPHONE_DEVICE_STORAGE_KEY) || "";
let activeRecording: ActiveReferenceRecording | null = null;
let initialized = false;

export function referenceRecorderMarkup(context: ReferenceRecordingContext, compact = false): string {
  return `
    <div class="reference-recorder ${compact ? "compact" : ""}" data-recorder-context="${escapeHtml(context)}">
      <div class="recorder-toolbar">
        <select class="select-input recorder-device" data-recorder-device aria-label="Microphone input">
          <option value="">Loading microphones...</option>
        </select>
        <button class="icon-button recorder-refresh" data-recorder-action="refresh" type="button" title="Refresh microphone inputs" aria-label="Refresh microphone inputs">↻</button>
        <button class="icon-button recorder-start" data-recorder-action="start" type="button" title="Record or replace reference" aria-label="Record or replace reference">●</button>
        <button class="icon-button recorder-stop hidden" data-recorder-action="stop" type="button" title="Stop recording" aria-label="Stop recording">■</button>
        <button class="icon-button recorder-remove" data-recorder-action="remove" type="button" title="Remove reference audio" aria-label="Remove reference audio">×</button>
        <span class="recorder-time" data-recorder-time>00:00 / 00:30</span>
      </div>
      <canvas class="recorder-waveform" data-recorder-waveform aria-label="Live microphone waveform"></canvas>
    </div>`;
}

function installStaticRecorders(): void {
  const targets: Array<[string, ReferenceRecordingContext]> = [
    ["#clone-dropzone", "clone"],
    ["#finish-dropzone", "finish"],
    ["#speaker-gallery-dropzone", "gallery"],
  ];
  for (const [selector, context] of targets) {
    const dropzone = document.querySelector<HTMLElement>(selector);
    if (!dropzone || dropzone.parentElement?.querySelector(`[data-recorder-context="${context}"]`)) continue;
    dropzone.insertAdjacentHTML("afterend", referenceRecorderMarkup(context));
  }
}

function microphoneOptionMarkup(device: MicrophoneDevice): string {
  const selected = device.name === selectedMicrophoneDevice ? " selected" : "";
  const label = device.isDefault ? `${device.name} (Default)` : device.name;
  return `<option value="${escapeHtml(device.name)}"${selected}>${escapeHtml(label)}</option>`;
}

function updateMicrophoneSelectors(): void {
  const hasSelected = microphoneDevices.some((device) => device.name === selectedMicrophoneDevice);
  if (!hasSelected) {
    selectedMicrophoneDevice = microphoneDevices.find((device) => device.isDefault)?.name
      || microphoneDevices[0]?.name
      || "";
    if (selectedMicrophoneDevice) {
      localStorage.setItem(MICROPHONE_DEVICE_STORAGE_KEY, selectedMicrophoneDevice);
    } else {
      localStorage.removeItem(MICROPHONE_DEVICE_STORAGE_KEY);
    }
  }
  const options = microphoneDevices.length > 0
    ? microphoneDevices.map(microphoneOptionMarkup).join("")
    : `<option value="">No microphone found</option>`;
  for (const select of document.querySelectorAll<HTMLSelectElement>("[data-recorder-device]")) {
    select.innerHTML = options;
    select.value = selectedMicrophoneDevice;
  }
}

async function refreshMicrophoneDevices(): Promise<void> {
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-recorder-action='refresh']")) {
    button.disabled = true;
  }
  try {
    microphoneDevices = await invoke<MicrophoneDevice[]>("list_microphone_devices");
    updateMicrophoneSelectors();
    if (microphoneDevices.length === 0) bindings?.showToast("No microphone input was found", "warning");
  } catch (error) {
    microphoneDevices = [];
    updateMicrophoneSelectors();
    bindings?.showToast(`Microphone scan failed: ${String(error)}`, "error");
  } finally {
    syncReferenceRecorderUi();
  }
}

function recorderRoot(context: ReferenceRecordingContext): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[data-recorder-context="${CSS.escape(context)}"]`);
}

function drawReferenceRecorder(context: ReferenceRecordingContext): void {
  const canvas = recorderRoot(context)?.querySelector<HTMLCanvasElement>("[data-recorder-waveform]");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.floor(width * dpr);
  const pixelHeight = Math.floor(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = cssVar("--bg-inner", "#0d1013");
  ctx.fillRect(0, 0, width, height);
  const mid = height / 2;
  ctx.strokeStyle = cssVar("--border", "#30363d");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();

  const active = activeRecording?.context === context ? activeRecording : null;
  if (!active || active.slices.length === 0) return;
  const visibleCount = Math.max(1, Math.floor(width / 3));
  const visible = active.slices.slice(-visibleCount);
  const barWidth = Math.max(1, width / visibleCount - 1);
  const leftPad = Math.max(0, width - visible.length * (barWidth + 1));
  ctx.fillStyle = cssVar("--accent", "#25b8ab");
  visible.forEach((slice, index) => {
    const high = Math.max(0.012, slice.high);
    const low = Math.min(-0.012, slice.low);
    const top = mid - high * (height * 0.46);
    const bottom = mid - low * (height * 0.46);
    ctx.fillRect(leftPad + index * (barWidth + 1), top, barWidth, Math.max(1, bottom - top));
  });
}

export function redrawReferenceRecorders(): void {
  for (const root of document.querySelectorAll<HTMLElement>("[data-recorder-context]")) {
    const context = root.dataset.recorderContext as ReferenceRecordingContext | undefined;
    if (context) drawReferenceRecorder(context);
  }
}

export function syncReferenceRecorderUi(): void {
  for (const root of document.querySelectorAll<HTMLElement>("[data-recorder-context]")) {
    const context = root.dataset.recorderContext as ReferenceRecordingContext | undefined;
    if (!context) continue;
    const activeHere = activeRecording?.context === context;
    const anyActive = Boolean(activeRecording);
    const start = root.querySelector<HTMLButtonElement>("[data-recorder-action='start']");
    const stop = root.querySelector<HTMLButtonElement>("[data-recorder-action='stop']");
    const remove = root.querySelector<HTMLButtonElement>("[data-recorder-action='remove']");
    const refresh = root.querySelector<HTMLButtonElement>("[data-recorder-action='refresh']");
    const device = root.querySelector<HTMLSelectElement>("[data-recorder-device]");
    const time = root.querySelector<HTMLElement>("[data-recorder-time]");
    if (start) {
      start.classList.toggle("hidden", Boolean(activeHere));
      start.disabled = anyActive || microphoneDevices.length === 0;
    }
    if (stop) {
      stop.classList.toggle("hidden", !activeHere);
      stop.disabled = Boolean(activeRecording?.stopping);
      stop.classList.toggle("busy", Boolean(activeHere && activeRecording?.stopping));
    }
    if (remove) remove.disabled = anyActive || !bindings?.referenceExists(context);
    if (refresh) refresh.disabled = anyActive;
    if (device) device.disabled = anyActive || microphoneDevices.length === 0;
    if (time) {
      const elapsed = activeHere ? activeRecording?.elapsedSeconds || 0 : 0;
      const limit = activeHere ? activeRecording?.maxSeconds || 30 : 30;
      time.textContent = `${formatDuration(elapsed)} / ${formatDuration(limit)}`;
    }
    root.classList.toggle("recording", Boolean(activeHere));
    drawReferenceRecorder(context);
  }
}

export function refreshReferenceRecorderUi(): void {
  updateMicrophoneSelectors();
  syncReferenceRecorderUi();
}

async function startReferenceRecording(context: ReferenceRecordingContext): Promise<void> {
  if (activeRecording) {
    bindings?.showToast("Stop the current microphone recording first", "warning");
    return;
  }
  try {
    const started = await invoke<RecordingStarted>("start_reference_recording", {
      deviceName: selectedMicrophoneDevice || null,
    });
    activeRecording = {
      context,
      sessionId: started.sessionId,
      deviceName: started.deviceName,
      maxSeconds: started.maxSeconds,
      elapsedSeconds: 0,
      slices: [],
      stopping: false,
    };
    syncReferenceRecorderUi();
  } catch (error) {
    bindings?.showToast(`Microphone recording failed: ${String(error)}`, "error");
    await refreshMicrophoneDevices();
  }
}

async function stopReferenceRecording(): Promise<void> {
  const recording = activeRecording;
  if (!recording || recording.stopping) return;
  recording.stopping = true;
  syncReferenceRecorderUi();
  try {
    const result = await invoke<RecordingResult>("stop_reference_recording");
    if (result.sessionId !== recording.sessionId) {
      throw new Error("The finalized microphone session did not match the active reference");
    }
    activeRecording = null;
    syncReferenceRecorderUi();
    await bindings?.applyRecording(recording.context, result);
    bindings?.showToast(
      result.reachedLimit ? "Reference recorded at the 30-second limit" : "Reference recording ready",
    );
  } catch (error) {
    activeRecording = null;
    syncReferenceRecorderUi();
    bindings?.showToast(`Could not finalize microphone recording: ${String(error)}`, "error");
  }
}

function updateLiveMeter(event: RecordingMeterEvent): void {
  const recording = activeRecording;
  if (!recording || event.sessionId !== recording.sessionId) return;
  recording.elapsedSeconds = event.elapsedSeconds;
  recording.slices.push({ low: event.low, high: event.high });
  if (recording.slices.length > 640) recording.slices.splice(0, recording.slices.length - 640);
  const time = recorderRoot(recording.context)?.querySelector<HTMLElement>("[data-recorder-time]");
  if (time) {
    time.textContent = `${formatDuration(recording.elapsedSeconds)} / ${formatDuration(recording.maxSeconds)}`;
  }
  drawReferenceRecorder(recording.context);
}

function bindDomEvents(): void {
  document.addEventListener("change", (event) => {
    const select = (event.target as HTMLElement).closest<HTMLSelectElement>("[data-recorder-device]");
    if (!select || activeRecording) return;
    selectedMicrophoneDevice = select.value;
    if (selectedMicrophoneDevice) {
      localStorage.setItem(MICROPHONE_DEVICE_STORAGE_KEY, selectedMicrophoneDevice);
    }
    updateMicrophoneSelectors();
  });
  document.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-recorder-action]");
    const root = button?.closest<HTMLElement>("[data-recorder-context]");
    const context = root?.dataset.recorderContext as ReferenceRecordingContext | undefined;
    if (!button || !context) return;
    const action = button.dataset.recorderAction;
    if (action === "start") void startReferenceRecording(context);
    else if (action === "stop") void stopReferenceRecording();
    else if (action === "remove") bindings?.removeReference(context);
    else if (action === "refresh") void refreshMicrophoneDevices();
  });
}

async function bindBackendEvents(): Promise<void> {
  await listen<RecordingMeterEvent>("reference-recording-meter", (event) => {
    updateLiveMeter(event.payload);
  });
  await listen<{ sessionId: string }>("reference-recording-limit", (event) => {
    if (event.payload.sessionId === activeRecording?.sessionId) void stopReferenceRecording();
  });
  await listen<{ sessionId: string; message: string }>("reference-recording-error", (event) => {
    if (event.payload.sessionId === activeRecording?.sessionId) void stopReferenceRecording();
  });
}

export async function initReferenceRecorder(recorderBindings: RecorderBindings): Promise<void> {
  bindings = recorderBindings;
  installStaticRecorders();
  updateMicrophoneSelectors();
  if (!initialized) {
    initialized = true;
    bindDomEvents();
    await bindBackendEvents();
  }
  syncReferenceRecorderUi();
  await refreshMicrophoneDevices();
}
