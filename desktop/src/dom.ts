import { t } from "./i18n";

export function el<T extends HTMLElement>(id: string): T {
  return document.querySelector<T>(id)!;
}

export function setText(id: string, text: string): void {
  const e = el<HTMLElement>(id);
  if (e) e.textContent = t(text);
}

export function cssVar(name: string, fallback: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function setProgress(id: string, current: number, total: number): void {
  const pct = total > 0 ? Math.min(100, Math.max(0, (current / total) * 100)) : 0;
  const e = el<HTMLElement>(id);
  if (e) e.style.width = `${pct}%`;
}

export function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "−";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function formatDuration(seconds: number): string {
  return Number.isFinite(seconds) ? formatTime(seconds) : "00:00";
}
