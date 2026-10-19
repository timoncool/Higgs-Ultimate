import { invoke } from "@tauri-apps/api/core";

export type StorageLayout = {
  portable: boolean;
  root: string;
  engineDir: string;
  modelsDir: string;
  speakersDir: string;
  tempDir: string;
  webviewDir: string | null;
};

const PORTABLE_PATH_PREFIX = "@portable/";
let layout: StorageLayout | null = null;

export async function initStorageLayout(): Promise<StorageLayout> {
  layout = await invoke<StorageLayout>("storage_layout");
  return layout;
}

function pathSeparator(): string {
  return layout?.root.includes("\\") ? "\\" : "/";
}

function normalized(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+$/, "");
}

function pathStartsWith(path: string, root: string): boolean {
  const candidate = normalized(path);
  const base = normalized(root);
  const windows = root.includes("\\");
  if (windows) {
    return candidate.toLowerCase() === base.toLowerCase()
      || candidate.toLowerCase().startsWith(`${base.toLowerCase()}/`);
  }
  return candidate === base || candidate.startsWith(`${base}/`);
}

export function storePortablePath(path: string): string {
  if (!path || !layout?.portable || !pathStartsWith(path, layout.root)) return path;
  const relative = normalized(path).slice(normalized(layout.root).length).replace(/^\/+/, "");
  return `${PORTABLE_PATH_PREFIX}${relative}`;
}

export function resolveStoredPath(path: string): string {
  if (!path || !layout?.portable || !path.startsWith(PORTABLE_PATH_PREFIX)) return path;
  const relative = path.slice(PORTABLE_PATH_PREFIX.length).replace(/[\\/]+/g, pathSeparator());
  return `${layout.root.replace(/[\\/]+$/, "")}${pathSeparator()}${relative}`;
}

export function getStoredPath(key: string): string {
  return resolveStoredPath(localStorage.getItem(key) || "");
}

export function setStoredPath(key: string, path: string): void {
  if (path) localStorage.setItem(key, storePortablePath(path));
  else localStorage.removeItem(key);
}
