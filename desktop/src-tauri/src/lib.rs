// Public so headless examples (`cargo run --example asr_smoke` / `diar_test`) can
// exercise the real decode + transcription paths; nothing else needs it public.
pub mod audio;
mod download;
mod engine;
pub mod parakeet;
mod recorder;

use engine::{
    AudioChunkCallback, AudioResult, Engine, EngineError, GenerateRequest, LoadModelRequest,
    ProgressCallback,
};
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{Read, Seek, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{Pid, System};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{App, AppHandle, Emitter, Manager, State};

static STUDIO_JOB_COUNTER: AtomicU64 = AtomicU64::new(1);
const REFERENCE_MAX_SECONDS: f64 = 30.0;

/// The engine is stored as Arc<Engine> inside the mutex. When a generate call
/// needs to run, it clones the Arc out of the lock and releases the mutex —
/// the engine stays alive for the duration of the call even if the UI triggers
/// an unload concurrently.
struct AppState {
    engine: Arc<Mutex<Option<Arc<Engine>>>>,
    engine_dir: Mutex<PathBuf>,
    nvml: Mutex<Option<Nvml>>,
    sys: Mutex<System>,
    download_control: Arc<download::DownloadControl>,
    api_server: Mutex<Option<ApiServerHandle>>,
    api_speakers: Arc<Mutex<Vec<ApiSpeakerPersona>>>,
    generation_queue: Arc<Mutex<()>>,
    minimize_to_tray: Arc<AtomicBool>,
}

impl AppState {
    fn engine_dir(&self) -> PathBuf {
        self.engine_dir.lock().unwrap().clone()
    }

    fn clone_engine(&self) -> Result<Arc<Engine>, String> {
        let guard = self.engine.lock().unwrap();
        (*guard).clone().ok_or("engine not loaded".to_string())
    }
}

#[derive(Clone)]
struct ApiServerHandle {
    stop: Arc<AtomicBool>,
    host: String,
    port: u16,
    api_key: String,
    started_at: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiServerConfig {
    host: String,
    port: u16,
    api_key: String,
    speakers: Option<Vec<ApiSpeakerPersona>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiSpeakerPersona {
    id: String,
    name: String,
    ref_path: String,
    ref_text: String,
    #[serde(default)]
    cache_path: String,
    #[serde(default)]
    normalize: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpeakerArchivePersona {
    id: String,
    name: String,
    ref_path: String,
    ref_name: String,
    ref_text: String,
    notes: String,
    photo_path: String,
    #[serde(default)]
    cache_path: String,
    #[serde(default)]
    normalize: bool,
    created_at: u64,
    updated_at: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiLogEvent {
    level: String,
    kind: String,
    method: String,
    route: String,
    status: u16,
    latency_ms: u128,
    message: String,
    job_id: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StudioJobEvent {
    id: String,
    source: String,
    workflow: String,
    status: String,
    label: String,
    phase: String,
    current: Option<i32>,
    total: Option<i32>,
    message: String,
}

fn next_studio_job_id(source: &str) -> String {
    let seq = STUDIO_JOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{source}-{}-{seq}", chrono_like_millis())
}

fn chrono_like_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn emit_studio_job(
    app: &AppHandle,
    id: &str,
    source: &str,
    workflow: &str,
    status: &str,
    label: &str,
    phase: &str,
    current: Option<i32>,
    total: Option<i32>,
    message: &str,
) {
    let _ = app.emit(
        "studio-job",
        StudioJobEvent {
            id: id.to_string(),
            source: source.to_string(),
            workflow: workflow.to_string(),
            status: status.to_string(),
            label: label.to_string(),
            phase: phase.to_string(),
            current,
            total,
            message: message.to_string(),
        },
    );
}

fn engine_filename() -> &'static str {
    if cfg!(windows) {
        "audiocpp_engine.dll"
    } else {
        "libaudiocpp_engine.so"
    }
}

fn engine_package_filenames() -> Vec<&'static str> {
    if cfg!(windows) {
        vec![
            "audiocpp_engine.dll",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "VCOMP140.DLL",
            "MSVCP140.dll",
            "VCRUNTIME140.dll",
            "VCRUNTIME140_1.dll",
        ]
    } else {
        vec![engine_filename()]
    }
}

fn engine_package_base_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with(".dll") || lower.ends_with(".so") || lower.ends_with(".dylib") {
        return trimmed
            .rsplit_once('/')
            .map(|(base, _)| base.to_string())
            .unwrap_or_else(|| trimmed.to_string());
    }
    trimmed.to_string()
}

/// Portable data root: the directory that contains the executable. Everything
/// the app writes (models, speakers, engine, whisper, webview state) lives here
/// so the whole folder can be moved or deleted without leaving anything behind
/// in the user profile. Replaces the old `~/audiocpp` layout.
fn app_root_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_engine_download_dir() -> PathBuf {
    app_root_dir().join("resources").join("engine")
}

fn user_models_root() -> PathBuf {
    app_root_dir().join("models")
}

fn resolve_download_dest_dir(dest_dir: &str) -> PathBuf {
    let path = PathBuf::from(dest_dir);
    if path.is_absolute() {
        return path;
    }

    let starts_with_models = path
        .components()
        .next()
        .and_then(|component| match component {
            std::path::Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        })
        .map(|value| value.eq_ignore_ascii_case("models"))
        .unwrap_or(false);

    if starts_with_models {
        // "models/<folder>/<file>" -> <app>/models/<folder>/<file>
        app_root_dir().join(path)
    } else {
        // bare relative path -> <app>/models/<...>
        user_models_root().join(path)
    }
}

fn engine_candidate_dirs(app: Option<&AppHandle>, download_dir: PathBuf) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(download_dir);

    if let Some(app) = app {
        if let Ok(resource_dir) = app.path().resource_dir() {
            candidates.push(resource_dir.join("engine"));
        }
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    candidates.push(exe_dir.join("resources").join("engine"));
    candidates.push(exe_dir);

    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("engine"),
    );

    candidates
}

fn find_engine_library(app: Option<&AppHandle>, state: &AppState) -> Option<PathBuf> {
    let filename = engine_filename();
    engine_candidate_dirs(app, state.engine_dir())
        .into_iter()
        .map(|dir| dir.join(filename))
        .find(|path| path.exists())
}

fn map_engine_err(e: &EngineError) -> String {
    match e {
        EngineError::Cancelled => "Generation cancelled".into(),
        EngineError::Generation(msg) => msg.clone(),
        _ => e.to_string(),
    }
}

fn is_higgs_asset_root(path: &Path) -> bool {
    path.join("config.json").exists() && path.join("tokenizer.json").exists()
}

fn higgs_asset_candidates(resource_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(value) = std::env::var_os("HIGGS_TTS_SMALL_ASSETS_ROOT") {
        candidates.push(PathBuf::from(value));
    }
    if let Some(value) = std::env::var_os("HIGGS_TTS_ASSETS_ROOT") {
        candidates.push(PathBuf::from(value));
    }

    if let Some(resource_dir) = resource_dir {
        let bundled_assets = resource_dir.join("higgs-assets");
        candidates.push(bundled_assets.join("higgs-audio-v3-tts-4b"));
        candidates.push(bundled_assets);
    }

    let dev_assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("higgs-assets");
    candidates.push(dev_assets.join("higgs-audio-v3-tts-4b"));
    candidates.push(dev_assets);

    candidates
}

fn find_higgs_asset_root(resource_dir: Option<PathBuf>) -> Option<PathBuf> {
    higgs_asset_candidates(resource_dir)
        .into_iter()
        .find(|path| is_higgs_asset_root(path))
}

fn configure_higgs_asset_env_from_resource_dir(resource_dir: Option<PathBuf>) {
    if let Some(root) = find_higgs_asset_root(resource_dir) {
        std::env::set_var("HIGGS_TTS_SMALL_ASSETS_ROOT", &root);
        std::env::set_var("HIGGS_TTS_ASSETS_ROOT", root);
    }
}

fn configure_higgs_asset_env(app: &App) {
    configure_higgs_asset_env_from_resource_dir(app.path().resource_dir().ok());
}

fn configure_higgs_asset_env_for_handle(app: &AppHandle) {
    configure_higgs_asset_env_from_resource_dir(app.path().resource_dir().ok());
}

fn model_assets_dest_root(model_path: &str) -> PathBuf {
    let path = PathBuf::from(model_path);
    if path.is_file() {
        return path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.clone());
    }

    let looks_like_weight = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            extension.eq_ignore_ascii_case("gguf") || extension.eq_ignore_ascii_case("safetensors")
        })
        .unwrap_or(false);

    if looks_like_weight {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.clone())
    } else {
        path
    }
}

fn copy_dir_contents(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    for entry in std::fs::read_dir(src).map_err(|e| format!("{}: {e}", src.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_contents(&src_path, &dest_path)?;
        } else if !dest_path.exists() {
            std::fs::copy(&src_path, &dest_path)
                .map_err(|e| format!("{} -> {}: {e}", src_path.display(), dest_path.display()))?;
        }
    }
    Ok(())
}

fn ensure_higgs_assets_in_model_dir(app: &AppHandle, model_path: &str) -> Result<(), String> {
    let dest_root = model_assets_dest_root(model_path);
    if is_higgs_asset_root(&dest_root) {
        return Ok(());
    }

    let Some(source_root) = find_higgs_asset_root(app.path().resource_dir().ok()) else {
        return Ok(());
    };

    if dest_root.exists() {
        let source_canonical = source_root.canonicalize().ok();
        let dest_canonical = dest_root.canonicalize().ok();
        if source_canonical.is_some() && source_canonical == dest_canonical {
            return Ok(());
        }
    }

    std::fs::create_dir_all(&dest_root).map_err(|e| {
        format!(
            "Higgs TTS assets are missing and the app could not create {}: {e}",
            dest_root.display()
        )
    })?;

    for filename in [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "higgs_audio_v2_tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    ] {
        let source = source_root.join(filename);
        let dest = dest_root.join(filename);
        if source.exists() && !dest.exists() {
            std::fs::copy(&source, &dest)
                .map_err(|e| format!("{} -> {}: {e}", source.display(), dest.display()))?;
        }
    }

    let source_assets = source_root.join("assets");
    if source_assets.exists() {
        copy_dir_contents(&source_assets, &dest_root.join("assets"))?;
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Engine lifecycle
// ═══════════════════════════════════════════════════════════════════════════

#[tauri::command]
fn engine_status(state: State<'_, AppState>) -> serde_json::Value {
    let guard = state.engine.lock().unwrap();
    let engine_loaded = guard.is_some();
    let model_loaded = guard.as_ref().map(|e| e.is_model_loaded()).unwrap_or(false);
    let generating = guard.as_ref().map(|e| e.is_generating()).unwrap_or(false);
    let supports_streaming = guard
        .as_ref()
        .map(|e| e.supports_streaming())
        .unwrap_or(false);
    serde_json::json!({
        "engineLoaded": engine_loaded,
        "modelLoaded": model_loaded,
        "generating": generating,
        "supportsStreaming": supports_streaming,
    })
}

#[tauri::command]
fn engine_version(state: State<'_, AppState>) -> String {
    let guard = state.engine.lock().unwrap();
    guard
        .as_ref()
        .map(|e| e.version())
        .unwrap_or_else(|| "engine not loaded".into())
}

#[tauri::command]
fn bundled_engine_path(app: AppHandle, state: State<'_, AppState>) -> Option<String> {
    find_engine_library(Some(&app), &state).map(|dll| dll.to_string_lossy().into_owned())
}

fn resolve_engine_path(
    app: Option<&AppHandle>,
    state: &AppState,
    library_path: Option<String>,
) -> Result<PathBuf, String> {
    if let Some(p) = library_path {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("Engine library not found at {}.", path.display()));
    }

    if let Some(dll) = find_engine_library(app, state) {
        return Ok(dll);
    }

    let download_dir = state.engine_dir();
    Err(format!(
        "Engine library not found. Click Download Engine DLLs or copy {} into {}.",
        engine_filename(),
        download_dir.display()
    ))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineDependencyStatus {
    name: String,
    pattern: String,
    category: String,
    required: bool,
    found_path: Option<String>,
    fix: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineDependencyDiagnostic {
    ok: bool,
    platform: String,
    engine_path: String,
    detected: Vec<EngineDependencyStatus>,
    missing: Vec<EngineDependencyStatus>,
    optional_missing: Vec<EngineDependencyStatus>,
    search_dirs: Vec<String>,
    suggestions: Vec<String>,
    raw_error: Option<String>,
}

#[derive(Clone)]
struct EngineDependencySpec {
    name: &'static str,
    pattern: &'static str,
    category: &'static str,
    required: bool,
    fix: &'static str,
}

fn diagnose_engine_dependencies(
    engine_path: &Path,
    raw_error: Option<&str>,
) -> EngineDependencyDiagnostic {
    #[cfg(target_os = "windows")]
    {
        diagnose_windows_engine_dependencies(engine_path, raw_error)
    }

    #[cfg(not(target_os = "windows"))]
    {
        EngineDependencyDiagnostic {
            ok: true,
            platform: std::env::consts::OS.to_string(),
            engine_path: engine_path.to_string_lossy().into_owned(),
            detected: Vec::new(),
            missing: Vec::new(),
            optional_missing: Vec::new(),
            search_dirs: Vec::new(),
            suggestions: Vec::new(),
            raw_error: raw_error.map(|e| e.to_string()),
        }
    }
}

#[cfg(target_os = "windows")]
fn diagnose_windows_engine_dependencies(
    engine_path: &Path,
    raw_error: Option<&str>,
) -> EngineDependencyDiagnostic {
    let specs = windows_engine_dependency_specs();
    let search_dirs = windows_dependency_search_dirs(engine_path);
    let mut detected = Vec::new();
    let mut missing = Vec::new();
    let mut optional_missing = Vec::new();
    let mut suggestions = Vec::new();

    for spec in specs {
        let found = find_dependency_pattern(&search_dirs, spec.pattern);
        let status = EngineDependencyStatus {
            name: spec.name.to_string(),
            pattern: spec.pattern.to_string(),
            category: spec.category.to_string(),
            required: spec.required,
            found_path: found.as_ref().map(|p| p.to_string_lossy().into_owned()),
            fix: spec.fix.to_string(),
        };

        if found.is_some() {
            detected.push(status);
        } else if spec.required {
            if !suggestions.iter().any(|s| s == spec.fix) {
                suggestions.push(spec.fix.to_string());
            }
            missing.push(status);
        } else {
            optional_missing.push(status);
        }
    }

    EngineDependencyDiagnostic {
        ok: missing.is_empty(),
        platform: "windows".to_string(),
        engine_path: engine_path.to_string_lossy().into_owned(),
        detected,
        missing,
        optional_missing,
        search_dirs: search_dirs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        suggestions,
        raw_error: raw_error.map(|e| e.to_string()),
    }
}

#[cfg(target_os = "windows")]
fn windows_engine_dependency_specs() -> Vec<EngineDependencySpec> {
    vec![
        EngineDependencySpec {
            name: "NVIDIA CUDA driver",
            pattern: "nvcuda.dll",
            category: "NVIDIA driver",
            required: true,
            fix: "Install or update the NVIDIA Studio/Game Ready driver, then reboot.",
        },
        EngineDependencySpec {
            name: "CUDA 13 cuBLAS",
            pattern: "cublas64_13.dll",
            category: "CUDA 13 runtime",
            required: true,
            fix: "Download Engine DLLs in the app, or install CUDA Toolkit 13.x and make sure its bin folder is on PATH.",
        },
        EngineDependencySpec {
            name: "CUDA 13 cuBLASLt",
            pattern: "cublasLt64_13.dll",
            category: "CUDA 13 runtime",
            required: true,
            fix: "Download Engine DLLs in the app, or install CUDA Toolkit 13.x and make sure its bin folder is on PATH.",
        },
        EngineDependencySpec {
            name: "Microsoft Visual C++ runtime",
            pattern: "vcruntime140.dll",
            category: "Microsoft VC++ Redistributable",
            required: true,
            fix: "Download Engine DLLs in the app, or install Microsoft Visual C++ Redistributable 2015-2022 x64.",
        },
        EngineDependencySpec {
            name: "Microsoft Visual C++ runtime 14.1",
            pattern: "vcruntime140_1.dll",
            category: "Microsoft VC++ Redistributable",
            required: true,
            fix: "Download Engine DLLs in the app, or install Microsoft Visual C++ Redistributable 2015-2022 x64.",
        },
        EngineDependencySpec {
            name: "Microsoft C++ standard library",
            pattern: "msvcp140.dll",
            category: "Microsoft VC++ Redistributable",
            required: true,
            fix: "Download Engine DLLs in the app, or install Microsoft Visual C++ Redistributable 2015-2022 x64.",
        },
        EngineDependencySpec {
            name: "Microsoft OpenMP runtime",
            pattern: "VCOMP140.DLL",
            category: "Microsoft VC++ Redistributable",
            required: true,
            fix: "Download Engine DLLs in the app, or install Microsoft Visual C++ Redistributable 2015-2022 x64.",
        },
    ]
}

#[cfg(target_os = "windows")]
fn windows_dependency_search_dirs(engine_path: &Path) -> Vec<PathBuf> {
    fn push_dir(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
        if !dir.exists() || !dir.is_dir() {
            return;
        }
        let normalized = dir.to_string_lossy().to_lowercase();
        if dirs
            .iter()
            .any(|existing| existing.to_string_lossy().to_lowercase() == normalized)
        {
            return;
        }
        dirs.push(dir);
    }

    let mut dirs = Vec::new();
    if let Some(parent) = engine_path.parent() {
        push_dir(&mut dirs, parent.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push_dir(&mut dirs, parent.to_path_buf());
            push_dir(&mut dirs, parent.join("resources").join("engine"));
        }
    }
    if let Some(windir) = std::env::var_os("WINDIR") {
        push_dir(&mut dirs, PathBuf::from(windir).join("System32"));
    }

    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy().to_ascii_uppercase();
        if key == "CUDA_PATH" || key.starts_with("CUDA_PATH_V") {
            let cuda_bin = PathBuf::from(value).join("bin");
            push_dir(&mut dirs, cuda_bin.clone());
            push_dir(&mut dirs, cuda_bin.join("x64"));
        }
    }

    let program_roots = ["ProgramFiles", "ProgramW6432"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    for root in program_roots {
        let cuda_root = root
            .join("NVIDIA GPU Computing Toolkit")
            .join("CUDA");
        for version in ["v13.3", "v13.2", "v13.1", "v13.0", "v13"] {
            let cuda_bin = cuda_root.join(version).join("bin");
            push_dir(&mut dirs, cuda_bin.clone());
            push_dir(&mut dirs, cuda_bin.join("x64"));
        }
    }

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            push_dir(&mut dirs, dir);
        }
    }

    dirs
}

#[cfg(target_os = "windows")]
fn find_dependency_pattern(search_dirs: &[PathBuf], pattern: &str) -> Option<PathBuf> {
    if !pattern.contains('*') {
        for dir in search_dirs {
            let candidate = dir.join(pattern);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        return None;
    }

    let lower = pattern.to_ascii_lowercase();
    let mut parts = lower.splitn(2, '*');
    let prefix = parts.next().unwrap_or_default();
    let suffix = parts.next().unwrap_or_default();

    for dir in search_dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if file_name.starts_with(prefix) && file_name.ends_with(suffix) {
                return Some(entry.path());
            }
        }
    }

    None
}

fn format_dependency_diagnostic_error(diagnostic: &EngineDependencyDiagnostic) -> String {
    if diagnostic.missing.is_empty() {
        return diagnostic
            .raw_error
            .clone()
            .unwrap_or_else(|| "Engine dependency check failed.".to_string());
    }

    let missing = diagnostic
        .missing
        .iter()
        .map(|dep| format!("{} ({})", dep.pattern, dep.category))
        .collect::<Vec<_>>()
        .join(", ");
    let fixes = diagnostic.suggestions.join(" ");
    format!(
        "Engine dependency check failed. Missing: {missing}. {fixes}"
    )
}

#[tauri::command]
fn diagnose_engine_load(
    app: AppHandle,
    state: State<'_, AppState>,
    library_path: Option<String>,
) -> Result<serde_json::Value, String> {
    let lib_path = resolve_engine_path(Some(&app), &state, library_path)?;
    let diagnostic = diagnose_engine_dependencies(&lib_path, None);
    serde_json::to_value(diagnostic).map_err(|e| e.to_string())
}

#[tauri::command]
async fn download_engine_dll(
    app: AppHandle,
    state: State<'_, AppState>,
    url: String,
) -> Result<serde_json::Value, String> {
    let dest_dir = state.engine_dir();
    let control = state.download_control.clone();
    let base_url = engine_package_base_url(&url);
    let files = engine_package_filenames();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut downloaded_files = Vec::new();
        let mut total_size = 0_u64;
        for filename in files {
            let file_url = format!("{}/{}", base_url, filename);
            let result =
                download::download_file(&file_url, &dest_dir, Some(filename), &app, control.clone())?;
            total_size += result.size;
            downloaded_files.push(result.path);
        }

        Ok::<serde_json::Value, download::DownloadError>(serde_json::json!({
            "path": dest_dir.to_string_lossy(),
            "size": total_size,
            "files": downloaded_files,
        }))
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(result)
}

#[tauri::command]
async fn load_engine(
    app: AppHandle,
    state: State<'_, AppState>,
    library_path: Option<String>,
) -> Result<serde_json::Value, String> {
    let lib_path = resolve_engine_path(Some(&app), &state, library_path)?;

    #[cfg(target_os = "windows")]
    {
        for dir in windows_dependency_search_dirs(&lib_path) {
            engine::add_dll_directory(&dir);
        }
    }

    let diagnostic = diagnose_engine_dependencies(&lib_path, None);
    if !diagnostic.ok {
        return Err(format_dependency_diagnostic_error(&diagnostic));
    }

    let load_path = lib_path.clone();
    let engine = tauri::async_runtime::spawn_blocking(move || Engine::load(&lib_path))
        .await
        .map_err(|e| format!("task join error: {e}"))?
        .map_err(|e| {
            let diagnostic = diagnose_engine_dependencies(&load_path, Some(&e.to_string()));
            if diagnostic.ok {
                e.to_string()
            } else {
                format_dependency_diagnostic_error(&diagnostic)
            }
        })?;

    let version = engine.version();
    let supports_streaming = engine.supports_streaming();
    *state.engine.lock().unwrap() = Some(Arc::new(engine));

    // Initialize NVML for hardware monitoring.
    {
        let mut nvml_guard = state.nvml.lock().unwrap();
        if nvml_guard.is_none() {
            if let Ok(nvml) = Nvml::init() {
                *nvml_guard = Some(nvml);
            }
        }
    }

    let _ = app.emit(
        "model-status",
        serde_json::json!({
            "engineLoaded": true,
            "modelLoaded": false,
            "supportsStreaming": supports_streaming,
        }),
    );
    Ok(serde_json::json!({
        "success": true,
        "version": version,
        "supportsStreaming": supports_streaming,
    }))
}

#[tauri::command]
fn unload_engine(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state.engine.lock().unwrap();
    if let Some(engine) = guard.as_ref() {
        engine.unload_model();
    }
    *guard = None;
    drop(guard);
    let _ = app.emit(
        "model-status",
        serde_json::json!({
            "engineLoaded": false,
            "modelLoaded": false,
            "supportsStreaming": false,
        }),
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Model management
// ═══════════════════════════════════════════════════════════════════════════

#[tauri::command]
async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    request: LoadModelRequest,
) -> Result<serde_json::Value, String> {
    configure_higgs_asset_env_for_handle(&app);
    ensure_higgs_assets_in_model_dir(&app, &request.model_root)?;

    let engine = {
        let guard = state.engine.lock().unwrap();
        guard.as_ref().ok_or("engine not loaded")?.clone()
    };
    let supports_streaming = engine.supports_streaming();

    let app_clone = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || engine.load_model(&request))
        .await
        .map_err(|e| format!("task join error: {e}"))?
        .map_err(|e| map_engine_err(&e))?;

    let _ = app_clone.emit(
        "model-status",
        serde_json::json!({
            "engineLoaded": true,
            "modelLoaded": true,
            "supportsStreaming": supports_streaming,
            "family": result.family,
            "displayName": result.display_name,
            "weightType": result.weight_type,
        }),
    );

    Ok(serde_json::json!({
        "success": true,
        "modelInfo": {
            "family": result.family,
            "displayName": result.display_name,
            "weightType": result.weight_type,
            "modelRoot": result.model_root,
        }
    }))
}

#[tauri::command]
fn unload_model(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let guard = state.engine.lock().unwrap();
    let engine = guard.as_ref().ok_or("engine not loaded")?;
    let supports_streaming = engine.supports_streaming();
    engine.unload_model();
    drop(guard);
    let _ = app.emit(
        "model-status",
        serde_json::json!({
            "engineLoaded": true,
            "modelLoaded": false,
            "supportsStreaming": supports_streaming,
        }),
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Cancel
// ═══════════════════════════════════════════════════════════════════════════

#[tauri::command]
fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    let guard = state.engine.lock().unwrap();
    if let Some(engine) = guard.as_ref() {
        engine.cancel();
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Generation
// ═══════════════════════════════════════════════════════════════════════════

fn build_progress(app: &AppHandle) -> ProgressCallback {
    let app = app.clone();
    Arc::new(move |current: i32, total: i32, phase: &str| {
        if emit_native_diagnostic(&app, phase) {
            return;
        }
        let _ = app.emit(
            "generation-progress",
            serde_json::json!({
                "current": current, "total": total, "phase": phase,
            }),
        );
    })
}

fn build_audio_chunk(app: &AppHandle) -> AudioChunkCallback {
    let app = app.clone();
    Arc::new(
        move |sample_rate: i32,
              channels: i32,
              start_sample: i64,
              samples: &[f32],
              is_final: bool| {
            let wav_base64 = AudioResult {
                sample_rate,
                channels,
                samples: samples.to_vec(),
            }
            .encode_base64_wav();
            let _ = app.emit(
                "generation-audio-chunk",
                serde_json::json!({
                    "sampleRate": sample_rate,
                    "channels": channels,
                    "startSample": start_sample,
                    "sampleCount": samples.len(),
                    "wavBase64": wav_base64,
                    "isFinal": is_final,
                }),
            );
        },
    )
}

fn noop_audio_chunk() -> AudioChunkCallback {
    Arc::new(|_, _, _, _, _| {})
}

fn option_bool(options: &serde_json::Value, key: &str) -> bool {
    options
        .get(key)
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn set_bool_option(options: &mut serde_json::Value, key: &str, value: bool) {
    if let Some(map) = options.as_object_mut() {
        map.insert(key.to_string(), serde_json::json!(value));
    }
}

fn set_string_option(options: &mut serde_json::Value, key: &str, value: &str) {
    if value.trim().is_empty() {
        return;
    }
    if let Some(map) = options.as_object_mut() {
        map.insert(key.to_string(), serde_json::json!(value));
    }
}

fn prepare_reference_audio(
    path: &str,
    options: &serde_json::Value,
    max_seconds: Option<f64>,
) -> Result<String, String> {
    let prepared = audio::prepare_reference_wav(
        path,
        option_bool(options, "normalize_reference"),
        0.95,
        max_seconds,
    )
    .map_err(|e| e.to_string())?;
    Ok(prepared.path)
}

fn prepare_voice_reference_audio(path: &str, options: &serde_json::Value) -> Result<String, String> {
    prepare_reference_audio(path, options, Some(REFERENCE_MAX_SECONDS))
}

fn prepare_continuation_audio(path: &str, options: &serde_json::Value) -> Result<String, String> {
    prepare_reference_audio(path, options, None)
}

#[tauri::command]
async fn generate_tts(
    app: AppHandle,
    state: State<'_, AppState>,
    request: GenerateRequest,
) -> Result<serde_json::Value, String> {
    let engine = state.clone_engine()?;
    let queue = state.generation_queue.clone();
    let mut options = request.options.unwrap_or(serde_json::json!({}));
    let stream_playback_enabled = option_bool(&options, "stream_playback");
    let stream_backend = engine.supports_streaming();
    set_bool_option(
        &mut options,
        "emit_stream_audio_chunks",
        stream_playback_enabled,
    );
    let progress = build_progress(&app);
    let audio_chunk = if stream_playback_enabled {
        build_audio_chunk(&app)
    } else {
        noop_audio_chunk()
    };

    let result = tauri::async_runtime::spawn_blocking(move || -> Result<_, EngineError> {
        let _queue = queue
            .lock()
            .map_err(|_| EngineError::Generation("generation queue poisoned".into()))?;
        if stream_backend {
            engine.generate_tts_stream(&request.text, &options, progress, audio_chunk)
        } else {
            engine.generate_tts(&request.text, &options, progress)
        }
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| map_engine_err(&e))?;

    Ok(serde_json::json!({
        "sampleRate": result.sample_rate,
        "channels": result.channels,
        "sampleCount": result.samples.len(),
        "wavBase64": result.encode_base64_wav(),
    }))
}

#[tauri::command]
async fn generate_voice_clone(
    app: AppHandle,
    state: State<'_, AppState>,
    request: GenerateRequest,
) -> Result<serde_json::Value, String> {
    let engine = state.clone_engine()?;
    let ref_path = request
        .ref_audio_path
        .clone()
        .ok_or("ref_audio_path is required for voice clone")?;
    let queue = state.generation_queue.clone();
    let mut options = request.options.unwrap_or(serde_json::json!({}));
    let ref_wav = prepare_voice_reference_audio(&ref_path, &options)?;
    let stream_playback_enabled = option_bool(&options, "stream_playback");
    let stream_backend = engine.supports_streaming();
    set_bool_option(
        &mut options,
        "emit_stream_audio_chunks",
        stream_playback_enabled,
    );
    let progress = build_progress(&app);
    let audio_chunk = if stream_playback_enabled {
        build_audio_chunk(&app)
    } else {
        noop_audio_chunk()
    };

    let result = tauri::async_runtime::spawn_blocking(move || -> Result<_, EngineError> {
        let _queue = queue
            .lock()
            .map_err(|_| EngineError::Generation("generation queue poisoned".into()))?;
        if stream_backend {
            engine.generate_voice_clone_stream(
                &request.text,
                &ref_wav,
                request.ref_text.as_deref(),
                &options,
                progress,
                audio_chunk,
            )
        } else {
            engine.generate_voice_clone(
                &request.text,
                &ref_wav,
                request.ref_text.as_deref(),
                &options,
                progress,
            )
        }
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| map_engine_err(&e))?;

    Ok(serde_json::json!({
        "sampleRate": result.sample_rate,
        "channels": result.channels,
        "sampleCount": result.samples.len(),
        "wavBase64": result.encode_base64_wav(),
    }))
}

#[tauri::command]
async fn generate_finish_sentence(
    app: AppHandle,
    state: State<'_, AppState>,
    request: engine::FinishSentenceRequest,
) -> Result<serde_json::Value, String> {
    let engine = state.clone_engine()?;
    let queue = state.generation_queue.clone();
    let mut options = request.options.unwrap_or(serde_json::json!({}));
    let audio_wav = prepare_continuation_audio(&request.audio_path, &options)?;
    let stream_playback_enabled = option_bool(&options, "stream_playback");
    let stream_backend = engine.supports_streaming();
    set_bool_option(
        &mut options,
        "emit_stream_audio_chunks",
        stream_playback_enabled,
    );
    let progress = build_progress(&app);
    let audio_chunk = if stream_playback_enabled {
        build_audio_chunk(&app)
    } else {
        noop_audio_chunk()
    };

    let result = tauri::async_runtime::spawn_blocking(move || -> Result<_, EngineError> {
        let _queue = queue
            .lock()
            .map_err(|_| EngineError::Generation("generation queue poisoned".into()))?;
        if stream_backend {
            engine.generate_finish_sentence_stream(
                &audio_wav,
                request.continuation_text.as_deref(),
                &options,
                progress,
                audio_chunk,
            )
        } else {
            engine.generate_finish_sentence(
                &audio_wav,
                request.continuation_text.as_deref(),
                &options,
                progress,
            )
        }
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| map_engine_err(&e))?;

    Ok(serde_json::json!({
        "sampleRate": result.sample_rate,
        "channels": result.channels,
        "sampleCount": result.samples.len(),
        "wavBase64": result.encode_base64_wav(),
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcription (Parakeet — NVIDIA parakeet-tdt-0.6b-v3, multilingual incl. RU)
//
// ASR is decoupled from the C++ TTS engine and runs in-process via the
// `parakeet-rs` crate (ONNX Runtime, CPU). The model lives in a folder of ONNX
// files next to the exe (models/parakeet/<variant>). ONNX Runtime is loaded
// dynamically from `onnxruntime.dll` next to the exe (fetched on first use, like
// the engine DLLs). The v3 model auto-detects language — no RU/EN toggle.
// ═══════════════════════════════════════════════════════════════════════════

// Official ONNX Runtime 1.24.2 (matches the `ort` rc.12 build, OrtApi 24), from
// the microsoft/onnxruntime GitHub release. The `onnxruntime-win-x64-1.24.2.zip`
// archive carries just the CPU runtime; we extract its `lib/onnxruntime.dll`
// (~14 MB) next to the exe. Far smaller than the NuGet package (~124 MB) which
// bundles headers, the managed assembly and every RID's native blob.
const ONNXRUNTIME_ZIP_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-x64-1.24.2.zip";
// The DLL lives under `<archive-root>/lib/onnxruntime.dll`. The archive root is
// the same as the zip stem, but we match on the suffix so a renamed release
// asset still resolves.
const ONNXRUNTIME_DLL_ENTRY_SUFFIX: &str = "lib/onnxruntime.dll";

fn onnxruntime_dll_path() -> PathBuf {
    app_root_dir().join("onnxruntime.dll")
}

/// Ensure `onnxruntime.dll` sits next to the exe (portable) and `ORT_DYLIB_PATH`
/// points at it. Downloads + extracts it from the ONNX Runtime GitHub release on
/// first use. Idempotent and cheap once the DLL exists.
fn ensure_onnx_runtime() -> Result<(), String> {
    let dll = onnxruntime_dll_path();
    if dll.exists() {
        std::env::set_var("ORT_DYLIB_PATH", &dll);
        return Ok(());
    }

    let resp = ureq::get(ONNXRUNTIME_ZIP_URL)
        .call()
        .map_err(|e| format!("download onnxruntime runtime: {e}"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_body().into_reader(), &mut buf)
        .map_err(|e| format!("read onnxruntime archive: {e}"))?;

    let reader = std::io::Cursor::new(buf);
    let mut archive =
        zip::ZipArchive::new(reader).map_err(|e| format!("onnxruntime archive is not a zip: {e}"))?;

    // The DLL is under `<root>/lib/onnxruntime.dll`; match on the suffix so a
    // differently-named archive root still resolves.
    let entry_name = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().replace('\\', "/")))
        .find(|name| name.ends_with(ONNXRUNTIME_DLL_ENTRY_SUFFIX))
        .ok_or_else(|| format!("{ONNXRUNTIME_DLL_ENTRY_SUFFIX} missing in onnxruntime archive"))?;
    let mut entry = archive
        .by_name(&entry_name)
        .map_err(|e| format!("{entry_name} missing in archive: {e}"))?;

    let tmp = dll.with_extension("dll.tmp");
    {
        let mut out = std::fs::File::create(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("write onnxruntime.dll: {e}"))?;
    }
    std::fs::rename(&tmp, &dll).map_err(|e| format!("finalize onnxruntime.dll: {e}"))?;

    std::env::set_var("ORT_DYLIB_PATH", &dll);
    Ok(())
}

/// Resolve the ASR model directory from the path the frontend stored. This is a
/// folder holding the ONNX files (encoder/decoder + vocab.txt). For robustness
/// we accept either the folder itself or any file inside it (older stored paths
/// may point at a weight file), taking that file's parent directory.
fn resolve_asr_model_dir(stored: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(stored);
    let dir = if p.is_dir() {
        p
    } else if p.is_file() {
        p.parent()
            .map(|d| d.to_path_buf())
            .ok_or_else(|| "ASR model path has no parent directory".to_string())?
    } else {
        return Err(format!("ASR model folder not found: {stored}"));
    };
    if !dir.join("vocab.txt").exists() {
        return Err(format!(
            "ASR model folder is missing vocab.txt: {}",
            dir.display()
        ));
    }
    Ok(dir)
}

#[tauri::command]
async fn transcribe_audio(
    audio_path: String,
    whisper_model_path: Option<String>,
    language: Option<String>,
) -> Result<serde_json::Value, String> {
    let _ = language; // Parakeet v3 auto-detects the language.
    let model_path = whisper_model_path
        .filter(|p| !p.is_empty())
        .ok_or_else(|| "No ASR model set. Download the Parakeet model in Settings.".to_string())?;
    let model_dir = resolve_asr_model_dir(&model_path)?;
    let wav_path = audio::ensure_wav(&audio_path).map_err(|e| e.to_string())?;

    let text = tauri::async_runtime::spawn_blocking(move || {
        ensure_onnx_runtime()?;
        parakeet::transcribe(&model_dir, &wav_path)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))??;

    Ok(serde_json::json!({ "text": text }))
}

/// Drop the warm Parakeet model from RAM (reloaded lazily next time).
#[tauri::command]
fn unload_asr() {
    parakeet::unload();
}

// ─── Диаризация (Sortformer v2) — вкладка «Транскрипт и диаризация» ───────────

/// Sortformer v2 ONNX для диаризации. Скачивается один раз в models/sortformer
/// (пресетом системы моделей). Источник — altunenes/parakeet-rs (см. доки крейта).
const SORTFORMER_MODEL_URL: &str =
    "https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.onnx";
const SORTFORMER_MODEL_FILE: &str = "diar_streaming_sortformer_4spk-v2.onnx";

fn sortformer_model_path() -> PathBuf {
    user_models_root()
        .join("sortformer")
        .join(SORTFORMER_MODEL_FILE)
}

/// Есть ли скачанная Sortformer-модель (для UI: показать кнопку скачивания).
#[tauri::command]
fn sortformer_status() -> serde_json::Value {
    let path = sortformer_model_path();
    serde_json::json!({
        "installed": path.exists(),
        "path": path.to_string_lossy(),
    })
}

/// Скачать Sortformer v2 в models/sortformer (та же машина прогресса, что модели).
#[tauri::command]
async fn download_sortformer(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let dest_dir = user_models_root().join("sortformer");
    let control = state.download_control.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        download::download_file(
            SORTFORMER_MODEL_URL,
            &dest_dir,
            Some(SORTFORMER_MODEL_FILE),
            &app,
            control,
        )
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "path": result.path, "size": result.size }))
}

/// Полный пайплайн вкладки: декод -> (диаризация Sortformer, если модель есть и
/// `diarize=true`) -> транскрипция по репликам, иначе посегментная транскрипция
/// всего клипа. Прогресс-стадии эмитятся как события `transcript-progress`.
#[tauri::command]
async fn diarize_transcribe(
    app: AppHandle,
    audio_path: String,
    whisper_model_path: Option<String>,
    diarize: Option<bool>,
) -> Result<serde_json::Value, String> {
    let model_path = whisper_model_path
        .filter(|p| !p.is_empty())
        .ok_or_else(|| "Не выбрана ASR-модель. Скачайте Parakeet в настройках.".to_string())?;
    let model_dir = resolve_asr_model_dir(&model_path)?;
    let want_diarize = diarize.unwrap_or(true);

    let emit = |phase: &str| {
        let _ = app.emit("transcript-progress", serde_json::json!({ "phase": phase }));
    };

    emit("decode");
    let wav_path = audio::ensure_wav(&audio_path).map_err(|e| e.to_string())?;

    let sortformer = sortformer_model_path();
    let have_sortformer = want_diarize && sortformer.exists();

    let phase_app = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        ensure_onnx_runtime()?;
        let sf: Option<&std::path::Path> = if have_sortformer {
            let _ = phase_app.emit(
                "transcript-progress",
                serde_json::json!({ "phase": "diarize" }),
            );
            Some(sortformer.as_path())
        } else {
            None
        };
        let _ = phase_app.emit(
            "transcript-progress",
            serde_json::json!({ "phase": "transcribe" }),
        );
        parakeet::transcribe_and_diarize(&model_dir, sf, &wav_path)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))??;

    emit("done");
    Ok(serde_json::json!({
        "segments": result.segments,
        "nSpeakers": result.n_speakers,
        "diarized": have_sortformer,
    }))
}

/// «Сделать голос»: собрать реф-клип спикера из отобранных сегментов транскрипта
/// и записать его временным WAV. Возвращает путь к wav + объединённый ref_text —
/// фронт заводит по ним персону тем же механизмом, что импорт войспака.
#[tauri::command]
async fn build_speaker_voice(
    audio_path: String,
    segments: Vec<parakeet::SpeakerSegment>,
    speaker: i32,
    speaker_name: String,
) -> Result<serde_json::Value, String> {
    let wav_path = audio::ensure_wav(&audio_path).map_err(|e| e.to_string())?;
    let (wav_bytes, ref_text) = tauri::async_runtime::spawn_blocking(move || {
        // до ~30с реф под Higgs-клон, паузы ~0.3с между кусками.
        parakeet::build_speaker_reference(&wav_path, &segments, speaker, 30.0, 0.3)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))??;

    // Записать во временный файл (фронт скопирует его в папку персоны).
    let safe: String = speaker_name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("higgs_voice_{safe}_{stamp}.wav"));
    std::fs::write(&tmp, &wav_bytes).map_err(|e| format!("write voice wav: {e}"))?;

    Ok(serde_json::json!({
        "path": tmp.to_string_lossy(),
        "refText": ref_text,
    }))
}

/// Level-match a generated WAV clip to a target integrated loudness (LUFS).
#[tauri::command]
fn normalize_wav(wav_base64: String, target_lufs: Option<f64>) -> Result<String, String> {
    let bytes = base64_decode(&wav_base64).map_err(|e| e.to_string())?;
    let out = audio::normalize_wav_bytes(&bytes, target_lufs.unwrap_or(-20.0))?;
    Ok(base64_encode(&out))
}

// ═══════════════════════════════════════════════════════════════════════════
// Model listing / download
// ═══════════════════════════════════════════════════════════════════════════

fn model_listing_candidates() -> Vec<PathBuf> {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    vec![
        exe.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("models"))
            .unwrap_or_else(|| PathBuf::from("models")),
        user_models_root(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("models"),
    ]
}

fn list_all_models() -> Vec<download::ModelListing> {
    let mut seen = HashSet::new();
    let mut listings = Vec::new();
    for root in model_listing_candidates() {
        if !root.exists() {
            continue;
        }
        for model in download::list_model_dirs(&root) {
            if seen.insert(model.path.clone()) {
                listings.push(model);
            }
        }
    }
    listings.sort_by(|a, b| a.name.cmp(&b.name));
    listings
}

#[tauri::command]
fn list_models(_state: State<'_, AppState>) -> Vec<download::ModelListing> {
    list_all_models()
}

#[tauri::command]
async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    request: download::DownloadRequest,
) -> Result<serde_json::Value, String> {
    let dest_dir = resolve_download_dest_dir(&request.dest_dir);
    let control = state.download_control.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        download::download_file(
            &request.url,
            &dest_dir,
            request.filename.as_deref(),
            &app,
            control,
        )
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "path": result.path, "size": result.size }))
}

#[tauri::command]
fn download_control(
    state: State<'_, AppState>,
    action: String,
) -> Result<serde_json::Value, String> {
    match action.as_str() {
        "pause" => state.download_control.pause(),
        "resume" => state.download_control.resume(),
        "cancel" | "stop" => state.download_control.cancel(),
        _ => return Err(format!("unknown download action: {action}")),
    }
    Ok(serde_json::json!({
        "active": state.download_control.is_active(),
        "paused": state.download_control.is_paused(),
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Local API server
// ═══════════════════════════════════════════════════════════════════════════

struct HttpRequest {
    method: String,
    route: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn emit_api_log(
    app: &AppHandle,
    level: &str,
    kind: &str,
    method: &str,
    route: &str,
    status: u16,
    latency_ms: u128,
    message: &str,
) {
    let _ = app.emit(
        "api-log",
        ApiLogEvent {
            level: level.to_string(),
            kind: kind.to_string(),
            method: method.to_string(),
            route: route.to_string(),
            status,
            latency_ms,
            message: message.to_string(),
            job_id: String::new(),
        },
    );
}

fn emit_native_diagnostic(app: &AppHandle, phase: &str) -> bool {
    let Some(message) = phase.strip_prefix("diag:") else {
        return false;
    };
    emit_api_log(
        app,
        "info",
        "engine",
        "VRAM",
        "/native",
        0,
        0,
        message.trim(),
    );
    true
}

fn http_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "OK",
    }
}

fn write_http_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        status,
        http_reason(status),
        content_type,
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}

fn json_response(value: serde_json::Value) -> (u16, &'static str, Vec<u8>, String) {
    (
        200,
        "application/json",
        serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec()),
        "ok".to_string(),
    )
}

fn json_error(
    status: u16,
    code: &str,
    message: impl Into<String>,
) -> (u16, &'static str, Vec<u8>, String) {
    let message = message.into();
    (
        status,
        "application/json",
        serde_json::to_vec(&serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        }))
        .unwrap_or_else(|_| b"{\"error\":{\"message\":\"error\"}}".to_vec()),
        message,
    )
}

fn api_response_format(body: &serde_json::Value) -> Result<&'static str, String> {
    let format = body
        .get("response_format")
        .and_then(|v| v.as_str())
        .unwrap_or("wav")
        .trim()
        .to_ascii_lowercase();
    match format.as_str() {
        "wav" => Ok("wav"),
        "mp3" => Ok("mp3"),
        _ => Err("response_format must be wav or mp3".to_string()),
    }
}

fn encode_api_audio_response(
    audio: &AudioResult,
    format: &str,
) -> Result<(&'static str, Vec<u8>), String> {
    match format {
        "wav" => Ok(("audio/wav", audio.encode_pcm16_wav())),
        "mp3" => audio.encode_mp3().map(|bytes| ("audio/mpeg", bytes)),
        _ => Err("response_format must be wav or mp3".to_string()),
    }
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut header_end = None;
    let mut content_length = 0usize;

    loop {
        let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if header_end.is_none() {
            header_end = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
            if let Some(end) = header_end {
                let headers = String::from_utf8_lossy(&buf[..end]);
                for line in headers.lines() {
                    if let Some((name, value)) = line.split_once(':') {
                        if name.trim().eq_ignore_ascii_case("content-length") {
                            content_length = value.trim().parse::<usize>().unwrap_or(0);
                        }
                    }
                }
            }
        }
        if let Some(end) = header_end {
            if buf.len() >= end + content_length {
                break;
            }
        }
        if buf.len() > 64 * 1024 * 1024 {
            return Err("request too large".into());
        }
    }

    let end = header_end.ok_or_else(|| "missing HTTP headers".to_string())?;
    let header_text = String::from_utf8_lossy(&buf[..end]);
    let mut lines = header_text.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_uppercase();
    let route = parts
        .next()
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    if method.is_empty() {
        return Err("missing HTTP method".into());
    }
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    let body_end = std::cmp::min(buf.len(), end + content_length);
    Ok(HttpRequest {
        method,
        route,
        headers,
        body: buf[end..body_end].to_vec(),
    })
}

fn header_value<'a>(req: &'a HttpRequest, name: &str) -> Option<&'a str> {
    let name = name.to_ascii_lowercase();
    req.headers
        .iter()
        .find(|(key, _)| key == &name)
        .map(|(_, value)| value.as_str())
}

fn api_authorized(req: &HttpRequest, api_key: &str) -> bool {
    header_value(req, "authorization")
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|token| token == api_key)
        .unwrap_or(false)
}

fn api_engine_status(engine_state: &Arc<Mutex<Option<Arc<Engine>>>>) -> serde_json::Value {
    let guard = engine_state.lock().unwrap();
    let engine_loaded = guard.is_some();
    let model_loaded = guard.as_ref().map(|e| e.is_model_loaded()).unwrap_or(false);
    let generating = guard.as_ref().map(|e| e.is_generating()).unwrap_or(false);
    let supports_streaming = guard
        .as_ref()
        .map(|e| e.supports_streaming())
        .unwrap_or(false);
    serde_json::json!({
        "engineLoaded": engine_loaded,
        "modelLoaded": model_loaded,
        "generating": generating,
        "supportsStreaming": supports_streaming,
    })
}

fn api_find_speaker<'a>(
    speakers: &'a [ApiSpeakerPersona],
    key: &str,
) -> Option<&'a ApiSpeakerPersona> {
    let key_lc = key.trim().to_lowercase();
    speakers
        .iter()
        .find(|speaker| speaker.id == key || speaker.name.to_lowercase() == key_lc)
}

fn api_current_speakers(
    speakers: &Arc<Mutex<Vec<ApiSpeakerPersona>>>,
) -> Vec<ApiSpeakerPersona> {
    speakers
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

fn api_clone_engine(engine_state: &Arc<Mutex<Option<Arc<Engine>>>>) -> Result<Arc<Engine>, String> {
    engine_state
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "engine not loaded".to_string())
}

fn api_generation_options(root: &serde_json::Value) -> serde_json::Value {
    let mut options = serde_json::Map::new();
    if let Some(higgs) = root.get("higgs").and_then(|v| v.as_object()) {
        for (key, value) in higgs {
            options.insert(key.clone(), value.clone());
        }
    }
    for key in [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "normalize_reference",
        "reference_cache_path",
    ] {
        if let Some(value) = root.get(key) {
            options.insert(key.to_string(), value.clone());
        }
    }
    serde_json::Value::Object(options)
}

fn api_job_label(text: &str, fallback: &str) -> String {
    let label: String = text.trim().chars().take(56).collect();
    if label.is_empty() {
        fallback.to_string()
    } else {
        label
    }
}

fn api_job_progress(
    app: &AppHandle,
    job_id: &str,
    workflow: &str,
    label: &str,
) -> ProgressCallback {
    let app = app.clone();
    let job_id = job_id.to_string();
    let workflow = workflow.to_string();
    let label = label.to_string();
    Arc::new(move |current, total, phase| {
        if emit_native_diagnostic(&app, phase) {
            return;
        }
        emit_studio_job(
            &app,
            &job_id,
            "api",
            &workflow,
            "generating",
            &label,
            phase,
            Some(current),
            Some(total),
            phase,
        );
    })
}

fn api_stream_options(mut options: serde_json::Value) -> serde_json::Value {
    set_bool_option(&mut options, "emit_stream_audio_chunks", false);
    options
}

fn api_generate_tts(
    engine: &Arc<Engine>,
    text: &str,
    options: &serde_json::Value,
    progress: ProgressCallback,
) -> Result<AudioResult, EngineError> {
    if engine.supports_streaming() {
        engine.generate_tts_stream(text, options, progress, noop_audio_chunk())
    } else {
        engine.generate_tts(text, options, progress)
    }
}

fn api_generate_voice_clone(
    engine: &Arc<Engine>,
    text: &str,
    ref_wav: &str,
    ref_text: Option<&str>,
    options: &serde_json::Value,
    progress: ProgressCallback,
) -> Result<AudioResult, EngineError> {
    if engine.supports_streaming() {
        engine.generate_voice_clone_stream(
            text,
            ref_wav,
            ref_text,
            options,
            progress,
            noop_audio_chunk(),
        )
    } else {
        engine.generate_voice_clone(text, ref_wav, ref_text, options, progress)
    }
}

fn api_generate_finish_sentence(
    engine: &Arc<Engine>,
    audio_wav: &str,
    continuation: Option<&str>,
    options: &serde_json::Value,
    progress: ProgressCallback,
) -> Result<AudioResult, EngineError> {
    if engine.supports_streaming() {
        engine.generate_finish_sentence_stream(
            audio_wav,
            continuation,
            options,
            progress,
            noop_audio_chunk(),
        )
    } else {
        engine.generate_finish_sentence(audio_wav, continuation, options, progress)
    }
}

fn write_streaming_response_header(stream: &mut TcpStream) -> Result<(), String> {
    let header = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: application/x-ndjson\r\n",
        "Cache-Control: no-cache\r\n",
        "X-Accel-Buffering: no\r\n",
        "Access-Control-Allow-Origin: *\r\n",
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n",
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
        "Connection: close\r\n\r\n"
    );
    stream.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_ndjson_event(
    writer: &Arc<Mutex<TcpStream>>,
    value: serde_json::Value,
) -> Result<(), String> {
    let mut bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    let mut stream = writer.lock().map_err(|_| "stream writer poisoned".to_string())?;
    stream.write_all(&bytes).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn api_stream_audio_callback(writer: Arc<Mutex<TcpStream>>) -> AudioChunkCallback {
    Arc::new(
        move |sample_rate: i32,
              channels: i32,
              start_sample: i64,
              samples: &[f32],
              is_final: bool| {
            let wav_base64 = AudioResult {
                sample_rate,
                channels,
                samples: samples.to_vec(),
            }
            .encode_base64_wav();
            let _ = write_ndjson_event(
                &writer,
                serde_json::json!({
                    "event": "audio",
                    "sampleRate": sample_rate,
                    "channels": channels,
                    "startSample": start_sample,
                    "sampleCount": samples.len(),
                    "isFinalChunk": is_final,
                    "encoding": "wav-base64",
                    "wavBase64": wav_base64,
                }),
            );
        },
    )
}

fn api_stream_progress_callback(
    app: &AppHandle,
    writer: Arc<Mutex<TcpStream>>,
    job_id: &str,
    workflow: &str,
    label: &str,
) -> ProgressCallback {
    let app = app.clone();
    let job_id = job_id.to_string();
    let workflow = workflow.to_string();
    let label = label.to_string();
    Arc::new(move |current, total, phase| {
        if emit_native_diagnostic(&app, phase) {
            return;
        }
        let _ = write_ndjson_event(
            &writer,
            serde_json::json!({
                "event": "progress",
                "current": current,
                "total": total,
                "phase": phase,
            }),
        );
        emit_studio_job(
            &app,
            &job_id,
            "api",
            &workflow,
            "generating",
            &label,
            phase,
            Some(current),
            Some(total),
            phase,
        );
    })
}

fn handle_api_stream_request(
    stream: &mut TcpStream,
    app: &AppHandle,
    req: &HttpRequest,
    engine_state: Arc<Mutex<Option<Arc<Engine>>>>,
    generation_queue: Arc<Mutex<()>>,
    api_key: &str,
    speakers: Arc<Mutex<Vec<ApiSpeakerPersona>>>,
) -> (u16, String) {
    if req.method != "POST" {
        let (status, content_type, body, message) =
            json_error(404, "not_found", "route not found");
        write_http_response(stream, status, content_type, &body);
        return (status, message);
    }
    if !api_authorized(req, api_key) {
        let (status, content_type, body, message) =
            json_error(401, "unauthorized", "invalid or missing local API key");
        write_http_response(stream, status, content_type, &body);
        return (status, message);
    }
    let speaker_list = api_current_speakers(&speakers);

    let body: serde_json::Value = match serde_json::from_slice(&req.body) {
        Ok(value) => value,
        Err(e) => {
            let (status, content_type, body, message) =
                json_error(400, "invalid_json", e.to_string());
            write_http_response(stream, status, content_type, &body);
            return (status, message);
        }
    };
    let format = match api_response_format(&body) {
        Ok(format) => format,
        Err(e) => {
        let (status, content_type, body, message) = json_error(
            400,
            "format_not_supported",
            e,
        );
        write_http_response(stream, status, content_type, &body);
        return (status, message);
        }
    };

    let engine = match api_clone_engine(&engine_state) {
        Ok(engine) => engine,
        Err(e) => {
            let (status, content_type, body, message) = json_error(409, "engine_not_loaded", e);
            write_http_response(stream, status, content_type, &body);
            return (status, message);
        }
    };
    if !engine.supports_streaming() {
        let (status, content_type, body, message) = json_error(
            409,
            "streaming_not_supported",
            "loaded Higgs engine DLL does not expose streaming generation",
        );
        write_http_response(stream, status, content_type, &body);
        return (status, message);
    }

    let writer_stream = match stream.try_clone() {
        Ok(stream) => stream,
        Err(e) => {
            let (status, content_type, body, message) =
                json_error(500, "stream_clone_failed", e.to_string());
            write_http_response(stream, status, content_type, &body);
            return (status, message);
        }
    };
    if let Err(e) = write_streaming_response_header(stream) {
        return (500, e);
    }
    let writer = Arc::new(Mutex::new(writer_stream));

    let mode = body
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let voice = body
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let text = body
        .get("input")
        .or_else(|| body.get("continuation_text"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if text.is_empty() && mode != "continue" && mode != "continue_speech" {
        let _ = write_ndjson_event(
            &writer,
            serde_json::json!({
                "event": "error",
                "error": {"code": "missing_input", "message": "input is required"},
            }),
        );
        return (400, "input is required".into());
    }

    let job_id = next_studio_job_id("api-stream");
    let mut options = api_generation_options(&body);
    set_bool_option(&mut options, "emit_stream_audio_chunks", true);
    set_bool_option(&mut options, "stream_playback", true);

    let workflow: String;
    let label: String;
    enum StreamRequestKind {
        Tts { text: String },
        VoiceClone {
            text: String,
            ref_wav: String,
            ref_text: Option<String>,
        },
        Finish {
            audio_wav: String,
            continuation: Option<String>,
        },
    }

    let request_kind = if mode == "continue" || mode == "continue_speech" || body.get("audio_path").is_some() {
        let audio_path = body
            .get("audio_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if audio_path.is_empty() {
            let _ = write_ndjson_event(
                &writer,
                serde_json::json!({
                    "event": "error",
                    "error": {"code": "missing_audio_path", "message": "audio_path is required"},
                }),
            );
            return (400, "audio_path is required".into());
        }
        let audio_wav = match prepare_continuation_audio(&audio_path, &options) {
            Ok(path) => path,
            Err(e) => {
                let _ = write_ndjson_event(
                    &writer,
                    serde_json::json!({
                        "event": "error",
                        "error": {"code": "invalid_audio", "message": e},
                    }),
                );
                return (400, "invalid audio".into());
            }
        };
        workflow = "finish".into();
        label = api_job_label(&text, "Continue speech");
        StreamRequestKind::Finish {
            audio_wav,
            continuation: body
                .get("continuation_text")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string()),
        }
    } else if voice != "default" || body.get("reference_audio_path").is_some() {
        let (ref_path, ref_text, speaker_name) = if voice != "default" {
            let speaker_key = voice.strip_prefix("speaker:").unwrap_or(voice).trim();
            let speaker = match api_find_speaker(speaker_list.as_slice(), speaker_key) {
                Some(speaker) => speaker,
                None => {
                    let message = format!("saved speaker identity not found: {}", speaker_key);
                    let _ = write_ndjson_event(
                        &writer,
                        serde_json::json!({
                            "event": "error",
                            "error": {"code": "speaker_not_found", "message": message},
                        }),
                    );
                    return (404, "saved speaker identity not found".into());
                }
            };
            if speaker.normalize && options.get("normalize_reference").is_none() {
                set_bool_option(&mut options, "normalize_reference", true);
            }
            set_string_option(&mut options, "reference_cache_path", &speaker.cache_path);
            (
                speaker.ref_path.clone(),
                if speaker.ref_text.trim().is_empty() {
                    None
                } else {
                    Some(speaker.ref_text.clone())
                },
                speaker.name.clone(),
            )
        } else {
            (
                body.get("reference_audio_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                body.get("reference_text")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string()),
                "Voice clone".to_string(),
            )
        };
        if ref_path.trim().is_empty() {
            let _ = write_ndjson_event(
                &writer,
                serde_json::json!({
                    "event": "error",
                    "error": {"code": "missing_reference_audio", "message": "reference_audio_path or saved speaker voice is required"},
                }),
            );
            return (400, "reference audio is required".into());
        }
        let ref_wav = match prepare_voice_reference_audio(&ref_path, &options) {
            Ok(path) => path,
            Err(e) => {
                let _ = write_ndjson_event(
                    &writer,
                    serde_json::json!({
                        "event": "error",
                        "error": {"code": "invalid_reference_audio", "message": e},
                    }),
                );
                return (400, "invalid reference audio".into());
            }
        };
        workflow = "voice_clone".into();
        label = format!("{} - {}", speaker_name, api_job_label(&text, "Voice clone"));
        StreamRequestKind::VoiceClone {
            text,
            ref_wav,
            ref_text,
        }
    } else {
        workflow = "tts".into();
        label = api_job_label(&text, "Plain TTS");
        StreamRequestKind::Tts { text }
    };

    emit_studio_job(
        app,
        &job_id,
        "api",
        &workflow,
        "queued",
        &label,
        "queued",
        None,
        None,
        "Waiting for generation slot",
    );
    let _ = write_ndjson_event(
        &writer,
        serde_json::json!({
            "event": "queued",
            "jobId": &job_id,
            "workflow": &workflow,
            "label": &label,
            "audioEncoding": "wav-base64",
            "finalEncoding": format!("{}-base64", format),
        }),
    );

    let _queue = match generation_queue.lock() {
        Ok(queue) => queue,
        Err(_) => {
            let _ = write_ndjson_event(
                &writer,
                serde_json::json!({
                    "event": "error",
                    "error": {"code": "queue_error", "message": "generation queue poisoned"},
                }),
            );
            return (500, "generation queue poisoned".into());
        }
    };

    emit_studio_job(
        app,
        &job_id,
        "api",
        &workflow,
        "generating",
        &label,
        "preparing",
        Some(0),
        Some(1),
        "Preparing stream",
    );
    let _ = write_ndjson_event(
        &writer,
        serde_json::json!({
            "event": "start",
            "jobId": &job_id,
            "workflow": &workflow,
            "label": &label,
        }),
    );

    let progress = api_stream_progress_callback(app, writer.clone(), &job_id, &workflow, &label);
    let audio_chunk = api_stream_audio_callback(writer.clone());
    let result = match request_kind {
        StreamRequestKind::Tts { text } => engine.generate_tts_stream(&text, &options, progress, audio_chunk),
        StreamRequestKind::VoiceClone {
            text,
            ref_wav,
            ref_text,
        } => engine.generate_voice_clone_stream(
            &text,
            &ref_wav,
            ref_text.as_deref(),
            &options,
            progress,
            audio_chunk,
        ),
        StreamRequestKind::Finish {
            audio_wav,
            continuation,
        } => engine.generate_finish_sentence_stream(
            &audio_wav,
            continuation.as_deref(),
            &options,
            progress,
            audio_chunk,
        ),
    };

    match result {
        Ok(audio) => {
            emit_studio_job(
                app,
                &job_id,
                "api",
                &workflow,
                "complete",
                &label,
                "complete",
                Some(1),
                Some(1),
                "Complete",
            );
            match encode_api_audio_response(&audio, format) {
                Ok((_content_type, final_bytes)) => {
                    let mut payload = serde_json::json!({
                    "event": "final",
                    "jobId": &job_id,
                    "sampleRate": audio.sample_rate,
                    "channels": audio.channels,
                    "sampleCount": audio.samples.len(),
                        "encoding": format!("{}-base64", format),
                    });
                    if let Some(map) = payload.as_object_mut() {
                        let key = if format == "mp3" { "mp3Base64" } else { "wavBase64" };
                        map.insert(key.into(), serde_json::json!(base64_encode(&final_bytes)));
                    }
                    let _ = write_ndjson_event(&writer, payload);
                    let _ = write_ndjson_event(
                        &writer,
                        serde_json::json!({ "event": "done", "jobId": &job_id }),
                    );
                    (200, "stream complete".into())
                }
                Err(e) => {
                    let _ = write_ndjson_event(
                        &writer,
                        serde_json::json!({
                            "event": "error",
                            "jobId": &job_id,
                            "error": {"code": "audio_encode_failed", "message": e},
                        }),
                    );
                    (500, "audio encode failed".into())
                }
            }
        }
        Err(e) => {
            let message = map_engine_err(&e);
            emit_studio_job(
                app,
                &job_id,
                "api",
                &workflow,
                if message.to_lowercase().contains("cancel") {
                    "cancelled"
                } else {
                    "failed"
                },
                &label,
                "failed",
                None,
                None,
                &message,
            );
            let _ = write_ndjson_event(
                &writer,
                serde_json::json!({
                    "event": "error",
                    "jobId": &job_id,
                    "error": {"code": "generation_failed", "message": message},
                }),
            );
            (500, "generation failed".into())
        }
    }
}

fn handle_api_request(
    app: &AppHandle,
    req: &HttpRequest,
    engine_state: Arc<Mutex<Option<Arc<Engine>>>>,
    generation_queue: Arc<Mutex<()>>,
    api_key: &str,
    speakers: Arc<Mutex<Vec<ApiSpeakerPersona>>>,
) -> (u16, &'static str, Vec<u8>, String) {
    if req.method == "OPTIONS" {
        return (204, "text/plain", Vec::new(), "preflight".to_string());
    }

    if req.route == "/health" {
        return json_response(serde_json::json!({
            "ok": true,
            "service": "Higgs Audio v3 Studio",
        }));
    }

    if !api_authorized(req, api_key) {
        return json_error(401, "unauthorized", "invalid or missing local API key");
    }
    let speaker_list = api_current_speakers(&speakers);

    match (req.method.as_str(), req.route.as_str()) {
        ("GET", "/v1/status") => {
            let mut status = api_engine_status(&engine_state);
            if let Some(map) = status.as_object_mut() {
                map.insert("speakerCount".into(), serde_json::json!(speaker_list.len()));
            }
            json_response(status)
        }
        ("GET", "/v1/models") => json_response(serde_json::json!({ "data": list_all_models() })),
        ("GET", "/v1/higgs/speakers") => json_response(serde_json::json!({
            "data": speaker_list.iter().map(|speaker| {
                let has_cache = !speaker.cache_path.trim().is_empty()
                    && PathBuf::from(&speaker.cache_path).is_file();
                serde_json::json!({
                    "id": speaker.id,
                    "name": speaker.name,
                    "voice": format!("speaker:{}", speaker.id),
                    "hasReferenceAudio": !speaker.ref_path.trim().is_empty(),
                    "hasTranscript": !speaker.ref_text.trim().is_empty(),
                    "hasCache": has_cache,
                    "normalizeReference": speaker.normalize,
                })
            }).collect::<Vec<_>>()
        })),
        ("POST", "/v1/higgs/cancel") => match api_clone_engine(&engine_state) {
            Ok(engine) => {
                engine.cancel();
                json_response(serde_json::json!({ "cancelled": true }))
            }
            Err(e) => json_error(409, "engine_not_loaded", e),
        },
        ("POST", "/v1/audio/speech") => {
            let body: serde_json::Value = match serde_json::from_slice(&req.body) {
                Ok(value) => value,
                Err(e) => return json_error(400, "invalid_json", e.to_string()),
            };
            let text = body
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                return json_error(400, "missing_input", "input is required");
            }
            let format = match api_response_format(&body) {
                Ok(format) => format,
                Err(e) => return json_error(400, "format_not_supported", e),
            };
            let voice = body
                .get("voice")
                .and_then(|v| v.as_str())
                .unwrap_or("default");
            let engine = match api_clone_engine(&engine_state) {
                Ok(engine) => engine,
                Err(e) => return json_error(409, "engine_not_loaded", e),
            };
            if voice != "default" {
                let speaker_key = voice.strip_prefix("speaker:").unwrap_or(voice).trim();
                let speaker = match api_find_speaker(speaker_list.as_slice(), speaker_key) {
                    Some(speaker) => speaker,
                    None => {
                        return json_error(
                            404,
                            "speaker_not_found",
                            format!("saved speaker identity not found: {}", speaker_key),
                        )
                    }
                };
                if speaker.ref_path.trim().is_empty() {
                    return json_error(
                        409,
                        "speaker_missing_reference",
                        format!("saved speaker '{}' has no reference audio", speaker.name),
                    );
                }
                let mut api_options = api_generation_options(&body);
                if speaker.normalize && api_options.get("normalize_reference").is_none() {
                    if let Some(map) = api_options.as_object_mut() {
                        map.insert("normalize_reference".into(), serde_json::json!(true));
                    }
                }
                set_string_option(&mut api_options, "reference_cache_path", &speaker.cache_path);
                let api_options = api_stream_options(api_options);
                let ref_wav = match prepare_voice_reference_audio(&speaker.ref_path, &api_options) {
                    Ok(path) => path,
                    Err(e) => return json_error(400, "invalid_reference_audio", e.to_string()),
                };
                let ref_text = if speaker.ref_text.trim().is_empty() {
                    None
                } else {
                    Some(speaker.ref_text.as_str())
                };
                let job_id = next_studio_job_id("api");
                let label = format!("{} - {}", speaker.name, api_job_label(&text, "Voice clone"));
                emit_studio_job(
                    app,
                    &job_id,
                    "api",
                    "voice_clone",
                    "queued",
                    &label,
                    "queued",
                    None,
                    None,
                    "Waiting for generation slot",
                );
                let _queue = match generation_queue.lock() {
                    Ok(queue) => {
                        emit_studio_job(
                            app,
                            &job_id,
                            "api",
                            "voice_clone",
                            "generating",
                            &label,
                            "preparing",
                            Some(0),
                            Some(1),
                            "Preparing voice clone",
                        );
                        queue
                    }
                    Err(_) => {
                        emit_studio_job(
                            app,
                            &job_id,
                            "api",
                            "voice_clone",
                            "failed",
                            &label,
                            "failed",
                            None,
                            None,
                            "generation queue poisoned",
                        );
                        return json_error(500, "queue_error", "generation queue poisoned");
                    }
                };
                return match api_generate_voice_clone(
                    &engine,
                    &text,
                    &ref_wav,
                    ref_text,
                    &api_options,
                    api_job_progress(app, &job_id, "voice_clone", &label),
                ) {
                    Ok(audio) => {
                        emit_studio_job(
                            app,
                            &job_id,
                            "api",
                            "voice_clone",
                            "complete",
                            &label,
                            "complete",
                            Some(1),
                            Some(1),
                            "Complete",
                        );
                        match encode_api_audio_response(&audio, format) {
                            Ok((content_type, bytes)) => (
                                200,
                                content_type,
                                bytes,
                                format!("speech generated with speaker {}", speaker.name),
                            ),
                            Err(e) => json_error(500, "audio_encode_failed", e),
                        }
                    }
                    Err(e) => {
                        let message = map_engine_err(&e);
                        emit_studio_job(
                            app,
                            &job_id,
                            "api",
                            "voice_clone",
                            if message.to_lowercase().contains("cancel") {
                                "cancelled"
                            } else {
                                "failed"
                            },
                            &label,
                            "failed",
                            None,
                            None,
                            &message,
                        );
                        json_error(500, "generation_failed", message)
                    }
                };
            }
            let api_options = api_stream_options(api_generation_options(&body));
            let job_id = next_studio_job_id("api");
            let label = api_job_label(&text, "Plain TTS");
            emit_studio_job(
                app,
                &job_id,
                "api",
                "tts",
                "queued",
                &label,
                "queued",
                None,
                None,
                "Waiting for generation slot",
            );
            let _queue = match generation_queue.lock() {
                Ok(queue) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "tts",
                        "generating",
                        &label,
                        "preparing",
                        Some(0),
                        Some(1),
                        "Preparing TTS",
                    );
                    queue
                }
                Err(_) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "tts",
                        "failed",
                        &label,
                        "failed",
                        None,
                        None,
                        "generation queue poisoned",
                    );
                    return json_error(500, "queue_error", "generation queue poisoned");
                }
            };
            match api_generate_tts(
                &engine,
                &text,
                &api_options,
                api_job_progress(app, &job_id, "tts", &label),
            ) {
                Ok(audio) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "tts",
                        "complete",
                        &label,
                        "complete",
                        Some(1),
                        Some(1),
                        "Complete",
                    );
                    match encode_api_audio_response(&audio, format) {
                        Ok((content_type, bytes)) => (
                            200,
                            content_type,
                            bytes,
                            "speech generated".to_string(),
                        ),
                        Err(e) => json_error(500, "audio_encode_failed", e),
                    }
                }
                Err(e) => {
                    let message = map_engine_err(&e);
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "tts",
                        if message.to_lowercase().contains("cancel") {
                            "cancelled"
                        } else {
                            "failed"
                        },
                        &label,
                        "failed",
                        None,
                        None,
                        &message,
                    );
                    json_error(500, "generation_failed", message)
                }
            }
        }
        ("POST", "/v1/higgs/voice-clone") => {
            let body: serde_json::Value = match serde_json::from_slice(&req.body) {
                Ok(value) => value,
                Err(e) => return json_error(400, "invalid_json", e.to_string()),
            };
            let text = body
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let ref_path = body
                .get("reference_audio_path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if text.is_empty() || ref_path.is_empty() {
                return json_error(
                    400,
                    "missing_clone_input",
                    "input and reference_audio_path are required",
                );
            }
            let format = match api_response_format(&body) {
                Ok(format) => format,
                Err(e) => return json_error(400, "format_not_supported", e),
            };
            let ref_text = body.get("reference_text").and_then(|v| v.as_str());
            let api_options = api_stream_options(api_generation_options(&body));
            let ref_wav = match prepare_voice_reference_audio(&ref_path, &api_options) {
                Ok(path) => path,
                Err(e) => return json_error(400, "invalid_reference_audio", e.to_string()),
            };
            let engine = match api_clone_engine(&engine_state) {
                Ok(engine) => engine,
                Err(e) => return json_error(409, "engine_not_loaded", e),
            };
            let job_id = next_studio_job_id("api");
            let label = api_job_label(&text, "Voice clone");
            emit_studio_job(
                app,
                &job_id,
                "api",
                "voice_clone",
                "queued",
                &label,
                "queued",
                None,
                None,
                "Waiting for generation slot",
            );
            let _queue = match generation_queue.lock() {
                Ok(queue) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "voice_clone",
                        "generating",
                        &label,
                        "preparing",
                        Some(0),
                        Some(1),
                        "Preparing voice clone",
                    );
                    queue
                }
                Err(_) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "voice_clone",
                        "failed",
                        &label,
                        "failed",
                        None,
                        None,
                        "generation queue poisoned",
                    );
                    return json_error(500, "queue_error", "generation queue poisoned");
                }
            };
            match api_generate_voice_clone(
                &engine,
                &text,
                &ref_wav,
                ref_text,
                &api_options,
                api_job_progress(app, &job_id, "voice_clone", &label),
            ) {
                Ok(audio) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "voice_clone",
                        "complete",
                        &label,
                        "complete",
                        Some(1),
                        Some(1),
                        "Complete",
                    );
                    match encode_api_audio_response(&audio, format) {
                        Ok((content_type, bytes)) => (
                            200,
                            content_type,
                            bytes,
                            "voice clone generated".to_string(),
                        ),
                        Err(e) => json_error(500, "audio_encode_failed", e),
                    }
                }
                Err(e) => {
                    let message = map_engine_err(&e);
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "voice_clone",
                        if message.to_lowercase().contains("cancel") {
                            "cancelled"
                        } else {
                            "failed"
                        },
                        &label,
                        "failed",
                        None,
                        None,
                        &message,
                    );
                    json_error(500, "generation_failed", message)
                }
            }
        }
        ("POST", "/v1/higgs/continue-speech") => {
            let body: serde_json::Value = match serde_json::from_slice(&req.body) {
                Ok(value) => value,
                Err(e) => return json_error(400, "invalid_json", e.to_string()),
            };
            let audio_path = body
                .get("audio_path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if audio_path.is_empty() {
                return json_error(400, "missing_audio_path", "audio_path is required");
            }
            let format = match api_response_format(&body) {
                Ok(format) => format,
                Err(e) => return json_error(400, "format_not_supported", e),
            };
            let continuation = body.get("continuation_text").and_then(|v| v.as_str());
            let api_options = api_stream_options(api_generation_options(&body));
            let audio_wav = match prepare_continuation_audio(&audio_path, &api_options) {
                Ok(path) => path,
                Err(e) => return json_error(400, "invalid_audio", e.to_string()),
            };
            let engine = match api_clone_engine(&engine_state) {
                Ok(engine) => engine,
                Err(e) => return json_error(409, "engine_not_loaded", e),
            };
            let job_id = next_studio_job_id("api");
            let label = api_job_label(continuation.unwrap_or(""), "Continue speech");
            emit_studio_job(
                app,
                &job_id,
                "api",
                "finish",
                "queued",
                &label,
                "queued",
                None,
                None,
                "Waiting for generation slot",
            );
            let _queue = match generation_queue.lock() {
                Ok(queue) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "finish",
                        "generating",
                        &label,
                        "preparing",
                        Some(0),
                        Some(1),
                        "Preparing continuation",
                    );
                    queue
                }
                Err(_) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "finish",
                        "failed",
                        &label,
                        "failed",
                        None,
                        None,
                        "generation queue poisoned",
                    );
                    return json_error(500, "queue_error", "generation queue poisoned");
                }
            };
            match api_generate_finish_sentence(
                &engine,
                &audio_wav,
                continuation,
                &api_options,
                api_job_progress(app, &job_id, "finish", &label),
            ) {
                Ok(audio) => {
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "finish",
                        "complete",
                        &label,
                        "complete",
                        Some(1),
                        Some(1),
                        "Complete",
                    );
                    match encode_api_audio_response(&audio, format) {
                        Ok((content_type, bytes)) => (
                            200,
                            content_type,
                            bytes,
                            "continuation generated".to_string(),
                        ),
                        Err(e) => json_error(500, "audio_encode_failed", e),
                    }
                }
                Err(e) => {
                    let message = map_engine_err(&e);
                    emit_studio_job(
                        app,
                        &job_id,
                        "api",
                        "finish",
                        if message.to_lowercase().contains("cancel") {
                            "cancelled"
                        } else {
                            "failed"
                        },
                        &label,
                        "failed",
                        None,
                        None,
                        &message,
                    );
                    json_error(500, "generation_failed", message)
                }
            }
        }
        ("POST", "/v1/higgs/multi-speaker") => json_error(
            501,
            "multi_speaker_pending",
            "multi-speaker API route is planned after the speaker library",
        ),
        ("POST", "/v1/higgs/transcribe-reference") => json_error(
            501,
            "transcribe_api_pending",
            "transcription API route is planned after API file upload handling",
        ),
        _ => json_error(404, "not_found", "route not found"),
    }
}

fn serve_api_connection(
    mut stream: TcpStream,
    app: AppHandle,
    engine_state: Arc<Mutex<Option<Arc<Engine>>>>,
    generation_queue: Arc<Mutex<()>>,
    api_key: String,
    speakers: Arc<Mutex<Vec<ApiSpeakerPersona>>>,
) {
    let start = Instant::now();
    let req = match read_http_request(&mut stream) {
        Ok(req) => req,
        Err(e) => {
            let (status, content_type, body, message) = json_error(400, "bad_request", e);
            write_http_response(&mut stream, status, content_type, &body);
            emit_api_log(
                &app,
                "error",
                "request",
                "",
                "",
                status,
                start.elapsed().as_millis(),
                &message,
            );
            return;
        }
    };
    if req.route == "/v1/higgs/audio/stream" && req.method != "OPTIONS" {
        let (status, message) = handle_api_stream_request(
            &mut stream,
            &app,
            &req,
            engine_state,
            generation_queue,
            &api_key,
            speakers,
        );
        emit_api_log(
            &app,
            if status >= 500 {
                "error"
            } else if status >= 400 {
                "warn"
            } else {
                "info"
            },
            "job",
            &req.method,
            &req.route,
            status,
            start.elapsed().as_millis(),
            &message,
        );
        return;
    }
    let (status, content_type, body, message) = handle_api_request(
        &app,
        &req,
        engine_state,
        generation_queue,
        &api_key,
        speakers,
    );
    write_http_response(&mut stream, status, content_type, &body);
    emit_api_log(
        &app,
        if status >= 500 {
            "error"
        } else if status >= 400 {
            "warn"
        } else {
            "info"
        },
        if req.route.contains("/higgs/") || req.route.contains("/audio/speech") {
            "job"
        } else {
            "request"
        },
        &req.method,
        &req.route,
        status,
        start.elapsed().as_millis(),
        &message,
    );
}

#[tauri::command]
fn api_server_start(
    app: AppHandle,
    state: State<'_, AppState>,
    config: ApiServerConfig,
) -> Result<serde_json::Value, String> {
    if config.api_key.trim().len() < 8 {
        return Err("API key must be at least 8 characters".into());
    }
    let host = if config.host.trim().is_empty() {
        "127.0.0.1".to_string()
    } else {
        config.host.trim().to_string()
    };
    let listener = TcpListener::bind((host.as_str(), config.port)).map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let local_addr = listener.local_addr().map_err(|e| e.to_string())?;
    let port = local_addr.port();

    let mut guard = state.api_server.lock().unwrap();
    if guard
        .as_ref()
        .map(|server| !server.stop.load(Ordering::SeqCst))
        .unwrap_or(false)
    {
        return Err("API server is already running".into());
    }
    {
        let mut speaker_guard = state.api_speakers.lock().unwrap();
        *speaker_guard = config.speakers.clone().unwrap_or_default();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = stop.clone();
    let thread_app = app.clone();
    let thread_engine = state.engine.clone();
    let thread_queue = state.generation_queue.clone();
    let api_key = config.api_key.clone();
    let speakers = state.api_speakers.clone();
    std::thread::spawn(move || {
        emit_api_log(
            &thread_app,
            "info",
            "server",
            "START",
            "/api",
            200,
            0,
            "API server started",
        );
        while !thread_stop.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let app = thread_app.clone();
                    let engine = thread_engine.clone();
                    let queue = thread_queue.clone();
                    let key = api_key.clone();
                    let speakers = speakers.clone();
                    std::thread::spawn(move || {
                        serve_api_connection(stream, app, engine, queue, key, speakers)
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    emit_api_log(
                        &thread_app,
                        "error",
                        "server",
                        "ACCEPT",
                        "/api",
                        500,
                        0,
                        &e.to_string(),
                    );
                    break;
                }
            }
        }
        emit_api_log(
            &thread_app,
            "info",
            "server",
            "STOP",
            "/api",
            200,
            0,
            "API server stopped",
        );
    });

    *guard = Some(ApiServerHandle {
        stop,
        host: host.clone(),
        port,
        api_key: config.api_key,
        started_at: Instant::now(),
    });

    Ok(serde_json::json!({
        "running": true,
        "host": host,
        "port": port,
        "baseUrl": format!("http://{}:{}/v1", local_addr.ip(), port),
    }))
}

#[tauri::command]
fn api_update_speakers(
    state: State<'_, AppState>,
    speakers: Vec<ApiSpeakerPersona>,
) -> Result<serde_json::Value, String> {
    let count = speakers.len();
    {
        let mut guard = state.api_speakers.lock().unwrap();
        *guard = speakers;
    }
    let running = state
        .api_server
        .lock()
        .unwrap()
        .as_ref()
        .map(|server| !server.stop.load(Ordering::SeqCst))
        .unwrap_or(false);
    Ok(serde_json::json!({
        "running": running,
        "speakerCount": count,
    }))
}

#[tauri::command]
fn api_server_stop(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let mut guard = state.api_server.lock().unwrap();
    if let Some(server) = guard.take() {
        server.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect((server.host.as_str(), server.port));
    }
    Ok(serde_json::json!({ "running": false }))
}

#[tauri::command]
fn api_server_status(state: State<'_, AppState>) -> serde_json::Value {
    let guard = state.api_server.lock().unwrap();
    if let Some(server) = guard.as_ref() {
        let running = !server.stop.load(Ordering::SeqCst);
        serde_json::json!({
            "running": running,
            "host": server.host,
            "port": server.port,
            "baseUrl": format!("http://{}:{}/v1", server.host, server.port),
            "apiKeySet": !server.api_key.is_empty(),
            "uptimeSeconds": server.started_at.elapsed().as_secs(),
        })
    } else {
        serde_json::json!({ "running": false })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// File helpers
// ═══════════════════════════════════════════════════════════════════════════

fn safe_file_part(value: &str) -> String {
    let safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim_matches('_');
    if safe.is_empty() {
        "speaker".into()
    } else {
        safe.chars().take(80).collect()
    }
}

fn speaker_store_root(_app: &AppHandle) -> Result<PathBuf, String> {
    // Portable: keep speaker personas next to the executable instead of %APPDATA%.
    let root = app_root_dir().join("speakers");
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    Ok(root)
}

fn speaker_dir(app: &AppHandle, id: &str, name: &str) -> Result<PathBuf, String> {
    let dir =
        speaker_store_root(app)?.join(format!("{}_{}", safe_file_part(name), safe_file_part(id)));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn speaker_cache_file(app: &AppHandle, id: &str, name: &str) -> Result<PathBuf, String> {
    let dir = speaker_dir(app, id, name)?.join("cache");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("speaker.hspkcache"))
}

// ───────────────────────────────────────────────────────────────────────────
// Standard voice pack (Nerual Dreming) — reference-audio + transcript pairs
// downloaded from Hugging Face and imported into the Speaker Gallery.
// ───────────────────────────────────────────────────────────────────────────

const VOICEPACK_URL: &str =
    "https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip";

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct VoicePackVoice {
    name: String,
    audio_path: String,
    text: String,
}

fn collect_voicepack_voices(root: &Path) -> Vec<VoicePackVoice> {
    let mut voices = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .unwrap_or_default();
            if !matches!(ext.as_str(), "wav" | "mp3" | "flac") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("voice")
                .to_string();
            // Matching transcript sidecar: "<stem>.lab" or "<stem>.txt".
            let mut text = String::new();
            for sidecar in ["lab", "txt"] {
                let t_path = path.with_extension(sidecar);
                if let Ok(content) = std::fs::read_to_string(&t_path) {
                    text = content.trim().to_string();
                    break;
                }
            }
            voices.push(VoicePackVoice {
                name,
                audio_path: path.to_string_lossy().into_owned(),
                text,
            });
        }
    }
    voices.sort_by(|a, b| a.name.cmp(&b.name));
    voices
}

/// Download (once) and extract the standard voice pack, returning the list of
/// voices so the frontend can turn each into a Speaker Gallery persona.
#[tauri::command]
fn download_voicepack(force: Option<bool>) -> Result<Vec<VoicePackVoice>, String> {
    use std::io::Write as _;
    let force = force.unwrap_or(false);
    let root = app_root_dir();
    let voicepack_dir = root.join("voicepack");

    if !force {
        let existing = collect_voicepack_voices(&voicepack_dir);
        if !existing.is_empty() {
            return Ok(existing);
        }
    }

    let downloads = root.join("downloads");
    std::fs::create_dir_all(&downloads).map_err(|e| e.to_string())?;
    let zip_path = downloads.join("voice-pack.zip");

    if force || !zip_path.exists() {
        let response = ureq::get(VOICEPACK_URL)
            .call()
            .map_err(|e| format!("voice-pack download failed: {e}"))?;
        let mut reader = response.into_body().into_reader();
        let file = std::fs::File::create(&zip_path).map_err(|e| e.to_string())?;
        let mut writer = std::io::BufWriter::new(file);
        std::io::copy(&mut reader, &mut writer).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
    }

    std::fs::create_dir_all(&voicepack_dir).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(&zip_path).map_err(|e| e.to_string())?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("voice-pack is not a valid zip: {e}"))?;
    archive
        .extract(&voicepack_dir)
        .map_err(|e| format!("voice-pack extract failed: {e}"))?;

    let voices = collect_voicepack_voices(&voicepack_dir);
    if voices.is_empty() {
        return Err("voice-pack contained no audio files".into());
    }
    Ok(voices)
}

fn add_zip_file<W: Write + Seek>(
    zip: &mut zip::ZipWriter<W>,
    source: &Path,
    archive_name: &str,
) -> Result<(), String> {
    if !source.exists() || !source.is_file() {
        return Ok(());
    }
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file(archive_name.replace('\\', "/"), options)
        .map_err(|e| e.to_string())?;
    let mut file = std::fs::File::open(source).map_err(|e| e.to_string())?;
    std::io::copy(&mut file, zip).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn speaker_cache_path(
    app: AppHandle,
    speaker_id: String,
    speaker_name: String,
) -> Result<serde_json::Value, String> {
    let path = speaker_cache_file(&app, &speaker_id, &speaker_name)?;
    Ok(serde_json::json!({
        "path": path.to_string_lossy(),
        "exists": path.exists(),
    }))
}

#[tauri::command]
fn save_binary_file(path: String, base64_data: String) -> Result<(), String> {
    let bytes = base64_decode(&base64_data).map_err(|e| e.to_string())?;
    std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn save_audio_file(path: String, base64_wav: String) -> Result<(), String> {
    save_binary_file(path, base64_wav)
}

#[tauri::command]
fn save_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn store_speaker_asset(
    app: AppHandle,
    speaker_id: String,
    speaker_name: String,
    source_path: String,
    asset_kind: String,
) -> Result<serde_json::Value, String> {
    let src = PathBuf::from(&source_path);
    if !src.exists() {
        return Err(format!("source file not found: {}", source_path));
    }
    let ext = src
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("dat");
    let stem = if asset_kind == "image" {
        "display"
    } else {
        "reference"
    };
    let filename = format!("{}.{}", stem, ext);
    let dir = speaker_dir(&app, &speaker_id, &speaker_name)?;
    let dest = dir.join(&filename);
    if let (Ok(src_canon), Ok(dest_canon)) = (src.canonicalize(), dest.canonicalize()) {
        if src_canon == dest_canon {
            return Ok(serde_json::json!({
                "path": dest.to_string_lossy(),
                "fileName": filename,
            }));
        }
    }
    std::fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "path": dest.to_string_lossy(),
        "fileName": filename,
    }))
}

#[tauri::command]
fn write_speaker_metadata(
    app: AppHandle,
    mut speaker: SpeakerArchivePersona,
) -> Result<serde_json::Value, String> {
    let dir = speaker_dir(&app, &speaker.id, &speaker.name)?;
    let cache_path = speaker_cache_file(&app, &speaker.id, &speaker.name)?;
    if speaker.cache_path.trim().is_empty() {
        speaker.cache_path = cache_path.to_string_lossy().into_owned();
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&speaker).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(dir.join("transcript.txt"), &speaker.ref_text).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("notes.txt"), &speaker.notes).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "dir": dir.to_string_lossy(),
        "cachePath": cache_path.to_string_lossy(),
        "cacheExists": cache_path.exists(),
    }))
}

#[tauri::command]
fn export_speaker_zip(
    app: AppHandle,
    path: String,
    speakers: Vec<SpeakerArchivePersona>,
) -> Result<(), String> {
    let file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let manifest = serde_json::json!({
        "app": "Higgs Audio v3 Studio",
        "format": "speaker-gallery-v1",
        "speakers": speakers,
    });
    zip.start_file("manifest.json", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    )
    .map_err(|e| e.to_string())?;

    let speakers: Vec<SpeakerArchivePersona> = serde_json::from_value(
        manifest
            .get("speakers")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    )
    .map_err(|e| e.to_string())?;

    for speaker in speakers {
        let dir = format!(
            "speakers/{}_{}",
            safe_file_part(&speaker.name),
            safe_file_part(&speaker.id)
        );
        zip.start_file(format!("{}/transcript.txt", dir), options)
            .map_err(|e| e.to_string())?;
        zip.write_all(speaker.ref_text.as_bytes())
            .map_err(|e| e.to_string())?;
        zip.start_file(format!("{}/notes.txt", dir), options)
            .map_err(|e| e.to_string())?;
        zip.write_all(speaker.notes.as_bytes())
            .map_err(|e| e.to_string())?;
        if !speaker.ref_path.trim().is_empty() {
            let src = PathBuf::from(&speaker.ref_path);
            let name = src
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("reference_audio");
            add_zip_file(&mut zip, &src, &format!("{}/{}", dir, name))?;
        }
        if !speaker.photo_path.trim().is_empty() {
            let src = PathBuf::from(&speaker.photo_path);
            let name = src
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("display_image");
            add_zip_file(&mut zip, &src, &format!("{}/{}", dir, name))?;
        }
        let cache_path = if !speaker.cache_path.trim().is_empty() {
            PathBuf::from(&speaker.cache_path)
        } else {
            speaker_cache_file(&app, &speaker.id, &speaker.name)?
        };
        add_zip_file(&mut zip, &cache_path, &format!("{}/cache/speaker.hspkcache", dir))?;
    }
    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn import_speaker_zip(app: AppHandle, path: String) -> Result<serde_json::Value, String> {
    let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut manifest_text = String::new();
    archive
        .by_name("manifest.json")
        .map_err(|e| e.to_string())?
        .read_to_string(&mut manifest_text)
        .map_err(|e| e.to_string())?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).map_err(|e| e.to_string())?;
    let mut speakers: Vec<SpeakerArchivePersona> = serde_json::from_value(
        manifest
            .get("speakers")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    )
    .map_err(|e| e.to_string())?;

    for speaker in speakers.iter_mut() {
        let dir_name = format!(
            "{}_{}",
            safe_file_part(&speaker.name),
            safe_file_part(&speaker.id)
        );
        let archive_dir = format!("speakers/{}", dir_name);
        let local_dir = speaker_dir(&app, &speaker.id, &speaker.name)?;
        speaker.ref_path.clear();
        speaker.photo_path.clear();
        speaker.cache_path.clear();

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
            let name = file.name().replace('\\', "/");
            if !name.starts_with(&format!("{}/", archive_dir)) || name.ends_with('/') {
                continue;
            }
            let rel = name
                .strip_prefix(&format!("{}/", archive_dir))
                .unwrap_or("")
                .trim_start_matches('/');
            let rel_path = Path::new(rel);
            if rel.is_empty()
                || rel_path.components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::Prefix(_)
                    )
                })
            {
                continue;
            }
            let leaf = Path::new(&name)
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("asset")
                .to_string();
            if leaf == "transcript.txt" || leaf == "notes.txt" {
                continue;
            }
            let out = local_dir.join(rel_path);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut outfile = std::fs::File::create(&out).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
            let leaf_lc = leaf.to_lowercase();
            if rel.starts_with("cache/") {
                if leaf_lc == "speaker.hspkcache" {
                    speaker.cache_path = out.to_string_lossy().into_owned();
                }
            } else if leaf_lc.starts_with("reference") || speaker.ref_path.is_empty() {
                speaker.ref_path = out.to_string_lossy().into_owned();
                speaker.ref_name = leaf.clone();
            } else if leaf_lc.starts_with("display") || speaker.photo_path.is_empty() {
                speaker.photo_path = out.to_string_lossy().into_owned();
            }
        }
        if speaker.cache_path.trim().is_empty() {
            speaker.cache_path = speaker_cache_file(&app, &speaker.id, &speaker.name)?
                .to_string_lossy()
                .into_owned();
        }
    }

    Ok(serde_json::json!({ "speakers": speakers }))
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn audio_waveform(audio_path: String, points: Option<usize>) -> Result<serde_json::Value, String> {
    let peaks =
        audio::waveform_peaks(&audio_path, points.unwrap_or(1200)).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "peaks": peaks }))
}

#[tauri::command]
async fn prepare_reference_upload(
    audio_path: String,
    max_seconds: Option<f64>,
) -> Result<serde_json::Value, String> {
    let max_seconds = max_seconds
        .filter(|seconds| *seconds > 0.0)
        .unwrap_or(REFERENCE_MAX_SECONDS);
    let prepared = tauri::async_runtime::spawn_blocking(move || {
        audio::prepare_reference_wav(&audio_path, false, 0.95, Some(max_seconds))
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    let file_name = PathBuf::from(&prepared.path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("reference.wav")
        .to_string();
    Ok(serde_json::json!({
        "path": prepared.path,
        "fileName": file_name,
        "durationSeconds": prepared.duration_seconds,
        "cropped": prepared.cropped,
    }))
}

#[tauri::command]
fn read_audio_as_wav(
    audio_path: String,
    target_sample_rate: Option<i32>,
) -> Result<serde_json::Value, String> {
    let (wav, sample_rate, channels, sample_count) =
        audio::decode_to_pcm16_wav(&audio_path, target_sample_rate).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": sample_count,
        "wavBase64": base64_encode(&wav),
    }))
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("Only http(s) URLs are allowed".into());
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", &url])
            .creation_flags(0x08000000)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Hardware snapshot — NVML + sysinfo (no shell-out, no sidecar)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HardwareSnapshot {
    gpu_name: String,
    total_vram: u64,
    used_vram: u64,
    free_vram: u64,
    gpu_utilization: f64,
    temperature: f64,
    power_draw: f64,
    power_limit: f64,
    process_ram: u64,
    total_ram: u64,
    used_ram: u64,
    message: String,
}

impl Default for HardwareSnapshot {
    fn default() -> Self {
        Self {
            gpu_name: String::new(),
            total_vram: 0,
            used_vram: 0,
            free_vram: 0,
            gpu_utilization: 0.0,
            temperature: 0.0,
            power_draw: 0.0,
            power_limit: 0.0,
            process_ram: 0,
            total_ram: 0,
            used_ram: 0,
            message: String::new(),
        }
    }
}

#[tauri::command]
fn hardware_snapshot(state: State<'_, AppState>) -> HardwareSnapshot {
    let mut snap = HardwareSnapshot::default();

    // --- GPU via NVML (lazy init on first poll) ---
    let mut nvml_guard = state.nvml.lock().unwrap();
    if nvml_guard.is_none() {
        match Nvml::init() {
            Ok(nvml) => {
                *nvml_guard = Some(nvml);
            }
            Err(e) => {
                snap.message = format!("NVML init failed: {e}");
            }
        }
    }
    if let Some(ref nvml) = *nvml_guard {
        match nvml.device_by_index(0) {
            Ok(device) => {
                snap.gpu_name = device.name().unwrap_or_else(|_| "NVIDIA GPU".into());

                if let Ok(mem) = device.memory_info() {
                    snap.total_vram = mem.total;
                    snap.used_vram = mem.used;
                    snap.free_vram = mem.free;
                }

                if let Ok(util) = device.utilization_rates() {
                    snap.gpu_utilization = util.gpu as f64;
                }

                if let Ok(temp) =
                    device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                {
                    snap.temperature = temp as f64;
                }

                if let Ok(power_mw) = device.power_usage() {
                    snap.power_draw = power_mw as f64 / 1000.0;
                }

                if let Ok(limit_mw) = device.power_management_limit() {
                    snap.power_limit = limit_mw as f64 / 1000.0;
                }
            }
            Err(e) => {
                snap.message = format!("NVML device error: {e}");
            }
        }
    } else {
        snap.message = "NVML unavailable — no NVIDIA GPU detected".into();
    }
    drop(nvml_guard);

    // --- System RAM + process RAM via sysinfo ---
    let mut sys_guard = state.sys.lock().unwrap();
    sys_guard.refresh_memory();
    snap.total_ram = sys_guard.total_memory();
    snap.used_ram = sys_guard.used_memory();

    let pid = Pid::from(std::process::id() as usize);
    if let Some(proc_info) = sys_guard.process(pid) {
        snap.process_ram = proc_info.memory();
    }

    snap
}

// ═══════════════════════════════════════════════════════════════════════════
// base64 helpers
// ═══════════════════════════════════════════════════════════════════════════

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(input: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        lookup[c as usize] = i as u8;
    }
    let input_bytes = input.as_bytes();
    let mut out = Vec::with_capacity(input_bytes.len() * 3 / 4);
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;
    for &byte in input_bytes {
        if byte == b'=' {
            break;
        }
        let val = lookup[byte as usize];
        if val == 255 {
            continue;
        }
        buffer = (buffer << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8 & 0xFF);
        }
    }
    Ok(out)
}

#[tauri::command]
fn set_minimize_to_tray(enabled: bool, state: State<'_, AppState>) -> Result<(), String> {
    state.minimize_to_tray.store(enabled, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn quit_app(app: AppHandle) -> Result<(), String> {
    app.exit(0);
    Ok(())
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn cancel_active_generation(app: &AppHandle) {
    let engine = app
        .state::<AppState>()
        .engine
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().cloned());
    if let Some(engine) = engine {
        engine.cancel();
        let _ = app.emit(
            "generation-progress",
            serde_json::json!({
                "current": 0,
                "total": 1,
                "phase": "cancelling",
            }),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Point `ort` (ONNX Runtime, used by Parakeet ASR via `load-dynamic`) at the
/// `onnxruntime.dll` that ships next to the exe. Portable-first: no reliance on
/// a system-wide install or PATH. If the user hasn't overridden `ORT_DYLIB_PATH`
/// and a bundled DLL exists, use it; otherwise leave ort's default search.
fn configure_onnx_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let root = app_root_dir();
    let candidates = [
        root.join("onnxruntime.dll"),
        root.join("resources").join("onnxruntime.dll"),
    ];
    if let Some(dll) = candidates.into_iter().find(|p| p.exists()) {
        std::env::set_var("ORT_DYLIB_PATH", dll);
    }
}

pub fn run() {
    configure_onnx_dylib();

    // Portable: keep WebView2 state (localStorage settings) inside the app folder.
    // WebView2 honours this env var when the host does not set an explicit folder.
    if std::env::var_os("WEBVIEW2_USER_DATA_FOLDER").is_none() {
        std::env::set_var(
            "WEBVIEW2_USER_DATA_FOLDER",
            app_root_dir().join("webview-data"),
        );
    }

    let engine_dir = default_engine_download_dir();

    tauri::Builder::default()
        .setup(|app| {
            configure_higgs_asset_env(app);
            let open_item = MenuItem::with_id(app, "open", "Open", true, None::<&str>)?;
            let cancel_item = MenuItem::with_id(
                app,
                "cancel_generation",
                "Cancel Generation",
                true,
                None::<&str>,
            )?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let tray_menu = Menu::with_items(app, &[&open_item, &cancel_item, &quit_item])?;
            let mut tray_builder = TrayIconBuilder::new()
                .tooltip("Higgs Ultimate")
                .menu(&tray_menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "open" => show_main_window(app),
                    "cancel_generation" => cancel_active_generation(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                tray_builder = tray_builder.icon(icon.clone());
            }
            tray_builder.build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<AppState>();
                if window.label() == "main" && state.minimize_to_tray.load(Ordering::SeqCst) {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .manage(AppState {
            engine: Arc::new(Mutex::new(None)),
            engine_dir: Mutex::new(engine_dir),
            nvml: Mutex::new(None),
            sys: Mutex::new(System::new()),
            download_control: Arc::new(download::DownloadControl::default()),
            api_server: Mutex::new(None),
            api_speakers: Arc::new(Mutex::new(Vec::new())),
            generation_queue: Arc::new(Mutex::new(())),
            minimize_to_tray: Arc::new(AtomicBool::new(false)),
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            engine_status,
            engine_version,
            bundled_engine_path,
            diagnose_engine_load,
            download_engine_dll,
            load_engine,
            unload_engine,
            load_model,
            unload_model,
            cancel_generation,
            generate_tts,
            generate_voice_clone,
            generate_finish_sentence,
            transcribe_audio,
            diarize_transcribe,
            build_speaker_voice,
            sortformer_status,
            download_sortformer,
            unload_asr,
            normalize_wav,
            recorder::start_recording,
            recorder::stop_recording,
            recorder::list_input_devices,
            list_models,
            download_model,
            download_control,
            api_server_start,
            api_update_speakers,
            api_server_stop,
            api_server_status,
            save_binary_file,
            save_audio_file,
            save_text_file,
            read_text_file,
            store_speaker_asset,
            download_voicepack,
            speaker_cache_path,
            write_speaker_metadata,
            export_speaker_zip,
            import_speaker_zip,
            prepare_reference_upload,
            audio_waveform,
            read_audio_as_wav,
            open_external_url,
            hardware_snapshot,
            set_minimize_to_tray,
            quit_app,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Higgs Audio v3 Studio");
}
