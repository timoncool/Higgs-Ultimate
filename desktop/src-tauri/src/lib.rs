mod audio;
mod download;
mod engine;

use engine::{Engine, EngineError, GenerateRequest, LoadModelRequest, ProgressCallback};
use nvml_wrapper::Nvml;
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sysinfo::{Pid, System};
use tauri::{App, AppHandle, Emitter, Manager, State};

/// The engine is stored as Arc<Engine> inside the mutex. When a generate call
/// needs to run, it clones the Arc out of the lock and releases the mutex —
/// the engine stays alive for the duration of the call even if the UI triggers
/// an unload concurrently.
struct AppState {
    engine: Mutex<Option<Arc<Engine>>>,
    engine_dir: Mutex<PathBuf>,
    nvml: Mutex<Option<Nvml>>,
    sys: Mutex<System>,
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

fn engine_filename() -> &'static str {
    if cfg!(windows) {
        "audiocpp_engine.dll"
    } else {
        "libaudiocpp_engine.so"
    }
}

fn default_engine_download_dir() -> PathBuf {
    if cfg!(windows) {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local_app_data)
                .join("Higgs Audio v3 Studio")
                .join("engine");
        }
    }

    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".higgs-audio-v3-studio")
        .join("engine")
}

fn user_home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn user_models_root() -> PathBuf {
    user_home_dir().join("audiocpp").join("models")
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
        user_home_dir().join("audiocpp").join(path)
    } else {
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
    serde_json::json!({
        "engineLoaded": engine_loaded,
        "modelLoaded": model_loaded,
        "generating": generating,
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

#[tauri::command]
async fn download_engine_dll(
    app: AppHandle,
    state: State<'_, AppState>,
    url: String,
) -> Result<serde_json::Value, String> {
    let dest_dir = state.engine_dir();
    let filename = engine_filename();
    let result = tauri::async_runtime::spawn_blocking(move || {
        download::download_file(&url, &dest_dir, Some(filename), &app)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "path": result.path, "size": result.size }))
}

#[tauri::command]
async fn load_engine(
    app: AppHandle,
    state: State<'_, AppState>,
    library_path: Option<String>,
) -> Result<serde_json::Value, String> {
    let lib_path = if let Some(p) = library_path {
        PathBuf::from(p)
    } else {
        if let Some(dll) = find_engine_library(Some(&app), &state) {
            dll
        } else {
            let download_dir = state.engine_dir();
            return Err(format!(
                "Engine library not found. Click Download DLL Engine or copy {} into {}.",
                engine_filename(),
                download_dir.display()
            ));
        }
    };

    #[cfg(target_os = "windows")]
    {
        if let Some(parent) = lib_path.parent() {
            engine::add_dll_directory(parent);
        }
        // Also add CUDA toolkit bin so cudart64/cublas64 DLLs are found
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let cuda_bin = PathBuf::from(&cuda_path).join("bin");
            if cuda_bin.exists() {
                engine::add_dll_directory(&cuda_bin);
            }
        }
    }

    let engine = tauri::async_runtime::spawn_blocking(move || Engine::load(&lib_path))
        .await
        .map_err(|e| format!("task join error: {e}"))?
        .map_err(|e| e.to_string())?;

    let version = engine.version();
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
        serde_json::json!({ "engineLoaded": true, "modelLoaded": false }),
    );
    Ok(serde_json::json!({ "success": true, "version": version }))
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
        serde_json::json!({ "engineLoaded": false, "modelLoaded": false }),
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
    engine.unload_model();
    drop(guard);
    let _ = app.emit(
        "model-status",
        serde_json::json!({ "engineLoaded": true, "modelLoaded": false }),
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
        let _ = app.emit(
            "generation-progress",
            serde_json::json!({
                "current": current, "total": total, "phase": phase,
            }),
        );
    })
}

#[tauri::command]
async fn generate_tts(
    app: AppHandle,
    state: State<'_, AppState>,
    request: GenerateRequest,
) -> Result<serde_json::Value, String> {
    let engine = state.clone_engine()?;
    let options = request.options.unwrap_or(serde_json::json!({}));
    let progress = build_progress(&app);

    let result = tauri::async_runtime::spawn_blocking(move || {
        engine.generate_tts(&request.text, &options, progress)
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
    // Convert non-WAV to temp WAV if needed
    let ref_wav = audio::ensure_wav(&ref_path).map_err(|e| e.to_string())?;
    let options = request.options.unwrap_or(serde_json::json!({}));
    let progress = build_progress(&app);

    let result = tauri::async_runtime::spawn_blocking(move || {
        engine.generate_voice_clone(
            &request.text,
            &ref_wav,
            request.ref_text.as_deref(),
            &options,
            progress,
        )
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
    // Convert non-WAV to temp WAV if needed
    let audio_wav = audio::ensure_wav(&request.audio_path).map_err(|e| e.to_string())?;
    let options = request.options.unwrap_or(serde_json::json!({}));
    let progress = build_progress(&app);

    let result = tauri::async_runtime::spawn_blocking(move || {
        engine.generate_finish_sentence(
            &audio_wav,
            request.continuation_text.as_deref(),
            &options,
            progress,
        )
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
// Transcription (whisper.cpp)
// ═══════════════════════════════════════════════════════════════════════════

#[tauri::command]
async fn transcribe_audio(
    state: State<'_, AppState>,
    audio_path: String,
    whisper_model_path: Option<String>,
    language: Option<String>,
) -> Result<serde_json::Value, String> {
    let engine = state.clone_engine()?;
    let whisper_path = whisper_model_path
        .ok_or_else(|| "No whisper model path set. Select one in Settings.".to_string())?;
    let audio_wav = audio::ensure_wav(&audio_path).map_err(|e| e.to_string())?;
    let lang = language.unwrap_or_else(|| "auto".into());
    let mut out_text = vec![0u8; 65536];

    let (status, out_text) = tauri::async_runtime::spawn_blocking(move || {
        let status = engine.transcribe(&whisper_path, &audio_wav, &lang, &mut out_text);
        (status, out_text)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?;

    let status = status.map_err(|e| e.to_string())?;
    if status != 0 {
        let msg = String::from_utf8_lossy(&out_text)
            .trim_end_matches('\0')
            .to_string();
        return Err(if msg.is_empty() {
            format!("transcribe failed (code {status})")
        } else {
            msg
        });
    }

    let text = String::from_utf8_lossy(&out_text)
        .trim_end_matches('\0')
        .to_string();
    Ok(serde_json::json!({ "text": text }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Model listing / download
// ═══════════════════════════════════════════════════════════════════════════

#[tauri::command]
fn list_models(_state: State<'_, AppState>) -> Vec<download::ModelListing> {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));

    let candidates = [
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
    ];

    let mut seen = HashSet::new();
    let mut listings = Vec::new();
    for root in candidates {
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
async fn download_model(
    app: AppHandle,
    request: download::DownloadRequest,
) -> Result<serde_json::Value, String> {
    let dest_dir = resolve_download_dest_dir(&request.dest_dir);
    let result = tauri::async_runtime::spawn_blocking(move || {
        download::download_file(&request.url, &dest_dir, request.filename.as_deref(), &app)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "path": result.path, "size": result.size }))
}

// ═══════════════════════════════════════════════════════════════════════════
// File helpers
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

pub fn run() {
    let engine_dir = default_engine_download_dir();

    tauri::Builder::default()
        .setup(|app| {
            configure_higgs_asset_env(app);
            Ok(())
        })
        .manage(AppState {
            engine: Mutex::new(None),
            engine_dir: Mutex::new(engine_dir),
            nvml: Mutex::new(None),
            sys: Mutex::new(System::new()),
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            engine_status,
            engine_version,
            bundled_engine_path,
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
            list_models,
            download_model,
            save_binary_file,
            save_audio_file,
            save_text_file,
            read_audio_as_wav,
            open_external_url,
            hardware_snapshot,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Higgs Audio v3 Studio");
}
