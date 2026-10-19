use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{AppHandle, Manager};

pub const PORTABLE_MARKER: &str = "portable.flag";

#[derive(Clone, Debug)]
struct RuntimePaths {
    portable: bool,
    executable_dir: PathBuf,
    engine_dir: PathBuf,
    models_dir: PathBuf,
    temp_dir: PathBuf,
    webview_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageLayoutInfo {
    portable: bool,
    root: PathBuf,
    engine_dir: PathBuf,
    models_dir: PathBuf,
    speakers_dir: PathBuf,
    temp_dir: PathBuf,
    webview_dir: Option<PathBuf>,
}

pub(crate) fn executable_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn user_home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn installed_engine_dir() -> PathBuf {
    if cfg!(windows) {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local_app_data)
                .join("Higgs Audio v3 Studio")
                .join("engine");
        }
    }

    user_home_dir()
        .join(".higgs-audio-v3-studio")
        .join("engine")
}

fn detect_paths() -> RuntimePaths {
    let executable_dir = executable_dir();
    let portable = std::env::var_os("HIGGS_STUDIO_PORTABLE")
        .map(|value| value != "0")
        .unwrap_or(false)
        || executable_dir.join(PORTABLE_MARKER).is_file();

    if portable {
        let data_dir = executable_dir.join("data");
        RuntimePaths {
            portable: true,
            engine_dir: executable_dir.join("resources").join("engine"),
            models_dir: executable_dir.join("models"),
            temp_dir: data_dir.join("temp"),
            webview_dir: Some(data_dir.join("webview")),
            executable_dir,
        }
    } else {
        RuntimePaths {
            portable: false,
            executable_dir,
            engine_dir: installed_engine_dir(),
            models_dir: user_home_dir().join("audiocpp").join("models"),
            temp_dir: std::env::temp_dir(),
            webview_dir: None,
        }
    }
}

fn paths() -> &'static RuntimePaths {
    static PATHS: OnceLock<RuntimePaths> = OnceLock::new();
    PATHS.get_or_init(detect_paths)
}

pub fn initialize() -> Result<(), String> {
    let paths = paths();
    if !paths.portable {
        return Ok(());
    }
    move_legacy_portable_dirs(&paths.executable_dir)?;

    for dir in [
        &paths.engine_dir,
        &paths.models_dir,
        &paths.temp_dir,
        paths
            .webview_dir
            .as_ref()
            .expect("portable webview directory"),
    ] {
        std::fs::create_dir_all(dir).map_err(|error| {
            format!(
                "Could not create portable folder {}: {error}",
                dir.display()
            )
        })?;
    }
    Ok(())
}

/// Higgs Ultimate's portable builds up to 0.4 kept the speakers and the WebView state beside the executable: they move
/// to the `data` folder once, so an update keeps the voices and the settings.
fn move_legacy_portable_dirs(executable_dir: &Path) -> Result<(), String> {
    let data_dir = executable_dir.join("data");
    for (old, new) in [("speakers", data_dir.join("speakers")), ("webview-data", data_dir.join("webview"))] {
        let old = executable_dir.join(old);
        if old.is_dir() && !new.exists() {
            std::fs::create_dir_all(&data_dir).map_err(|error| format!("Could not create {}: {error}", data_dir.display()))?;
            std::fs::rename(&old, &new).map_err(|error| format!("Could not move {} to {}: {error}", old.display(), new.display()))?;
        }
    }
    Ok(())
}

pub fn is_portable() -> bool {
    paths().portable
}

pub fn engine_dir() -> PathBuf {
    paths().engine_dir.clone()
}

pub fn models_root() -> PathBuf {
    paths().models_dir.clone()
}

pub fn temp_dir() -> PathBuf {
    paths().temp_dir.clone()
}

pub fn webview_dir() -> Option<PathBuf> {
    paths().webview_dir.clone()
}

pub fn resolve_download_dest_dir(dest_dir: &str) -> PathBuf {
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
        if paths().portable {
            paths().executable_dir.join(path)
        } else {
            user_home_dir().join("audiocpp").join(path)
        }
    } else {
        paths().models_dir.join(path)
    }
}

pub fn speaker_store_root(app: &AppHandle) -> Result<PathBuf, String> {
    let root = if paths().portable {
        paths().executable_dir.join("data").join("speakers")
    } else {
        app.path()
            .app_data_dir()
            .map_err(|error| error.to_string())?
            .join("speakers")
    };
    std::fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    Ok(root)
}

pub fn layout_info(app: &AppHandle) -> Result<StorageLayoutInfo, String> {
    let speakers_dir = speaker_store_root(app)?;
    let root = if paths().portable {
        paths().executable_dir.clone()
    } else {
        app.path()
            .app_data_dir()
            .map_err(|error| error.to_string())?
    };
    Ok(StorageLayoutInfo {
        portable: paths().portable,
        root,
        engine_dir: paths().engine_dir.clone(),
        models_dir: paths().models_dir.clone(),
        speakers_dir,
        temp_dir: paths().temp_dir.clone(),
        webview_dir: paths().webview_dir.clone(),
    })
}
