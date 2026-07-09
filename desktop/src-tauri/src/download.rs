use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadRequest {
    pub url: String,
    pub dest_dir: String,
    pub filename: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
    pub speed_mbps: f64,
    pub percent: f64,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadResult {
    pub path: String,
    pub size: u64,
}

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("download already running")]
    Busy,
    #[error("download cancelled")]
    Cancelled,
}

#[derive(Debug, Default)]
pub struct DownloadControl {
    active: AtomicBool,
    paused: AtomicBool,
    cancelled: AtomicBool,
}

impl DownloadControl {
    pub fn begin(&self) -> bool {
        if self
            .active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        self.paused.store(false, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
        true
    }

    pub fn finish(&self) {
        self.active.store(false, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        if self.active.load(Ordering::SeqCst) {
            self.paused.store(true, Ordering::SeqCst);
        }
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    pub fn cancel(&self) {
        if self.active.load(Ordering::SeqCst) {
            self.cancelled.store(true, Ordering::SeqCst);
            self.paused.store(false, Ordering::SeqCst);
        }
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

pub fn download_file(
    url: &str,
    dest_dir: &Path,
    filename: Option<&str>,
    app: &AppHandle,
    control: Arc<DownloadControl>,
) -> Result<DownloadResult, DownloadError> {
    if !control.begin() {
        return Err(DownloadError::Busy);
    }
    let result = download_file_inner(url, dest_dir, filename, app, control.clone());
    control.finish();
    result
}

fn emit_download_progress(
    app: &AppHandle,
    downloaded: u64,
    total: u64,
    start_time: std::time::Instant,
    status: &str,
) {
    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let speed_mbps = if elapsed_secs > 0.0 && status == "running" {
        (downloaded as f64 / 1_000_000.0) / elapsed_secs
    } else {
        0.0
    };
    let percent = if total > 0 {
        (downloaded as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    let _ = app.emit(
        "download-progress",
        DownloadProgress {
            downloaded,
            total,
            speed_mbps,
            percent,
            status: status.to_string(),
        },
    );
}

// Parallelize downloads of files ≥ this size across this many connections.
// A single HTTP stream does not saturate a fast link against the HF CDN; N
// concurrent byte-range requests do (aria2-style).
const PARALLEL_MIN_SIZE: u64 = 16 * 1024 * 1024;
const PARALLEL_CONNS: u64 = 8;

#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], off: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_write(buf, off)
}
#[cfg(not(windows))]
fn write_at(f: &File, buf: &[u8], off: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.write_at(buf, off)
}

/// Learn the file size and whether the (possibly redirected) server honours
/// byte ranges, via a 1-byte ranged probe. HF's CDN returns 206 + content-range.
fn probe_size(url: &str) -> (u64, bool) {
    match ureq::get(url).header("Range", "bytes=0-0").call() {
        Ok(resp) => {
            if resp.status().as_u16() == 206 {
                let total = resp
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.rsplit('/').next())
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0);
                (total, total > 0)
            } else {
                let total = resp
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                (total, false)
            }
        }
        Err(_) => (0, false),
    }
}

fn download_range(
    url: &str,
    file: &Arc<File>,
    start: u64,
    end: u64,
    downloaded: &Arc<AtomicU64>,
    control: &DownloadControl,
    abort: &Arc<AtomicBool>,
) -> Result<(), DownloadError> {
    let resp = ureq::get(url)
        .header("Range", &format!("bytes={start}-{end}"))
        .call()
        .map_err(|e| DownloadError::Http(e.to_string()))?;
    let mut reader = resp.into_body().into_reader();
    let mut buf = [0u8; 262144];
    let mut offset = start;
    loop {
        if control.is_cancelled() || abort.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        while control.is_paused() {
            if control.is_cancelled() || abort.load(Ordering::Relaxed) {
                return Err(DownloadError::Cancelled);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        let mut w = 0;
        while w < n {
            let k = write_at(file, &buf[w..n], offset + w as u64)
                .map_err(|e| DownloadError::Io(e.to_string()))?;
            if k == 0 {
                return Err(DownloadError::Io("short write".into()));
            }
            w += k;
        }
        offset += n as u64;
        downloaded.fetch_add(n as u64, Ordering::Relaxed);
    }
    Ok(())
}

fn download_parallel(
    url: &str,
    tmp_path: &Path,
    total: u64,
    app: &AppHandle,
    control: &Arc<DownloadControl>,
) -> Result<u64, DownloadError> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(tmp_path)
        .map_err(|e| DownloadError::Io(e.to_string()))?;
    file.set_len(total)
        .map_err(|e| DownloadError::Io(e.to_string()))?;
    let file = Arc::new(file);

    let conns = PARALLEL_CONNS.min((total / (4 * 1024 * 1024)).max(1)).max(1);
    let chunk = total / conns;
    let downloaded = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let abort = Arc::new(AtomicBool::new(false));
    let error: Arc<std::sync::Mutex<Option<DownloadError>>> =
        Arc::new(std::sync::Mutex::new(None));

    let mut handles = Vec::new();
    for i in 0..conns {
        let start = i * chunk;
        let end = if i == conns - 1 { total - 1 } else { start + chunk - 1 };
        let url = url.to_string();
        let file = file.clone();
        let downloaded = downloaded.clone();
        let finished = finished.clone();
        let abort = abort.clone();
        let error = error.clone();
        let control = control.clone();
        handles.push(std::thread::spawn(move || {
            if let Err(e) = download_range(&url, &file, start, end, &downloaded, &control, &abort) {
                if !matches!(e, DownloadError::Cancelled) {
                    abort.store(true, Ordering::Relaxed); // stop siblings
                    let mut slot = error.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                }
            }
            finished.fetch_add(1, Ordering::SeqCst);
        }));
    }

    let start_time = std::time::Instant::now();
    loop {
        let got = downloaded.load(Ordering::Relaxed);
        if control.is_cancelled() {
            abort.store(true, Ordering::Relaxed);
        }
        let status = if control.is_paused() { "paused" } else { "running" };
        emit_download_progress(app, got, total, start_time, status);
        if finished.load(Ordering::SeqCst) >= conns as usize {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    for h in handles {
        let _ = h.join();
    }

    if let Some(e) = error.lock().unwrap().take() {
        let _ = std::fs::remove_file(tmp_path);
        return Err(e);
    }
    if control.is_cancelled() {
        let _ = std::fs::remove_file(tmp_path);
        emit_download_progress(app, downloaded.load(Ordering::Relaxed), total, start_time, "cancelled");
        return Err(DownloadError::Cancelled);
    }
    Ok(total)
}

fn download_single(
    url: &str,
    tmp_path: &Path,
    dest_path: &Path,
    control: &DownloadControl,
    app: &AppHandle,
) -> Result<DownloadResult, DownloadError> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| DownloadError::Http(e.to_string()))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = response.into_body().into_reader();
    let file = File::create(tmp_path).map_err(|e| DownloadError::Io(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    let start_time = std::time::Instant::now();
    let mut buf = [0u8; 262144];
    let mut local_written: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    loop {
        if control.is_cancelled() {
            drop(writer);
            let _ = std::fs::remove_file(tmp_path);
            emit_download_progress(app, local_written, total, start_time, "cancelled");
            return Err(DownloadError::Cancelled);
        }
        while control.is_paused() {
            if control.is_cancelled() {
                drop(writer);
                let _ = std::fs::remove_file(tmp_path);
                emit_download_progress(app, local_written, total, start_time, "cancelled");
                return Err(DownloadError::Cancelled);
            }
            if last_emit.elapsed().as_millis() > 500 {
                last_emit = std::time::Instant::now();
                emit_download_progress(app, local_written, total, start_time, "paused");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        writer
            .write_all(&buf[..n])
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        local_written += n as u64;
        if last_emit.elapsed().as_millis() > 200 {
            last_emit = std::time::Instant::now();
            emit_download_progress(app, local_written, total, start_time, "running");
        }
    }

    writer.flush().map_err(|e| DownloadError::Io(e.to_string()))?;
    drop(writer);
    std::fs::rename(tmp_path, dest_path).map_err(|e| DownloadError::Io(e.to_string()))?;
    emit_download_progress(app, local_written, total, start_time, "complete");
    Ok(DownloadResult {
        path: dest_path.to_string_lossy().into_owned(),
        size: local_written,
    })
}

fn download_file_inner(
    url: &str,
    dest_dir: &Path,
    filename: Option<&str>,
    app: &AppHandle,
    control: Arc<DownloadControl>,
) -> Result<DownloadResult, DownloadError> {
    if url.is_empty() || !url.starts_with("http") {
        return Err(DownloadError::InvalidUrl("URL must start with http".into()));
    }

    let filename = filename.map(|s| s.to_string()).unwrap_or_else(|| {
        url.rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "model.bin".to_string())
    });

    std::fs::create_dir_all(dest_dir).map_err(|e| DownloadError::Io(e.to_string()))?;
    let dest_path = dest_dir.join(&filename);
    let tmp_path = dest_dir.join(format!("{filename}.tmp"));

    // Fast path: parallel byte-range download for large, range-capable files.
    let (total, ranges_ok) = probe_size(url);
    if ranges_ok && total >= PARALLEL_MIN_SIZE {
        match download_parallel(url, &tmp_path, total, app, &control) {
            Ok(size) => {
                std::fs::rename(&tmp_path, &dest_path)
                    .map_err(|e| DownloadError::Io(e.to_string()))?;
                emit_download_progress(app, size, total, std::time::Instant::now(), "complete");
                return Ok(DownloadResult {
                    path: dest_path.to_string_lossy().into_owned(),
                    size,
                });
            }
            Err(DownloadError::Cancelled) => return Err(DownloadError::Cancelled),
            Err(_) => {
                let _ = std::fs::remove_file(&tmp_path); // fall back to single stream
            }
        }
    }

    download_single(url, &tmp_path, &dest_path, &control, app)
}

pub fn list_model_dirs(models_root: &Path) -> Vec<ModelListing> {
    let mut listings = Vec::new();
    if let Ok(entries) = std::fs::read_dir(models_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let gguf_path = path.join("model.gguf");
            let safetensors_path = path.join("model.safetensors");
            let fallback_gguf = first_file_with_extension(&path, "gguf");
            let fallback_safetensors = first_file_with_extension(&path, "safetensors");
            let weight_path = if gguf_path.exists() {
                Some(gguf_path)
            } else if let Some(found) = fallback_gguf {
                Some(found)
            } else if safetensors_path.exists() {
                Some(safetensors_path)
            } else {
                fallback_safetensors
            };
            let has_gguf = weight_path
                .as_ref()
                .and_then(|p| p.extension())
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            let has_config = path.join("config.json").exists();

            let Some(weight_path) = weight_path else {
                continue;
            };

            let dir_name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();

            let size_bytes = std::fs::metadata(weight_path).map(|m| m.len()).unwrap_or(0);

            listings.push(ModelListing {
                name: dir_name,
                path: path.to_string_lossy().into_owned(),
                format: if has_gguf {
                    "gguf".into()
                } else {
                    "safetensors".into()
                },
                size_bytes,
                has_config,
            });
        }
    }
    listings.sort_by(|a, b| a.name.cmp(&b.name));
    listings
}

fn first_file_with_extension(root: &Path, extension: &str) -> Option<std::path::PathBuf> {
    let mut matches: Vec<_> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelListing {
    pub name: String,
    pub path: String,
    pub format: String,
    pub size_bytes: u64,
    pub has_config: bool,
}
