// ─── Системные библиотеки: диагностика и автозакачка ────────────────────────
//
// Приложение само выясняет, каких внешних зависимостей не хватает, и предлагает
// скачать их одной кнопкой. Цель — свести ручную установку к единственному шагу
// (драйвер NVIDIA), убрав из README требование ставить CUDA Toolkit и VC++ Redist.
//
// Три класса зависимостей движка (audiocpp_engine.dll / ggml-cuda.dll):
//   • CUDA 13 runtime — cudart64_13 / cublas64_13 / cublasLt64_13. Редистрибутивные
//     DLL берём из официальных PyPI-wheel'ов NVIDIA (wheel = zip; DLL лежат под
//     nvidia/cu13/bin/x86_64/*.dll). Редистрибуция cudart/cublas в составе
//     приложения разрешена NVIDIA CUDA Toolkit EULA (Attachment A).
//   • VC++ runtime — MSVCP140 / VCRUNTIME140 / VCRUNTIME140_1 / VCOMP140.
//     Портативные DLL лежат в engines/-папке HF-репозитория Higgs-Audio-v3-Studio.
//   • Драйвер NVIDIA — nvcuda.dll/nvml.dll в System32. DLL-кой не ставится: это
//     часть драйвера. Детектим через NVML, а «скачивание» = открыть сайт NVIDIA.
//
// Скачанные DLL кладём в resources/engine/ рядом с audiocpp_engine.dll — ту же
// папку загрузчик уже добавляет в путь поиска DLL (windows_dependency_search_dirs).

use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::AppHandle;

use crate::download::{self, DownloadControl};

// ── Имена DLL по классам ────────────────────────────────────────────────────

/// CUDA 13 runtime DLL, которых требует движок сверх драйвера.
pub const CUDA_DLLS: &[&str] = &["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"];

/// VC++ runtime DLL (2015–2022 x64), которых требуют движок и bsroformer.
pub const VCRUNTIME_DLLS: &[&str] = &[
    "MSVCP140.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
    "VCOMP140.DLL",
];

// ── Источники закачки ───────────────────────────────────────────────────────

/// PyPI-wheel CUDA runtime (cudart64_13.dll). Версия 13.3.29 совпадает с ABI
/// локально проверенного CUDA Toolkit 13.3; DLL внутри байт-в-байт идентична.
pub const CUDART_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/d2/27/b53a5e0397842a5c11f0e1a39d4e5b2f22638a4126e83b3c4e196f62c969/nvidia_cuda_runtime-13.3.29-py3-none-win_amd64.whl";

/// PyPI-wheel cuBLAS (cublas64_13.dll + cublasLt64_13.dll). Версия 13.6.0.2 — DLL
/// байт-в-байт совпадают с cuBLAS из локального CUDA 13.3 (проверено размерами).
/// ~394 МБ — основной вес закачки CUDA (cublasLt в распаковке ~442 МБ).
pub const CUBLAS_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/08/8f/890a96ea1ff615100296977cce23296052dcb8c114d4e451201ec39df9bf/nvidia_cublas-13.6.0.2-py3-none-win_amd64.whl";

/// База HF-репозитория для VC++ DLL (каждая — отдельный файл в engines/).
pub const VCRUNTIME_HF_BASE: &str =
    "https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/engines";

/// Страница драйверов NVIDIA (для кнопки «Открыть сайт» — DLL-кой не ставится).
/// Дублируется в config.ts как NVIDIA_DRIVER_URL; здесь — источник истины бэка.
#[allow(dead_code)]
pub const NVIDIA_DRIVER_URL: &str = "https://www.nvidia.com/Download/index.aspx";

/// Приблизительные размеры для UI (МБ), чтобы показать «Скачать ~XXX МБ» до старта.
pub const CUDA_DOWNLOAD_MB: u64 = 397; // cudart 2.6 + cublas 394
pub const VCRUNTIME_DOWNLOAD_MB: u64 = 1; // 4 DLL, ~1 МБ суммарно

// ── Класс зависимости ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepKind {
    Cuda,
    Vcruntime,
}

impl DepKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cuda" => Some(DepKind::Cuda),
            "vcruntime" | "vc" | "vcredist" => Some(DepKind::Vcruntime),
            _ => None,
        }
    }

    fn dlls(&self) -> &'static [&'static str] {
        match self {
            DepKind::Cuda => CUDA_DLLS,
            DepKind::Vcruntime => VCRUNTIME_DLLS,
        }
    }
}

// ── Результат диагностики ───────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DriverStatus {
    pub ok: bool,
    pub version: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DllGroupStatus {
    pub ok: bool,
    /// DLL, которых не хватает (по именам). Пусто, если всё на месте.
    pub missing: Vec<String>,
    /// Найденные DLL с путём (для диагностики/копипаста).
    pub found: Vec<FoundDll>,
    /// Оценка размера закачки недостающего, МБ (для кнопки).
    pub download_mb: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FoundDll {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EnvCheck {
    pub driver: DriverStatus,
    pub cuda: DllGroupStatus,
    pub vcruntime: DllGroupStatus,
}

// ── Поиск DLL ───────────────────────────────────────────────────────────────

/// Директории, в которых ищем системные DLL: сперва папка движка (куда кладём
/// скачанное), затем рядом с exe, дерево разработки и весь PATH. Регистр имени
/// DLL в файловой системе Windows не важен, но сравниваем без учёта регистра.
pub fn search_dirs(engine_dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let push = |d: PathBuf, dirs: &mut Vec<PathBuf>| {
        if !d.as_os_str().is_empty() && !dirs.iter().any(|e| e == &d) {
            dirs.push(d);
        }
    };

    push(engine_dir.to_path_buf(), &mut dirs);

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push(parent.to_path_buf(), &mut dirs);
            push(parent.join("resources").join("engine"), &mut dirs);
        }
    }
    push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("engine"),
        &mut dirs,
    );

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            push(dir, &mut dirs);
        }
    }

    dirs
}

/// Найти конкретную DLL по имени (без учёта регистра) в списке директорий.
pub fn find_dll(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    let target = name.to_ascii_lowercase();
    for dir in dirs {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        // Регистронезависимый обход (на случай VCOMP140.DLL vs vcomp140.dll и т.п.).
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    == target
                {
                    let p = entry.path();
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

/// Собрать статус группы DLL (какие есть, каких нет).
fn check_group(dirs: &[PathBuf], names: &[&str], download_mb: u64) -> DllGroupStatus {
    let mut missing = Vec::new();
    let mut found = Vec::new();
    for &name in names {
        match find_dll(dirs, name) {
            Some(path) => found.push(FoundDll {
                name: name.to_string(),
                path: path.to_string_lossy().into_owned(),
            }),
            None => missing.push(name.to_string()),
        }
    }
    DllGroupStatus {
        ok: missing.is_empty(),
        download_mb: if missing.is_empty() { 0 } else { download_mb },
        missing,
        found,
    }
}

// ── Детект драйвера NVIDIA ───────────────────────────────────────────────────

/// Версия драйвера NVIDIA через NVML (nvml.dll на Windows / libnvidia-ml на
/// Linux — часть драйвера; загрузка = попытка `LoadLibrary`). Возвращает None,
/// если NVML не грузится (драйвер не установлен) или в системе нет GPU NVIDIA.
/// Не требует загруженного движка. NVML прикрывает LoadLibrary внутри, отдельная
/// проба nvml.dll не нужна.
pub fn detect_driver() -> DriverStatus {
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => DriverStatus {
            ok: true,
            version: nvml.sys_driver_version().ok(),
        },
        Err(_) => DriverStatus {
            ok: false,
            version: None,
        },
    }
}

// ── Публичная диагностика ────────────────────────────────────────────────────

/// Полная проверка окружения: драйвер + CUDA runtime + VC++ runtime.
pub fn env_check(engine_dir: &Path) -> EnvCheck {
    let dirs = search_dirs(engine_dir);
    EnvCheck {
        driver: detect_driver(),
        cuda: check_group(&dirs, CUDA_DLLS, CUDA_DOWNLOAD_MB),
        vcruntime: check_group(&dirs, VCRUNTIME_DLLS, VCRUNTIME_DOWNLOAD_MB),
    }
}

// ── Извлечение DLL из wheel/zip ──────────────────────────────────────────────

/// Из скачанного wheel (zip) достать все *.dll и разложить их плоско в dest_dir,
/// беря только имя файла. Возвращает список записанных имён.
pub fn extract_dlls_from_wheel(wheel_path: &Path, dest_dir: &Path) -> Result<Vec<String>, String> {
    let file =
        std::fs::File::open(wheel_path).map_err(|e| format!("открыть wheel {}: {e}", wheel_path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("wheel не является zip: {e}"))?;
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("создать {}: {e}", dest_dir.display()))?;

    let mut written = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("чтение записи wheel: {e}"))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let leaf = name.rsplit('/').next().unwrap_or(&name).to_string();
        if !leaf.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        let out = dest_dir.join(&leaf);
        let tmp = out.with_extension("dll.part");
        {
            let mut f =
                std::fs::File::create(&tmp).map_err(|e| format!("создать {}: {e}", tmp.display()))?;
            std::io::copy(&mut entry, &mut f).map_err(|e| format!("распаковка {leaf}: {e}"))?;
        }
        std::fs::rename(&tmp, &out).map_err(|e| format!("финализация {leaf}: {e}"))?;
        written.push(leaf);
    }
    Ok(written)
}

// ── Закачка недостающих зависимостей ─────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadEnvResult {
    pub kind: String,
    /// Имена DLL, которые оказались на месте после закачки.
    pub installed: Vec<String>,
    /// Имена DLL, которых всё ещё не хватает (закачка не помогла).
    pub missing: Vec<String>,
    pub ok: bool,
}

/// Скачать недостающие DLL класса `kind` в `engine_dir` (та же прогресс-машина
/// download-progress, что у моделей). CUDA — из PyPI-wheel'ов с распаковкой,
/// VC++ — прямыми файлами из HF. Идемпотентно: уже имеющиеся DLL не трогаем.
///
/// `wheel_tmp_dir` — куда складывать временные wheel (обычно системный temp).
pub fn download_env_deps(
    kind: DepKind,
    engine_dir: &Path,
    wheel_tmp_dir: &Path,
    app: &AppHandle,
    control: std::sync::Arc<DownloadControl>,
) -> Result<DownloadEnvResult, String> {
    std::fs::create_dir_all(engine_dir).map_err(|e| format!("создать {}: {e}", engine_dir.display()))?;

    // Проверяем, что реально отсутствует, чтобы не качать 394 МБ ради уже
    // имеющегося файла.
    let dirs = search_dirs(engine_dir);
    let group = check_group(&dirs, kind.dlls(), 0);
    if group.ok {
        return Ok(DownloadEnvResult {
            kind: kind_str(kind).to_string(),
            installed: kind.dlls().iter().map(|s| s.to_string()).collect(),
            missing: Vec::new(),
            ok: true,
        });
    }
    let missing_lower: Vec<String> = group.missing.iter().map(|s| s.to_ascii_lowercase()).collect();

    match kind {
        DepKind::Cuda => {
            // cudart-wheel закрывает cudart64_13; cublas-wheel — cublas64_13 +
            // cublasLt64_13. Качаем только те wheel, чьи DLL отсутствуют.
            let need_cudart = missing_lower.iter().any(|m| m == "cudart64_13.dll");
            let need_cublas = missing_lower
                .iter()
                .any(|m| m == "cublas64_13.dll" || m == "cublaslt64_13.dll");

            if need_cudart {
                download_wheel_extract(CUDART_WHEEL_URL, "cuda-runtime.whl", engine_dir, wheel_tmp_dir, app, &control)?;
            }
            if need_cublas {
                download_wheel_extract(CUBLAS_WHEEL_URL, "cublas.whl", engine_dir, wheel_tmp_dir, app, &control)?;
            }
        }
        DepKind::Vcruntime => {
            // Каждая VC++ DLL — отдельный файл в HF engines/. Качаем только те,
            // которых не хватает, прямо в engine_dir.
            for &dll in VCRUNTIME_DLLS {
                if !missing_lower.contains(&dll.to_ascii_lowercase()) {
                    continue;
                }
                let url = format!("{VCRUNTIME_HF_BASE}/{dll}");
                download::download_file(&url, engine_dir, Some(dll), app, control.clone())
                    .map_err(|e| format!("скачать {dll}: {e}"))?;
            }
        }
    }

    // Повторная проверка после закачки.
    let dirs = search_dirs(engine_dir);
    let after = check_group(&dirs, kind.dlls(), 0);
    Ok(DownloadEnvResult {
        kind: kind_str(kind).to_string(),
        installed: after.found.iter().map(|f| f.name.clone()).collect(),
        missing: after.missing.clone(),
        ok: after.ok,
    })
}

fn kind_str(kind: DepKind) -> &'static str {
    match kind {
        DepKind::Cuda => "cuda",
        DepKind::Vcruntime => "vcruntime",
    }
}

/// Скачать wheel во временную папку той же прогресс-машиной, распаковать DLL в
/// engine_dir, удалить временный wheel.
fn download_wheel_extract(
    url: &str,
    tmp_name: &str,
    engine_dir: &Path,
    wheel_tmp_dir: &Path,
    app: &AppHandle,
    control: &std::sync::Arc<DownloadControl>,
) -> Result<(), String> {
    let res = download::download_file(url, wheel_tmp_dir, Some(tmp_name), app, control.clone())
        .map_err(|e| format!("скачать {tmp_name}: {e}"))?;
    let wheel_path = PathBuf::from(&res.path);
    let extract = extract_dlls_from_wheel(&wheel_path, engine_dir);
    let _ = std::fs::remove_file(&wheel_path);
    extract?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depkind_parse_accepts_known_aliases() {
        assert_eq!(DepKind::parse("cuda"), Some(DepKind::Cuda));
        assert_eq!(DepKind::parse("CUDA"), Some(DepKind::Cuda));
        assert_eq!(DepKind::parse("vcruntime"), Some(DepKind::Vcruntime));
        assert_eq!(DepKind::parse("vc"), Some(DepKind::Vcruntime));
        assert_eq!(DepKind::parse("vcredist"), Some(DepKind::Vcruntime));
        assert_eq!(DepKind::parse("nonsense"), None);
    }

    #[test]
    fn find_dll_is_case_insensitive() {
        let dir = tempdir();
        // Кладём файл в верхнем регистре, ищем в нижнем и наоборот.
        std::fs::write(dir.join("VCOMP140.DLL"), b"x").unwrap();
        let dirs = vec![dir.clone()];
        assert!(find_dll(&dirs, "vcomp140.dll").is_some());
        assert!(find_dll(&dirs, "VCOMP140.DLL").is_some());
        assert!(find_dll(&dirs, "missing.dll").is_none());
        cleanup(&dir);
    }

    #[test]
    fn check_group_reports_missing_and_found() {
        let dir = tempdir();
        std::fs::write(dir.join("cudart64_13.dll"), b"x").unwrap();
        let dirs = vec![dir.clone()];
        let g = check_group(&dirs, CUDA_DLLS, 100);
        assert!(!g.ok);
        assert_eq!(g.found.len(), 1);
        assert_eq!(g.found[0].name, "cudart64_13.dll");
        // cublas64_13 + cublasLt64_13 отсутствуют.
        assert_eq!(g.missing.len(), 2);
        assert!(g.missing.iter().any(|m| m == "cublas64_13.dll"));
        assert!(g.missing.iter().any(|m| m == "cublasLt64_13.dll"));
        assert_eq!(g.download_mb, 100);
        cleanup(&dir);
    }

    #[test]
    fn check_group_ok_when_all_present() {
        let dir = tempdir();
        for &d in VCRUNTIME_DLLS {
            std::fs::write(dir.join(d), b"x").unwrap();
        }
        let dirs = vec![dir.clone()];
        let g = check_group(&dirs, VCRUNTIME_DLLS, 5);
        assert!(g.ok);
        assert!(g.missing.is_empty());
        assert_eq!(g.found.len(), VCRUNTIME_DLLS.len());
        assert_eq!(g.download_mb, 0); // ничего качать не нужно
        cleanup(&dir);
    }

    #[test]
    fn extract_dlls_from_wheel_flattens_and_filters() {
        let dir = tempdir();
        let wheel = dir.join("fake.whl");
        build_zip(
            &wheel,
            &[
                ("nvidia/cu13/bin/x86_64/cudart64_13.dll", b"DLLDATA".as_ref()),
                ("nvidia/cu13/bin/x86_64/notes.txt", b"ignore me".as_ref()),
            ],
        );
        let dest = dir.join("out");
        let written = extract_dlls_from_wheel(&wheel, &dest).unwrap();
        assert_eq!(written, vec!["cudart64_13.dll".to_string()]);
        assert_eq!(
            std::fs::read(dest.join("cudart64_13.dll")).unwrap(),
            b"DLLDATA"
        );
        assert!(!dest.join("notes.txt").exists());
        cleanup(&dir);
    }

    #[test]
    fn env_check_group_shapes_present() {
        // Не зависит от реального железа: проверяем, что структура собирается и
        // группы отражают содержимое пустой engine-папки (всё missing).
        let dir = tempdir();
        // Чтобы PATH не «нашёл» настоящие DLL на этой машине, тест смотрит только
        // на группы через check_group напрямую на пустой директории.
        let dirs = vec![dir.clone()];
        let cuda = check_group(&dirs, CUDA_DLLS, CUDA_DOWNLOAD_MB);
        let vc = check_group(&dirs, VCRUNTIME_DLLS, VCRUNTIME_DOWNLOAD_MB);
        assert_eq!(cuda.missing.len(), CUDA_DLLS.len());
        assert_eq!(vc.missing.len(), VCRUNTIME_DLLS.len());
        assert_eq!(cuda.download_mb, CUDA_DOWNLOAD_MB);
        cleanup(&dir);
    }

    // ── тестовые утилиты ────────────────────────────────────────────────────

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "higgs_envdeps_test_{}_{}",
            std::process::id(),
            unique()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn unique() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static C: AtomicU64 = AtomicU64::new(0);
        C.fetch_add(1, Ordering::Relaxed)
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn build_zip(path: &Path, entries: &[(&str, &[u8])]) {
        use std::io::Write;
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let opts: zip::write::SimpleFileOptions =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, data) in entries {
            zip.start_file(*name, opts).unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap();
    }
}
