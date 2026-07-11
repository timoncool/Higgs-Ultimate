// ─── Очистка голоса: вокал-сепарация референсов (Mel-Band Roformer) ──────────
//
// Перед клонированием референс полезно очистить от фона/музыки — модель клона
// тогда цепляется за чистый голос, а не за подложку. Движок — BSRoformer.cpp
// (нативный C++/ggml, CUDA/CPU, GGUF-кванты) от chenmozhijin, подключён сайдкаром:
// CLI берёт `<model.gguf> <in.wav> <out.wav>` и пишет вокал-стем.
//
// Портируемость как у остального: сайдкар-exe и ggml-DLL лежат рядом с exe в
// resources/bsroformer/, модель — в models/bsroformer/. Обе части качаются лениво
// пресетом при первом включении тумблера (движок ~165 МБ, модель ~250 МБ).
//
// Модель по умолчанию — Mel-Band Roformer vocals `voc_fv6` в кванте Q8_0
// (рекомендация репозитория GGUF: качество ≈ FP32 при 1/4 размера).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{app_root_dir, user_models_root};

/// Сайдкар-CLI движка (Windows). Рядом лежат ggml*.dll — их подхватывает загрузчик
/// из той же папки.
pub const ENGINE_CLI_FILE: &str = "bs_roformer-cli.exe";

/// GGUF-модель вокал-сепарации по умолчанию (Mel-Band Roformer voc_fv6, Q8_0).
pub const MODEL_FILE: &str = "voc_fv6-Q8_0.gguf";

/// Готовый Windows+CUDA билд движка (v0.1.0, CUDA 13.1 — совпадает с локальным
/// тулкитом 13.x; на не-CUDA машине ggml-cuda.dll не загрузится, но целевая
/// платформа — RTX). Архив плоский: exe + 4 ggml-DLL.
pub const ENGINE_ZIP_URL: &str =
    "https://github.com/chenmozhijin/BSRoformer.cpp/releases/download/v0.1.0/BSRoformer-windows-cuda-13.1.0.zip";

/// Прямая ссылка на GGUF voc_fv6-Q8_0 из официального репозитория движка.
pub const MODEL_URL: &str = "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/GaboxR67/MelBandRoformers/melbandroformers/vocals/voc_fv6-Q8_0.gguf?download=true";

/// Папка движка рядом с exe (сайдкар + ggml-DLL).
pub fn engine_dir() -> PathBuf {
    app_root_dir().join("resources").join("bsroformer")
}

/// Кандидаты расположения сайдкар-exe: рядом с exe (портатив/бандл) и в дереве
/// разработки. Первый существующий выигрывает.
fn engine_cli_candidates() -> Vec<PathBuf> {
    let mut dirs = vec![engine_dir()];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // Tauri-бандл кладёт resources рядом с exe.
            dirs.push(parent.join("resources").join("bsroformer"));
        }
    }
    dirs.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("bsroformer"),
    );
    dirs.into_iter().map(|d| d.join(ENGINE_CLI_FILE)).collect()
}

/// Путь к найденному сайдкар-exe (или None, если движок не установлен).
pub fn engine_cli_path() -> Option<PathBuf> {
    engine_cli_candidates().into_iter().find(|p| p.exists())
}

/// Папка модели вокал-сепарации (портатив: models/bsroformer рядом с exe).
pub fn model_dir() -> PathBuf {
    user_models_root().join("bsroformer")
}

/// Кандидаты расположения GGUF-модели: штатная папка рядом с exe и дерево
/// разработки (чтобы headless-примеры из target/debug/examples тоже нашли).
fn model_candidates() -> Vec<PathBuf> {
    vec![
        model_dir().join(MODEL_FILE),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("bsroformer")
            .join(MODEL_FILE),
    ]
}

/// Путь к GGUF-модели: первый существующий кандидат, иначе штатный путь (для UI).
pub fn model_path() -> PathBuf {
    model_candidates()
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| model_dir().join(MODEL_FILE))
}

/// Установлены ли обе части (движок + модель) — для UI.
pub fn is_installed() -> bool {
    engine_cli_path().is_some() && model_path().exists()
}

/// Прогнать WAV через вокал-сепаратор и вернуть путь к очищенному стему.
///
/// `in_wav` — уже декодированный (симфонией) WAV; движку нужен вход 44.1 кГц,
/// поэтому здесь пере-декодируем/ресемплим в mono 44.1k через общий audio-путь.
/// Выход движка (stereo 44.1k) отдаём как есть — downstream (prepare_reference_wav)
/// сам сведёт в mono под клон.
pub fn clean_voice(in_wav: &str) -> Result<String, String> {
    let cli = engine_cli_path()
        .ok_or_else(|| "движок очистки голоса не установлен".to_string())?;
    let model = model_path();
    if !model.exists() {
        return Err(format!(
            "модель очистки голоса не найдена: {}",
            model.display()
        ));
    }

    // Вход движка: mono 44.1 кГц WAV во временной папке.
    let (wav_bytes, sr, _ch, _n) = crate::audio::decode_to_pcm16_wav(in_wav, Some(44_100))
        .map_err(|e| format!("декод входа для очистки: {e}"))?;
    if sr != 44_100 {
        return Err(format!("ожидался ресемпл в 44100 Гц, получено {sr}"));
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir();
    let stem_in = tmp_dir.join(format!("higgs_clean_in_{stamp}.wav"));
    let stem_out = tmp_dir.join(format!("higgs_clean_out_{stamp}.wav"));
    std::fs::write(&stem_in, &wav_bytes).map_err(|e| format!("запись входа очистки: {e}"))?;

    run_cli(&cli, &model, &stem_in, &stem_out)?;

    // Убрать временный вход; выход оставляем — его читает вызывающая сторона.
    let _ = std::fs::remove_file(&stem_in);

    if !stem_out.exists() {
        return Err("движок очистки не создал выходной файл".to_string());
    }
    Ok(stem_out.to_string_lossy().into_owned())
}

/// Запустить сайдкар без окна консоли, дождаться завершения, проверить код.
fn run_cli(cli: &Path, model: &Path, input: &Path, output: &Path) -> Result<(), String> {
    let mut cmd = Command::new(cli);
    cmd.arg(model).arg(input).arg(output);
    // Рабочая директория = папка движка, чтобы ggml-DLL гарантированно нашлись.
    if let Some(dir) = cli.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let output_res = cmd
        .output()
        .map_err(|e| format!("запуск движка очистки: {e}"))?;
    if !output_res.status.success() {
        let stderr = String::from_utf8_lossy(&output_res.stderr);
        let stdout = String::from_utf8_lossy(&output_res.stdout);
        let tail: String = stderr
            .lines()
            .chain(stdout.lines())
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!(
            "движок очистки завершился с ошибкой ({}): {tail}",
            output_res.status
        ));
    }
    Ok(())
}
