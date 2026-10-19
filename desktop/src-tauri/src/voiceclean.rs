// Voice cleaning: vocal separation of a reference before cloning (Mel-Band Roformer).
//
// A reference with music or background under the voice makes the clone pick that up too; separating the vocals first
// gives the model the voice alone. The engine is BSRoformer.cpp (chenmozhijin, C++/ggml, CUDA), run as a sidecar CLI:
// `<model.gguf> <in.wav> <out.wav>` writes the vocal stem. The engine sits in resources/bsroformer beside the
// executable and the model in models/bsroformer; both are downloaded on demand (engine ~165 MB, model ~250 MB).
//
// Model: Mel-Band Roformer vocals voc_fv6, Q8_0 (the GGUF repository's recommendation: about FP32 quality at a
// quarter of the size).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::storage;

pub const ENGINE_CLI_FILE: &str = "bs_roformer-cli.exe";

pub const MODEL_FILE: &str = "voc_fv6-Q8_0.gguf";

/// Windows CUDA 13 build of the engine: a flat archive with the CLI and four ggml DLLs.
pub const ENGINE_ZIP_URL: &str =
    "https://github.com/chenmozhijin/BSRoformer.cpp/releases/download/v0.1.0/BSRoformer-windows-cuda-13.1.0.zip";

pub const MODEL_URL: &str = "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/df802a6773d25ba6ef785ff619daa3e510503168/GaboxR67/MelBandRoformers/melbandroformers/vocals/voc_fv6-Q8_0.gguf";

pub fn engine_dir() -> PathBuf {
    storage::engine_dir()
        .parent()
        .map(|resources| resources.join("bsroformer"))
        .unwrap_or_else(|| PathBuf::from("resources").join("bsroformer"))
}

pub fn engine_cli_path() -> Option<PathBuf> {
    let path = engine_dir().join(ENGINE_CLI_FILE);
    path.exists().then_some(path)
}

pub fn model_dir() -> PathBuf {
    storage::models_root().join("bsroformer")
}

pub fn model_path() -> PathBuf {
    model_dir().join(MODEL_FILE)
}

pub fn is_installed() -> bool {
    engine_cli_path().is_some() && model_path().exists()
}

/// Separates the vocals of a WAV and returns the path of the stem (stereo 44.1 kHz; the reference preparation mixes
/// it to mono). The caller removes the stem.
pub fn clean_voice(in_wav: &str) -> Result<String, String> {
    let cli = engine_cli_path().ok_or_else(|| "The voice-cleaning engine is not installed".to_string())?;
    let model = model_path();
    if !model.exists() {
        return Err(format!("Voice-cleaning model not found: {}", model.display()));
    }

    // The engine takes 44.1 kHz input.
    let (wav_bytes, sample_rate, _, _) = crate::audio::decode_to_pcm16_wav(in_wav, Some(44_100))
        .map_err(|error| format!("Could not decode the reference for cleaning: {error}"))?;
    if sample_rate != 44_100 {
        return Err(format!("Expected 44100 Hz after resampling, got {sample_rate}"));
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let temp_dir = storage::temp_dir();
    let stem_in = temp_dir.join(format!("higgs_clean_in_{stamp}.wav"));
    let stem_out = temp_dir.join(format!("higgs_clean_out_{stamp}.wav"));
    std::fs::write(&stem_in, &wav_bytes).map_err(|error| format!("Could not write the cleaning input: {error}"))?;

    let ran = run_cli(&cli, &model, &stem_in, &stem_out);
    let _ = std::fs::remove_file(&stem_in);
    ran?;

    if !stem_out.exists() {
        return Err("The voice-cleaning engine wrote no output".to_string());
    }
    Ok(stem_out.to_string_lossy().into_owned())
}

fn run_cli(cli: &Path, model: &Path, input: &Path, output: &Path) -> Result<(), String> {
    let mut command = Command::new(cli);
    command.arg(model).arg(input).arg(output);
    // the ggml DLLs sit beside the CLI
    if let Some(dir) = cli.parent() {
        command.current_dir(dir);
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let result = command
        .output()
        .map_err(|error| format!("Could not start the voice-cleaning engine: {error}"))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        let mut tail: Vec<&str> = stderr.lines().chain(stdout.lines()).rev().take(6).collect();
        tail.reverse();
        return Err(format!("The voice-cleaning engine failed ({}): {}", result.status, tail.join(" | ")));
    }
    Ok(())
}
