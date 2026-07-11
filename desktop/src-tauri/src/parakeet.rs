//! Parakeet ASR via the `parakeet-rs` crate (ONNX Runtime, CPU).
//!
//! Replaces the earlier parakeet.cpp FFI (`resources/parakeet/parakeet.dll`).
//! ONNX Runtime is NOT static-linked: `ort` is built with `load-dynamic`, so at
//! runtime it dlopens `onnxruntime.dll`. That DLL is fetched once by
//! `ensure_onnx_runtime` (lib.rs) and placed next to the exe, portable-style —
//! same idea as the audiocpp engine DLLs. `ORT_DYLIB_PATH` points ort at it.
//! Besides that DLL, the only thing shipped is the model folder (encoder/decoder
//! ONNX + vocab).
//!
//! The model is loaded ONCE into a warm session (keyed by the model directory)
//! and reused across calls, so only the first transcription pays the load cost.
//! The session is guarded by a Mutex, so transcriptions are serialized (the
//! `ParakeetTDT` handle is `&mut self` per call — never concurrent).

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

/// ONNX execution config for the ASR/diarization sessions. Lowers the graph
/// optimization level to Level1: on the int8 quant the default Level3 optimizer
/// hangs for minutes when creating the CPU session (it spins on
/// DynamicQuantizeLinear/MatMulInteger). Overridable via `HIGGS_ASR_OPT_LEVEL`
/// (0=Disable, 1=Level1, 2=Level2, 3=Level3) for benchmarking.
pub fn exec_config() -> ExecutionConfig {
    use ort::session::builder::GraphOptimizationLevel;
    let level = std::env::var("HIGGS_ASR_OPT_LEVEL")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(1);
    ExecutionConfig::new().with_custom_configure(move |b| {
        let lvl = match level {
            0 => GraphOptimizationLevel::Disable,
            2 => GraphOptimizationLevel::Level2,
            3 => GraphOptimizationLevel::Level3,
            _ => GraphOptimizationLevel::Level1,
        };
        Ok(b.with_optimization_level(lvl)?)
    })
}

/// A warm TDT model, kept alive and reused across transcriptions. `dir` is the
/// model folder it was loaded from — a different folder forces a reload.
struct Session {
    dir: String,
    model: ParakeetTDT,
}

fn cell() -> &'static Mutex<Option<Session>> {
    static S: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// Load a 16 kHz mono `Vec<f32>` from a WAV, resampling if needed. The
/// parakeet-rs feature extractor requires 16 kHz input (it downmixes channels
/// itself but does not resample), so we normalize the rate here.
fn load_wav_16k_mono(wav_path: &str) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(wav_path).map_err(|e| format!("open wav {wav_path}: {e}"))?;
    let spec = reader.spec();

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("read wav samples: {e}"))?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("read wav samples: {e}"))?
        }
    };

    let channels = spec.channels.max(1) as usize;
    let mono: Vec<f32> = if channels > 1 {
        interleaved
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        interleaved
    };

    if spec.sample_rate == 16_000 {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, spec.sample_rate, 16_000))
    }
}

/// Linear resampling to a target rate. Adequate for ASR feature extraction; the
/// mel front-end is tolerant and this avoids pulling in a heavy DSP dependency.
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input[idx.min(input.len() - 1)];
        let b = input[(idx + 1).min(input.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Transcribe `wav_path` with the multilingual Parakeet TDT model located in
/// `model_dir`. The model auto-detects language (no RU/EN toggle needed).
pub fn transcribe(model_dir: &Path, wav_path: &str) -> Result<String, String> {
    let dir = model_dir.to_string_lossy().into_owned();

    let audio = load_wav_16k_mono(wav_path)?;

    let mut guard = cell()
        .lock()
        .map_err(|_| "parakeet state poisoned".to_string())?;

    // Load once; reload only if a different model directory is requested.
    let need_reload = guard.as_ref().map(|s| s.dir != dir).unwrap_or(true);
    if need_reload {
        let model = ParakeetTDT::from_pretrained(model_dir, Some(exec_config()))
            .map_err(|e| format!("parakeet: failed to load model in {dir}: {e}"))?;
        *guard = Some(Session {
            dir: dir.clone(),
            model,
        });
    }
    let session = guard.as_mut().expect("session present after load");

    let result = session
        .model
        .transcribe_samples(audio, 16_000, 1, Some(TimestampMode::Sentences))
        .map_err(|e| format!("parakeet transcribe failed: {e}"))?;

    Ok(result.text.trim().to_string())
}

/// Drop the warm model, freeing its RAM (reloaded lazily on next use).
pub fn unload() {
    if let Ok(mut g) = cell().lock() {
        *g = None;
    }
}
