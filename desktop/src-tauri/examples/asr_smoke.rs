//! Headless smoke test for Parakeet ASR (parakeet-rs).
//!
//!   cargo run --example asr_smoke -- <model_dir> <audio.wav>
//!
//! Loads the TDT model from <model_dir> and transcribes <audio.wav>, printing
//! the recognized text. Exercises the exact `parakeet::transcribe` path the app
//! uses (warm cache + 16 kHz resample). Requires onnxruntime.dll next to the
//! example exe (or ORT_DYLIB_PATH set).

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("usage: asr_smoke <model_dir> <audio.wav>");
    let wav = args.next().expect("usage: asr_smoke <model_dir> <audio.wav>");

    // Portable-style dylib discovery: onnxruntime.dll next to the exe.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())) {
            let dll = dir.join("onnxruntime.dll");
            if dll.exists() {
                std::env::set_var("ORT_DYLIB_PATH", dll);
            }
        }
    }

    let start = std::time::Instant::now();
    match higgs_audio_studio_lib::parakeet::transcribe(&PathBuf::from(&model_dir), &wav) {
        Ok(text) => {
            let secs = start.elapsed().as_secs_f32();
            println!("--- transcript ({secs:.2}s) ---");
            println!("{text}");
        }
        Err(e) => {
            eprintln!("transcription failed: {e}");
            std::process::exit(1);
        }
    }
}
