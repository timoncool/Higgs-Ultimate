//! Load + first-transcription benchmark for Parakeet TDT int8 at a given ONNX
//! graph optimization level.
//!
//!   HIGGS_ASR_OPT_LEVEL=1 cargo run --example asr_bench -- <model_dir> <audio.wav>
//!   HIGGS_ASR_OPT_LEVEL=3 cargo run --example asr_bench -- <model_dir> <audio.wav>
//!
//! Measures the model load (session creation, where the Level3 optimizer stalls
//! on int8) and the first transcription separately.

use std::path::PathBuf;
use std::time::Instant;

use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("usage: asr_bench <model_dir> <audio.wav>");
    let wav = args.next().expect("usage: asr_bench <model_dir> <audio.wav>");

    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())) {
            let dll = dir.join("onnxruntime.dll");
            if dll.exists() {
                std::env::set_var("ORT_DYLIB_PATH", dll);
            }
        }
    }

    let level = std::env::var("HIGGS_ASR_OPT_LEVEL").unwrap_or_else(|_| "1".into());
    println!("opt_level = {level}");

    // Reuse the app's exec_config() (reads HIGGS_ASR_OPT_LEVEL).
    let cfg = higgs_audio_studio_lib::parakeet::exec_config();

    let t0 = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&PathBuf::from(&model_dir), Some(cfg))
        .expect("load model");
    let load = t0.elapsed().as_secs_f32();
    println!("load = {load:.2}s");

    // Read the wav to 16k mono via hound the same way the app does.
    let mut reader = hound::WavReader::open(&wav).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    assert_eq!(spec.sample_rate, 16_000, "bench expects a 16k wav");

    let t1 = Instant::now();
    let res = model
        .transcribe_samples(samples, 16_000, spec.channels, Some(TimestampMode::Sentences))
        .expect("transcribe");
    let first = t1.elapsed().as_secs_f32();
    println!("first_transcribe = {first:.2}s");
    println!("total = {:.2}s", load + first);
    println!("text = {}", res.text.trim());
}
