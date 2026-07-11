//! Headless 2-speaker diarization + transcription test.
//!
//! Склеивает синтетический 2-спикерный wav из ДВУХ РАЗНЫХ голосов войспака (A /
//! пауза / B / пауза / A), прогоняет полный пайплайн вкладки (Sortformer diarize
//! + Parakeet transcribe_turns) и smoke «Сделать голос» (собрать реф-wav+ref_text
//! из результата). GUI не запускается.
//!
//! Usage:
//!   diar_test <tdt_dir> <sortformer.onnx> <voiceA.(wav|mp3)> <voiceB> <out_dir>
//!
//! ORT_DYLIB_PATH подхватывается из onnxruntime.dll рядом с exe примера.

use std::path::{Path, PathBuf};
use std::time::Instant;

use higgs_audio_studio_lib::{audio, parakeet};

const SR: i32 = 16_000;

fn decode_16k_mono(path: &str) -> Vec<f32> {
    // Через тот же symphonia-путь приложения -> PCM16 WAV @16k -> f32.
    let (wav, sr, _ch, _n) = audio::decode_to_pcm16_wav(path, Some(SR)).expect("decode");
    assert_eq!(sr, SR);
    // прочитать PCM16 из байтов
    let mut rdr = hound::WavReader::new(std::io::Cursor::new(wav)).expect("wav reader");
    rdr.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn write_wav(path: &Path, samples: &[f32]) {
    let bytes = audio::encode_pcm16_wav(samples, SR, 1);
    std::fs::write(path, bytes).expect("write wav");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let tdt_dir = args.next().expect("tdt_dir");
    let sortformer = args.next().expect("sortformer.onnx");
    let voice_a = args.next().expect("voiceA");
    let voice_b = args.next().expect("voiceB");
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir).ok();

    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(dir) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        {
            let dll = dir.join("onnxruntime.dll");
            if dll.exists() {
                std::env::set_var("ORT_DYLIB_PATH", dll);
            }
        }
    }

    println!("=== decode voices to 16k mono ===");
    let a = decode_16k_mono(&voice_a);
    let b = decode_16k_mono(&voice_b);
    println!(
        "A = {:.2}s ({}),  B = {:.2}s ({})",
        a.len() as f32 / SR as f32,
        Path::new(&voice_a).file_name().unwrap().to_string_lossy(),
        b.len() as f32 / SR as f32,
        Path::new(&voice_b).file_name().unwrap().to_string_lossy(),
    );

    // Обрезать каждый до ~6с, чтобы клип был компактным, но с запасом реплик.
    let clip = |v: &[f32], secs: f32| v[..v.len().min((secs * SR as f32) as usize)].to_vec();
    let a1 = clip(&a, 6.0);
    let b1 = clip(&b, 6.0);
    let a2 = clip(&a[a.len().min((3.0 * SR as f32) as usize)..], 4.0);

    // Синтетический 2-спикерный: A / пауза 0.7с / B / пауза 0.7с / A.
    let gap = vec![0.0f32; (0.7 * SR as f32) as usize];
    let mut mix: Vec<f32> = Vec::new();
    mix.extend_from_slice(&a1);
    mix.extend_from_slice(&gap);
    mix.extend_from_slice(&b1);
    mix.extend_from_slice(&gap);
    mix.extend_from_slice(&a2);
    let mix_path = out_dir.join("synthetic_2spk.wav");
    write_wav(&mix_path, &mix);
    let mix_str = mix_path.to_string_lossy().into_owned();
    println!(
        "mixed 2-speaker wav = {} ({:.2}s)",
        mix_path.display(),
        mix.len() as f32 / SR as f32
    );

    // 1) Сырая диаризация Sortformer.
    println!("\n=== diarize (Sortformer v2) ===");
    let t = Instant::now();
    let raw = parakeet::diarize(Path::new(&sortformer), &mix_str).expect("diarize");
    println!("diarize took {:.2}s, raw turns = {}", t.elapsed().as_secs_f32(), raw.len());
    let mut spk: Vec<i32> = raw.iter().map(|t| t.speaker).collect();
    spk.sort_unstable();
    spk.dedup();
    println!("raw speakers found = {} {:?}", spk.len(), spk);
    for tn in &raw {
        println!("  spk{} [{:.2}..{:.2}]", tn.speaker, tn.start, tn.end);
    }

    // 2) Полный пайплайн вкладки.
    println!("\n=== transcribe_and_diarize (full pipeline) ===");
    let t = Instant::now();
    let res = parakeet::transcribe_and_diarize(
        Path::new(&tdt_dir),
        Some(Path::new(&sortformer)),
        &mix_str,
    )
    .expect("pipeline");
    println!("pipeline took {:.2}s", t.elapsed().as_secs_f32());
    println!("n_speakers = {}", res.n_speakers);
    for s in &res.segments {
        println!("  SPK{} [{:.2}..{:.2}] {}", s.speaker + 1, s.start, s.end, s.text);
    }

    // 3) Smoke «Сделать голос» для каждого найденного спикера.
    println!("\n=== make-voice smoke ===");
    let mut speakers: Vec<i32> = res.segments.iter().map(|s| s.speaker).collect();
    speakers.sort_unstable();
    speakers.dedup();
    for sp in speakers {
        match parakeet::build_speaker_reference(&mix_str, &res.segments, sp, 30.0, 0.3) {
            Ok((wav, text)) => {
                let p = out_dir.join(format!("voice_spk{}.wav", sp + 1));
                std::fs::write(&p, &wav).expect("write ref wav");
                let dur = {
                    let r = hound::WavReader::new(std::io::Cursor::new(&wav)).unwrap();
                    r.duration() as f32 / SR as f32
                };
                std::fs::write(
                    out_dir.join(format!("voice_spk{}.txt", sp + 1)),
                    text.as_bytes(),
                )
                .ok();
                println!(
                    "  SPK{}: ref wav {} ({:.2}s), ref_text = {:?}",
                    sp + 1,
                    p.display(),
                    dur,
                    text
                );
            }
            Err(e) => println!("  SPK{}: make-voice failed: {}", sp + 1, e),
        }
    }
    println!("\nDONE");
}
