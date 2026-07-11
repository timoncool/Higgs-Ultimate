//! Headless-тест очистки голоса (вокал-сепарация Mel-Band Roformer / BSRoformer.cpp).
//!
//! Берёт речевой wav, домешивает синтетический фон (тон 440 Гц + белый шум +
//! музыкоподобный аккорд с тремоло), гоняет через `voiceclean::clean_voice` и
//! сверяет: RMS фона в паузах речи упал, речь сохранилась. Печатает тайминг.
//! GUI не запускается. Движок и модель должны лежать штатно (resources/bsroformer,
//! models/bsroformer) или рядом с exe примера.
//!
//! Usage:
//!   voiceclean_test <speech.(wav|mp3|…)> <out_dir>

use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

use higgs_audio_studio_lib::{audio, voiceclean};

const SR: i32 = 44_100;

fn decode_44k_mono(path: &str) -> Vec<f32> {
    let (wav, sr, _ch, _n) = audio::decode_to_pcm16_wav(path, Some(SR)).expect("decode");
    assert_eq!(sr, SR);
    let mut rdr = hound::WavReader::new(std::io::Cursor::new(wav)).expect("wav reader");
    rdr.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn rms(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let speech_path = args.next().expect("speech path");
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&out_dir).ok();

    println!("=== decode speech to 44.1k mono ===");
    let speech = decode_44k_mono(&speech_path);
    let n = speech.len();
    println!(
        "speech = {:.2}s ({})",
        n as f32 / SR as f32,
        Path::new(&speech_path).file_name().unwrap().to_string_lossy()
    );

    // Синтетический фон: 440 Гц тон + белый шум + аккорд 220/277/330 с тремоло 3 Гц.
    let mut rng: u32 = 0x1234_5678;
    let mut white = || {
        // xorshift → [-1,1)
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut bg = vec![0.0f32; n];
    for (i, b) in bg.iter_mut().enumerate() {
        let t = i as f32 / SR as f32;
        let tone = 0.10 * (2.0 * PI * 440.0 * t).sin();
        let noise = 0.05 * white();
        let trem = 0.5 * (1.0 + (2.0 * PI * 3.0 * t).sin());
        let chord = 0.09 * trem
            * ((2.0 * PI * 220.0 * t).sin()
                + (2.0 * PI * 277.0 * t).sin()
                + (2.0 * PI * 330.0 * t).sin())
            / 3.0;
        *b = tone + noise + chord;
    }

    let mut mix = vec![0.0f32; n];
    for i in 0..n {
        mix[i] = (0.8 * speech[i] + bg[i]).clamp(-1.0, 1.0);
    }
    let mix_path = out_dir.join("mixed.wav");
    std::fs::write(&mix_path, audio::encode_pcm16_wav(&mix, SR, 1)).expect("write mix");
    println!(
        "mixed = {} (speech RMS {:.5}, bg RMS {:.5})",
        mix_path.display(),
        rms(&speech),
        rms(&bg)
    );

    if !voiceclean::is_installed() {
        eprintln!(
            "\n[SKIP] движок/модель очистки не установлены (engine_cli={:?}, model={})",
            voiceclean::engine_cli_path(),
            voiceclean::model_path().display()
        );
        std::process::exit(2);
    }

    println!("\n=== clean_voice (Mel-Band Roformer, sidecar) ===");
    let t = Instant::now();
    let cleaned_path = voiceclean::clean_voice(&mix_path.to_string_lossy()).expect("clean_voice");
    let elapsed = t.elapsed().as_secs_f32();
    println!(
        "clean took {:.3}s for {:.2}s clip  →  {:.1}x realtime",
        elapsed,
        n as f32 / SR as f32,
        (n as f32 / SR as f32) / elapsed
    );

    let cleaned = decode_44k_mono(&cleaned_path);
    let out_copy = out_dir.join("cleaned.wav");
    std::fs::write(&out_copy, audio::encode_pcm16_wav(&cleaned, SR, 1)).expect("write cleaned");
    let m = n.min(cleaned.len());

    // Маска пауз речи: сглаженная огибающая чистой речи ниже порога.
    let win = (0.02 * SR as f32) as usize;
    let mut env = vec![0.0f32; m];
    let mut acc = 0.0f32;
    for i in 0..m {
        acc += speech[i].abs();
        if i >= win {
            acc -= speech[i - win].abs();
        }
        env[i] = acc / win as f32;
    }
    let thr = 0.02f32;
    let pause: Vec<usize> = (0..m).filter(|&i| env[i] < thr).collect();
    let active: Vec<usize> = (0..m).filter(|&i| env[i] >= thr).collect();

    let pick = |idx: &[usize], src: &[f32]| idx.iter().map(|&i| src[i]).collect::<Vec<f32>>();
    let mix_pause = rms(&pick(&pause, &mix));
    let clean_pause = rms(&pick(&pause, &cleaned));
    let reduction = mix_pause / clean_pause.max(1e-9);

    println!("\n=== separation quality ===");
    println!("pause fraction: {:.1}%", 100.0 * pause.len() as f32 / m as f32);
    println!("  mixed  pauses RMS: {:.5}", mix_pause);
    println!("  clean  pauses RMS: {:.5}", clean_pause);
    println!(
        "  background reduction: {:.1}x  ({:.1} dB)",
        reduction,
        20.0 * reduction.log10()
    );
    println!("  speech-ref active RMS: {:.5}", rms(&pick(&active, &speech)));
    println!("  cleaned  active RMS:   {:.5}", rms(&pick(&active, &cleaned)));

    // Корреляция очищенного с чистой речью в активных участках (речь сохранилась?).
    let a = pick(&active, &cleaned);
    let b = pick(&active, &speech);
    let ma = a.iter().sum::<f32>() / a.len() as f32;
    let mb = b.iter().sum::<f32>() / b.len() as f32;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..a.len() {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    let corr = cov / (va.sqrt() * vb.sqrt()).max(1e-9);
    println!("  corr(cleaned, speech_ref) in active: {:.3}", corr);

    println!("\nartifacts: {}  +  {}", mix_path.display(), out_copy.display());
    println!("DONE");
}
