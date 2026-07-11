//! Parakeet ASR + Sortformer diarization via the `parakeet-rs` crate (ONNX
//! Runtime, CPU). Replaces the earlier parakeet.cpp FFI.
//!
//! ONNX Runtime is NOT static-linked: `ort` is built with `load-dynamic`, so at
//! runtime it dlopens `onnxruntime.dll`. That DLL is fetched once by
//! `ensure_onnx_runtime` (lib.rs) and placed next to the exe, portable-style —
//! same idea as the audiocpp engine DLLs. `ORT_DYLIB_PATH` points ort at it.
//! Besides that DLL, the only things shipped are the model folders (TDT encoder/
//! decoder ONNX + vocab, and the Sortformer diarization ONNX).
//!
//! The TDT model is loaded ONCE into a warm session (keyed by the model
//! directory) and reused across calls, so only the first transcription pays the
//! load cost. The session is guarded by a Mutex, so transcriptions are serialized
//! (the `ParakeetTDT` handle is `&mut self` per call — never concurrent).
//!
//! Diarization (Sortformer v2) is a separate model, loaded per call — it is used
//! far less often than transcription, so a warm cache is not worth the RAM.
//!
//! Word segmentation (`segment_words`), `diarize`, `turns`, `transcribe` and
//! `transcribe_turns` are ports of the dub-studio dub-asr crate (itself a port of
//! dubengine/asr.py + diarize.py): pause > 0.6 s, sentence end .!?…, max 8.0 s.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

/// Target sample rate — parakeet-rs requires exactly 16 kHz mono.
pub const TARGET_SR: u32 = 16_000;

/// ONNX execution config for the ASR/diarization sessions. Defaults the graph
/// optimization level to Level1 as a conservative guard: on some CPU/toolset
/// combos the default Level3 optimizer stalls for minutes when creating an int8
/// session (it spins on DynamicQuantizeLinear/MatMulInteger). On this build
/// (onnxruntime 1.24.2, ort rc.12) Level3 does NOT stall and is ~0.2 s faster to
/// load (~1.38 s vs ~1.60 s), with identical transcription output — so Level1 is
/// a small, safe cost, not a win. Overridable via `HIGGS_ASR_OPT_LEVEL`
/// (0=Disable, 1=Level1, 2=Level2, 3=Level3) for those who want the extra speed.
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

// ─── типы результата ─────────────────────────────────────────────────────────

/// Слово со временем (секунды).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// Сегмент реплики: [start,end] + текст. Привязан к спикеру (0 если один голос).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub speaker: i32,
}

/// Одна реплика диаризации: [start,end] в секундах, speaker — контиг. id 0..k-1.
#[derive(Debug, Clone)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

/// Результат всего пайплайна вкладки: сегменты + число спикеров.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptResult {
    pub segments: Vec<SpeakerSegment>,
    pub n_speakers: usize,
}

fn ends_sentence(word: &str) -> bool {
    matches!(
        word.chars().last(),
        Some('.') | Some('!') | Some('?') | Some('…')
    )
}

/// Разбить поток слов на сегменты: разрыв на паузе > max_gap, конце предложения
/// (.!?…) или превышении max_dur. Порт _segment из dubengine/asr.py.
pub fn segment_words(words: &[Word], max_gap: f64, max_dur: f64) -> Vec<(f64, f64, String)> {
    let mut segs: Vec<Vec<Word>> = Vec::new();
    let mut cur: Vec<Word> = Vec::new();

    for w in words {
        if let (Some(last), Some(first)) = (cur.last(), cur.first()) {
            let gap = w.start - last.end;
            let dur = last.end - first.start;
            if gap > max_gap || dur > max_dur {
                segs.push(std::mem::take(&mut cur));
            }
        }
        cur.push(w.clone());
        if ends_sentence(&w.word) {
            segs.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        segs.push(cur);
    }

    segs.into_iter()
        .filter(|ws| !ws.is_empty())
        .map(|ws| {
            let start = ws.first().unwrap().start;
            let end = ws.last().unwrap().end;
            let text = ws
                .iter()
                .map(|x| x.word.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            (start, end, text)
        })
        .collect()
}

// ─── тёплая TDT-модель ───────────────────────────────────────────────────────

struct Session {
    dir: String,
    model: ParakeetTDT,
}

fn cell() -> &'static Mutex<Option<Session>> {
    static S: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// Прогнать тёплую модель на семплах и получить словные таймстемпы.
fn transcribe_words_warm(model: &mut ParakeetTDT, audio: &[f32]) -> Result<Vec<Word>, String> {
    let res = model
        .transcribe_samples(audio.to_vec(), TARGET_SR, 1, Some(TimestampMode::Words))
        .map_err(|e| format!("parakeet transcribe failed: {e}"))?;
    let audio_end = audio.len() as f64 / TARGET_SR as f64;
    Ok(res
        .tokens
        .into_iter()
        .filter(|t| !t.text.trim().is_empty())
        .map(|t| {
            let start = t.start as f64;
            let mut end = (t.end as f64).max(start);
            if end <= start {
                end = audio_end.max(start);
            }
            Word {
                word: t.text.trim().to_string(),
                start,
                end,
            }
        })
        .collect())
}

/// Загрузить (или переиспользовать) тёплую TDT-модель и выполнить `f`.
fn with_warm_model<T>(
    model_dir: &Path,
    f: impl FnOnce(&mut ParakeetTDT) -> Result<T, String>,
) -> Result<T, String> {
    let dir = model_dir.to_string_lossy().into_owned();
    let mut guard = cell()
        .lock()
        .map_err(|_| "parakeet state poisoned".to_string())?;
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
    f(&mut session.model)
}

// ─── публичный ASR API ───────────────────────────────────────────────────────

/// Простая транскрипция всего клипа (для существующей вкладки транскрипции). Язык
/// определяется моделью автоматически (v3 мультиязычная).
pub fn transcribe(model_dir: &Path, wav_path: &str) -> Result<String, String> {
    let audio = load_wav_16k_mono(wav_path)?;
    with_warm_model(model_dir, |model| {
        let result = model
            .transcribe_samples(audio, TARGET_SR, 1, Some(TimestampMode::Sentences))
            .map_err(|e| format!("parakeet transcribe failed: {e}"))?;
        Ok(result.text.trim().to_string())
    })
}

/// Single-speaker: словные таймстемпы -> сегменты по паузам/пунктуации/длине.
pub fn transcribe_segments(model_dir: &Path, wav_path: &str) -> Result<Vec<SpeakerSegment>, String> {
    let audio = load_wav_16k_mono(wav_path)?;
    with_warm_model(model_dir, |model| {
        let words = transcribe_words_warm(model, &audio)?;
        Ok(segment_words(&words, 0.6, 8.0)
            .into_iter()
            .map(|(start, end, text)| SpeakerSegment {
                start,
                end,
                text,
                speaker: 0,
            })
            .collect())
    })
}

/// DIARIZE-FIRST: транскрибировать КАЖДУЮ реплику отдельно (один спикер на
/// сегмент). Клип на turn -> словные таймстемпы -> паузная разбивка внутри turn,
/// чтобы длинный монолог не стал одним гигантским сегментом. Порт transcribe_turns.
pub fn transcribe_turns(
    model_dir: &Path,
    wav_path: &str,
    turns: &[Turn],
) -> Result<Vec<SpeakerSegment>, String> {
    let audio = load_wav_16k_mono(wav_path)?;
    let sr = TARGET_SR as f64;
    let min_len = (0.2 * sr) as usize;
    with_warm_model(model_dir, |model| {
        let mut out = Vec::new();
        for t in turns {
            let a = (t.start * sr) as usize;
            let b = ((t.end * sr) as usize).min(audio.len());
            if b <= a || (b - a) < min_len {
                continue; // слишком коротко для транскрипции
            }
            let clip = &audio[a..b];
            let words = transcribe_words_warm(model, clip)?;
            for (s, e, text) in segment_words(&words, 0.6, 8.0) {
                out.push(SpeakerSegment {
                    start: t.start + s,
                    end: t.start + e,
                    text,
                    speaker: t.speaker,
                });
            }
        }
        Ok(out)
    })
}

/// Диаризация: Sortformer v2 -> реплики [(start,end,speaker)] в секундах, speaker
/// перенумерован 0..k-1. Sortformer отдаёт start/end в СЕМПЛАХ (при 16 кГц).
pub fn diarize(sortformer_onnx: &Path, wav_path: &str) -> Result<Vec<Turn>, String> {
    let audio = load_wav_16k_mono(wav_path)?;
    let mut sf = Sortformer::with_config(
        sortformer_onnx,
        Some(exec_config()),
        DiarizationConfig::callhome(),
    )
    .map_err(|e| format!("sortformer load failed: {e}"))?;
    let segs = sf
        .diarize(audio, TARGET_SR, 1)
        .map_err(|e| format!("sortformer diarize failed: {e}"))?;
    let mut raw: Vec<Turn> = segs
        .iter()
        .map(|s| Turn {
            start: s.start as f64 / TARGET_SR as f64,
            end: s.end as f64 / TARGET_SR as f64,
            speaker: s.speaker_id as i32,
        })
        .collect();
    raw.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Перенумеровать спикеров в контиг. 0..k-1.
    let mut labels: Vec<i32> = raw.iter().map(|t| t.speaker).collect();
    labels.sort_unstable();
    labels.dedup();
    for t in &mut raw {
        t.speaker = labels.iter().position(|&l| l == t.speaker).unwrap_or(0) as i32;
    }
    Ok(raw)
}

/// Результат turns(): слитые реплики + число спикеров.
pub struct DiarTurns {
    pub turns: Vec<Turn>,
    pub n_speakers: usize,
}

/// Порт diarize.turns(): слить подряд идущие реплики одного спикера (merge_gap), и
/// если «настоящих» спикеров (суммарно >= min_speaker_dur) меньше двух — схлопнуть
/// в single-speaker (turns=[], n=1): это штатная graceful-деградация. Иначе
/// перенумеровать спикеров 0..k-1.
pub fn turns(
    sortformer_onnx: &Path,
    wav_path: &str,
    merge_gap: f64,
    min_speaker_dur: f64,
) -> Result<DiarTurns, String> {
    let single = || DiarTurns {
        turns: Vec::new(),
        n_speakers: 1,
    };

    let raw = diarize(sortformer_onnx, wav_path)?;
    if raw.is_empty() {
        return Ok(single());
    }

    // Слить подряд идущие реплики одного спикера с зазором <= merge_gap.
    let mut merged: Vec<[f64; 3]> = vec![[raw[0].start, raw[0].end, raw[0].speaker as f64]];
    for t in &raw[1..] {
        let last = merged.last_mut().unwrap();
        if t.speaker as f64 == last[2] && t.start - last[1] <= merge_gap {
            last[1] = last[1].max(t.end);
        } else {
            merged.push([t.start, t.end, t.speaker as f64]);
        }
    }

    // Суммарная длительность на спикера -> «настоящие» спикеры (>= min_speaker_dur).
    let mut dur: HashMap<i32, f64> = HashMap::new();
    for m in &merged {
        *dur.entry(m[2] as i32).or_insert(0.0) += m[1] - m[0];
    }
    let real: Vec<i32> = dur
        .iter()
        .filter(|(_, &d)| d >= min_speaker_dur)
        .map(|(&s, _)| s)
        .collect();
    if real.len() < 2 {
        return Ok(single()); // реально один голос
    }
    let realset: std::collections::HashSet<i32> = real.iter().copied().collect();

    // Крошечную реплику не-настоящего спикера переназначить ближайшей настоящей.
    let real_turns: Vec<[f64; 3]> = merged
        .iter()
        .filter(|m| realset.contains(&(m[2] as i32)))
        .copied()
        .collect();
    for m in &mut merged {
        if !realset.contains(&(m[2] as i32)) {
            let mid = (m[0] + m[1]) / 2.0;
            let nearest = real_turns
                .iter()
                .min_by(|a, b| {
                    let da = (mid - (a[0] + a[1]) / 2.0).abs();
                    let db = (mid - (b[0] + b[1]) / 2.0).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|x| x[2])
                .unwrap_or(m[2]);
            m[2] = nearest;
        }
    }

    // Перенумеровать метки в 0..k-1 по возрастанию.
    let mut labels: Vec<i32> = merged.iter().map(|m| m[2] as i32).collect();
    labels.sort_unstable();
    labels.dedup();
    let remap: HashMap<i32, i32> = labels.iter().enumerate().map(|(i, &l)| (l, i as i32)).collect();

    let out: Vec<Turn> = merged
        .iter()
        .map(|m| Turn {
            start: m[0],
            end: m[1],
            speaker: remap[&(m[2] as i32)],
        })
        .collect();
    Ok(DiarTurns {
        turns: out,
        n_speakers: labels.len(),
    })
}

/// Полный пайплайн вкладки: диаризация -> транскрипция. Если спикер один
/// (или диаризация схлопнулась) — простая посегментная транскрипция всего клипа.
/// `sortformer_onnx` = None форсирует single-speaker путь (модель не скачана).
pub fn transcribe_and_diarize(
    model_dir: &Path,
    sortformer_onnx: Option<&Path>,
    wav_path: &str,
) -> Result<TranscriptResult, String> {
    let diar = match sortformer_onnx {
        Some(sf) => turns(sf, wav_path, 0.5, 3.0)?,
        None => DiarTurns {
            turns: Vec::new(),
            n_speakers: 1,
        },
    };

    if diar.turns.is_empty() {
        // Один голос: посегментная транскрипция всего клипа.
        let segments = transcribe_segments(model_dir, wav_path)?;
        return Ok(TranscriptResult {
            segments,
            n_speakers: 1,
        });
    }

    let segments = transcribe_turns(model_dir, wav_path, &diar.turns)?;
    Ok(TranscriptResult {
        segments,
        n_speakers: diar.n_speakers,
    })
}

/// Drop the warm TDT model, freeing its RAM (reloaded lazily on next use).
pub fn unload() {
    if let Ok(mut g) = cell().lock() {
        *g = None;
    }
}

// ─── загрузка/подготовка аудио ───────────────────────────────────────────────

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

    if spec.sample_rate == TARGET_SR {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, spec.sample_rate, TARGET_SR))
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

// ─── «Сделать голос»: сборка реф-клипа спикера ───────────────────────────────

/// Отобрать лучшие сегменты одного спикера (длинные, суммарно до `max_total_s`),
/// вырезать их семплы из исходного 16к-моно клипа, склеить с паузами `gap_s` и
/// вернуть готовый reference wav (16 кГц mono, PCM16) + объединённый ref_text.
///
/// `segments` — все сегменты транскрипта (по всем спикерам); берём только
/// принадлежащие `speaker`. Порядок склейки — по времени (естественная речь).
pub fn build_speaker_reference(
    wav_path: &str,
    segments: &[SpeakerSegment],
    speaker: i32,
    max_total_s: f64,
    gap_s: f64,
) -> Result<(Vec<u8>, String), String> {
    let audio = load_wav_16k_mono(wav_path)?;
    let sr = TARGET_SR as f64;

    // Кандидаты этого спикера с непустым текстом.
    let mut cands: Vec<&SpeakerSegment> = segments
        .iter()
        .filter(|s| s.speaker == speaker && !s.text.trim().is_empty() && s.end > s.start)
        .collect();
    if cands.is_empty() {
        return Err("у спикера нет пригодных сегментов".into());
    }

    // Отбор: длинные сначала (лучший референс), пока не наберём max_total_s.
    cands.sort_by(|a, b| {
        (b.end - b.start)
            .partial_cmp(&(a.end - a.start))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut chosen: Vec<&SpeakerSegment> = Vec::new();
    let mut total = 0.0;
    for c in cands {
        if total >= max_total_s {
            break;
        }
        chosen.push(c);
        total += c.end - c.start;
    }
    // Склеивать в хронологическом порядке — естественнее для клона.
    chosen.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let gap = vec![0.0f32; (gap_s * sr).max(0.0) as usize];
    let mut out_samples: Vec<f32> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (i, c) in chosen.iter().enumerate() {
        let a = (c.start * sr) as usize;
        let b = ((c.end * sr) as usize).min(audio.len());
        if b <= a {
            continue;
        }
        if i > 0 {
            out_samples.extend_from_slice(&gap);
        }
        out_samples.extend_from_slice(&audio[a..b]);
        texts.push(c.text.trim().to_string());
    }
    if out_samples.is_empty() {
        return Err("не удалось вырезать семплы спикера".into());
    }

    let wav = crate::audio::encode_pcm16_wav(&out_samples, TARGET_SR as i32, 1);
    let ref_text = texts.join(" ");
    Ok((wav, ref_text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(word: &str, start: f64, end: f64) -> Word {
        Word {
            word: word.into(),
            start,
            end,
        }
    }

    #[test]
    fn splits_on_sentence_end() {
        let words = vec![w("Hello", 0.0, 0.4), w("world.", 0.4, 0.8), w("Next", 0.9, 1.2)];
        let segs = segment_words(&words, 0.6, 8.0);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].2, "Hello world.");
        assert_eq!(segs[1].2, "Next");
    }

    #[test]
    fn splits_on_pause() {
        let words = vec![w("a", 0.0, 0.2), w("b", 2.0, 2.2)]; // пауза 1.8с > 0.6
        let segs = segment_words(&words, 0.6, 8.0);
        assert_eq!(segs.len(), 2);
    }
}
