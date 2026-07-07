use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("I/O error: {0}")]
    Io(String),
}

#[derive(Clone, Debug)]
pub struct PreparedAudio {
    pub path: String,
    pub duration_seconds: f64,
    pub cropped: bool,
}

/// If the file is already a WAV, return the path as-is.
/// Otherwise, decode it with symphonia and write a temp WAV, return that path.
pub fn ensure_wav(path: &str) -> Result<String, AudioError> {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "wav" {
        return Ok(path.to_string());
    }

    // Decode with symphonia
    let (samples, sample_rate, channels) = decode_any_format(path)?;

    // Write temp WAV
    let temp_dir = std::env::temp_dir();
    let temp_name = format!(
        "higgs_ref_{}.wav",
        p.file_stem().and_then(|s| s.to_str()).unwrap_or("audio")
    );
    let temp_path = temp_dir.join(temp_name);
    let wav_bytes = encode_pcm16_wav(&samples, sample_rate, channels);
    std::fs::write(&temp_path, &wav_bytes).map_err(|e| AudioError::Io(e.to_string()))?;

    Ok(temp_path.to_string_lossy().into_owned())
}

pub fn prepare_reference_wav(
    path: &str,
    normalize: bool,
    target_peak: f32,
    max_seconds: Option<f64>,
) -> Result<PreparedAudio, AudioError> {
    prepare_to_temp_wav(path, normalize, target_peak, max_seconds, "higgs_ref")
}

fn prepare_to_temp_wav(
    path: &str,
    normalize: bool,
    target_peak: f32,
    max_seconds: Option<f64>,
    prefix: &str,
) -> Result<PreparedAudio, AudioError> {
    let p = Path::new(path);
    let (samples, sample_rate, channels) = decode_any_format(path)?;
    let duration_seconds = if sample_rate > 0 && channels > 0 {
        samples.len() as f64 / (sample_rate as f64 * channels as f64)
    } else {
        0.0
    };
    let max_samples = max_seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| {
            (seconds * sample_rate.max(1) as f64 * channels.max(1) as f64)
                .round()
                .max(1.0) as usize
        });
    let cropped = max_samples
        .map(|limit| samples.len() > limit)
        .unwrap_or(false);
    let mut processed: Vec<f32> = if let Some(limit) = max_samples {
        samples.iter().copied().take(limit).collect()
    } else {
        samples
    };

    if normalize {
        let peak = processed
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        let target = target_peak.clamp(0.01, 1.0);
        let gain = if peak > 0.00001 { target / peak } else { 1.0 };
        for sample in &mut processed {
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }

    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "wav" && !normalize && !cropped {
        return Ok(PreparedAudio {
            path: path.to_string(),
            duration_seconds,
            cropped: false,
        });
    }

    let temp_dir = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let temp_name = format!(
        "{}_{}_{}.wav",
        prefix,
        p.file_stem().and_then(|s| s.to_str()).unwrap_or("audio"),
        stamp
    );
    let temp_path = temp_dir.join(temp_name);
    let wav_bytes = encode_pcm16_wav(&processed, sample_rate, channels);
    std::fs::write(&temp_path, &wav_bytes).map_err(|e| AudioError::Io(e.to_string()))?;

    Ok(PreparedAudio {
        path: temp_path.to_string_lossy().into_owned(),
        duration_seconds,
        cropped,
    })
}

pub fn waveform_peaks(path: &str, points: usize) -> Result<Vec<f32>, AudioError> {
    let (samples, _, _) = decode_any_format(path)?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let points = points.clamp(64, 4096).min(samples.len().max(1));
    let mut peaks = Vec::with_capacity(points);
    for i in 0..points {
        let start = i * samples.len() / points;
        let end = ((i + 1) * samples.len() / points)
            .max(start + 1)
            .min(samples.len());
        let peak = samples[start..end]
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()))
            .min(1.0);
        peaks.push(peak);
    }
    Ok(peaks)
}

pub fn decode_to_pcm16_wav(
    path: &str,
    target_sample_rate: Option<i32>,
) -> Result<(Vec<u8>, i32, i32, usize), AudioError> {
    let (samples, source_rate, channels) = decode_any_format(path)?;
    let sample_rate = target_sample_rate
        .filter(|rate| *rate > 0)
        .unwrap_or(source_rate);
    let samples = if sample_rate != source_rate {
        resample_linear_mono(&samples, source_rate, sample_rate)
    } else {
        samples
    };
    let sample_count = samples.len();
    let wav = encode_pcm16_wav(&samples, sample_rate, channels);
    Ok((wav, sample_rate, channels, sample_count))
}

fn resample_linear_mono(samples: &[f32], source_rate: i32, target_rate: i32) -> Vec<f32> {
    if samples.is_empty() || source_rate <= 0 || target_rate <= 0 || source_rate == target_rate {
        return samples.to_vec();
    }
    let ratio = target_rate as f64 / source_rate as f64;
    let out_len = (samples.len() as f64 * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        let next_idx = (src_idx + 1).min(samples.len() - 1);
        out.push(samples[src_idx] * (1.0 - frac) + samples[next_idx] * frac);
    }
    out
}

fn decode_any_format(path: &str) -> Result<(Vec<f32>, i32, i32), AudioError> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| AudioError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let prober = symphonia::default::get_probe();
    let probe_result = prober
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioError::Decode(format!("probe failed: {e}")))?;
    let mut format = probe_result.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Decode("no audio track found".into()))?
        .clone();

    let sample_rate = track.codec_params.sample_rate.unwrap_or(24000) as i32;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::Decode(format!("decoder init failed: {e}")))?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track.id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let num_ch = decoded.spec().channels.count().max(1);
                let num_frames = decoded.frames();
                match decoded {
                    AudioBufferRef::F32(ref buf) => {
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f];
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::S16(ref buf) => {
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f] as f32 / 32768.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::S32(ref buf) => {
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f] as f32 / 2147483648.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::U8(ref buf) => {
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += (buf.chan(c)[f] as f32 - 128.0) / 128.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::F64(ref buf) => {
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f] as f32;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    _ => {
                        // For other formats, use convert to f32 buffer
                        let mut f32_buf = decoded.make_equivalent::<f32>();
                        decoded.convert(&mut f32_buf);
                        for f in 0..num_frames {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += f32_buf.chan(c)[f];
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                }
            }
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        }
    }

    Ok((samples, sample_rate, 1)) // always output mono
}

pub fn encode_pcm16_wav(samples: &[f32], sample_rate: i32, channels: i32) -> Vec<u8> {
    let ch = channels as u16;
    let bits: u16 = 16;
    let data_bytes = (samples.len() * 2) as u32;
    let byte_rate = sample_rate as u32 * ch as u32 * bits as u32 / 8;
    let block_align = ch * bits / 8;

    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        out.extend_from_slice(&((clamped * 32767.0).round() as i16).to_le_bytes());
    }
    out
}
