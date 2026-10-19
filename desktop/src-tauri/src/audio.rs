use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
    if has_wave_header(p) {
        return Ok(path.to_string());
    }

    let (samples, sample_rate, channels) = decode_any_format(path)?;
    let temp_path = unique_temp_wav_path("higgs_ref", p);
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

pub fn trim_reference_wav(
    path: &str,
    start_seconds: f64,
    end_seconds: f64,
    max_seconds: f64,
) -> Result<PreparedAudio, AudioError> {
    if !start_seconds.is_finite() || !end_seconds.is_finite() {
        return Err(AudioError::Decode("trim times must be finite".into()));
    }

    let source = Path::new(path);
    let (samples, sample_rate, channels) = decode_any_format(path)?;
    if sample_rate <= 0 || channels <= 0 || samples.is_empty() {
        return Err(AudioError::Decode("reference audio is empty".into()));
    }

    let total_frames = samples.len() / channels as usize;
    let total_seconds = total_frames as f64 / sample_rate as f64;
    let start = start_seconds.clamp(0.0, total_seconds);
    let end = end_seconds.clamp(0.0, total_seconds);
    if end - start < 0.1 {
        return Err(AudioError::Decode(
            "trim selection must be at least 0.10 seconds".into(),
        ));
    }
    if max_seconds > 0.0 && end - start > max_seconds + 0.001 {
        return Err(AudioError::Decode(format!(
            "trim selection cannot exceed {max_seconds:.2} seconds"
        )));
    }

    let start_frame = (start * sample_rate as f64).floor() as usize;
    let end_frame = (end * sample_rate as f64).ceil().min(total_frames as f64) as usize;
    let start_sample = start_frame.saturating_mul(channels as usize);
    let end_sample = end_frame
        .saturating_mul(channels as usize)
        .min(samples.len());
    if start_sample >= end_sample {
        return Err(AudioError::Decode("trim selection is empty".into()));
    }

    let trimmed = &samples[start_sample..end_sample];
    let temp_path = unique_temp_wav_path("higgs_trim", source);
    let wav_bytes = encode_pcm16_wav(trimmed, sample_rate, channels);
    std::fs::write(&temp_path, &wav_bytes).map_err(|e| AudioError::Io(e.to_string()))?;

    Ok(PreparedAudio {
        path: temp_path.to_string_lossy().into_owned(),
        duration_seconds: (end_frame - start_frame) as f64 / sample_rate as f64,
        cropped: true,
    })
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
    let max_samples = max_seconds.filter(|seconds| *seconds > 0.0).map(|seconds| {
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

    if has_wave_header(p) && !normalize && !cropped {
        return Ok(PreparedAudio {
            path: path.to_string(),
            duration_seconds,
            cropped: false,
        });
    }

    let temp_path = unique_temp_wav_path(prefix, p);
    let wav_bytes = encode_pcm16_wav(&processed, sample_rate, channels);
    std::fs::write(&temp_path, &wav_bytes).map_err(|e| AudioError::Io(e.to_string()))?;

    Ok(PreparedAudio {
        path: temp_path.to_string_lossy().into_owned(),
        duration_seconds,
        cropped,
    })
}

fn has_wave_header(path: &Path) -> bool {
    let mut header = [0u8; 12];
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    matches!(&header[..4], b"RIFF" | b"RF64" | b"BW64") && &header[8..] == b"WAVE"
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

fn unique_temp_wav_path(prefix: &str, source: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    crate::storage::temp_dir().join(format!(
        "{prefix}_{stem}_{}_{stamp}_{sequence}.wav",
        std::process::id()
    ))
}

fn decode_opus_stream(
    format: &mut dyn symphonia::core::formats::FormatReader,
    track: &symphonia::core::formats::Track,
    source: &Path,
) -> Result<(Vec<f32>, i32, i32), AudioError> {
    use audiopus::coder::Decoder;
    use audiopus::{Channels, SampleRate};

    let extra_data = track.codec_params.extra_data.as_deref();
    let header_channels = extra_data
        .filter(|header| header.len() >= 19 && &header[..8] == b"OpusHead")
        .map(|header| header[9] as usize);
    let channel_count = header_channels
        .or_else(|| track.codec_params.channels.map(|channels| channels.count()))
        .unwrap_or(1);
    let channels = match channel_count {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        count => {
            return Err(AudioError::Decode(format!(
                "Opus audio with {count} channels is not supported; use mono or stereo"
            )));
        }
    };
    if let Some(header) =
        extra_data.filter(|header| header.len() >= 19 && &header[..8] == b"OpusHead")
    {
        if header[18] != 0 {
            return Err(AudioError::Decode(
                "mapped multi-stream Opus audio is not supported; use mono or stereo Opus".into(),
            ));
        }
    }

    let mut remaining_pre_skip = extra_data
        .filter(|header| header.len() >= 19 && &header[..8] == b"OpusHead")
        .map(|header| u16::from_le_bytes([header[10], header[11]]) as usize)
        .or_else(|| track.codec_params.delay.map(|delay| delay as usize))
        .unwrap_or(0);
    let output_gain = extra_data
        .filter(|header| header.len() >= 19 && &header[..8] == b"OpusHead")
        .map(|header| i16::from_le_bytes([header[16], header[17]]) as f32)
        .map(|gain_q8| 10.0f32.powf(gain_q8 / (20.0 * 256.0)))
        .unwrap_or(1.0);

    let mut decoder = Decoder::new(SampleRate::Hz48000, channels)
        .map_err(|e| AudioError::Decode(format!("failed to initialize the Opus decoder: {e}")))?;
    let mut packet_pcm = vec![0.0f32; 5760 * channel_count];
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                return Err(AudioError::Decode(
                    "the Opus stream changed format while decoding".into(),
                ));
            }
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(AudioError::Decode(format!(
                    "failed to read an Opus packet from {}: {e}",
                    source.display()
                )));
            }
        };
        if packet.track_id() != track.id {
            continue;
        }
        if packet.buf().is_empty() {
            return Err(AudioError::Decode(format!(
                "{} contained an empty Opus packet",
                source.display()
            )));
        }

        let decoded_frames = decoder
            .decode_float(Some(packet.buf()), packet_pcm.as_mut_slice(), false)
            .map_err(|e| {
                AudioError::Decode(format!(
                    "failed to decode an Opus packet from {}: {e}",
                    source.display()
                ))
            })?;
        let packet_trim_start = (packet.trim_start() as usize).min(decoded_frames);
        remaining_pre_skip = remaining_pre_skip.saturating_sub(packet_trim_start);
        let additional_pre_skip = if packet_trim_start == 0 {
            let skip = remaining_pre_skip.min(decoded_frames);
            remaining_pre_skip -= skip;
            skip
        } else {
            0
        };
        let frame_start = packet_trim_start + additional_pre_skip;
        let frame_end = decoded_frames.saturating_sub(packet.trim_end() as usize);
        if frame_start >= frame_end {
            continue;
        }
        for frame in frame_start..frame_end {
            let mut mono = 0.0f32;
            for channel in 0..channel_count {
                mono += packet_pcm[frame * channel_count + channel];
            }
            samples.push((mono / channel_count as f32 * output_gain).clamp(-1.0, 1.0));
        }
    }

    if samples.is_empty() {
        return Err(AudioError::Decode(format!(
            "{} did not contain any decodable Opus samples",
            source.display()
        )));
    }
    Ok((samples, 48000, 1))
}

fn decode_any_format(path: &str) -> Result<(Vec<f32>, i32, i32), AudioError> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL, CODEC_TYPE_OPUS};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let p = Path::new(path);
    let file = std::fs::File::open(p).map_err(|e| AudioError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let prober = symphonia::default::get_probe();
    let format_options = FormatOptions {
        enable_gapless: true,
        ..FormatOptions::default()
    };
    let probe_result = prober
        .format(&hint, mss, &format_options, &MetadataOptions::default())
        .map_err(|e| AudioError::Decode(format!("probe failed: {e}")))?;
    let mut format = probe_result.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Decode("no audio track found".into()))?
        .clone();

    if track.codec_params.codec == CODEC_TYPE_OPUS {
        return decode_opus_stream(format.as_mut(), &track, p);
    }

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(24000) as i32;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::Decode(format!("decoder init failed for {}: {e}", p.display())))?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                return Err(AudioError::Decode(
                    "the audio stream changed format while decoding".into(),
                ));
            }
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(AudioError::Decode(format!(
                    "failed to read an audio packet from {}: {e}",
                    p.display()
                )));
            }
        };

        if packet.track_id() != track.id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                sample_rate = decoded.spec().rate as i32;
                let num_ch = decoded.spec().channels.count().max(1);
                let num_frames = decoded.frames();
                let frame_start = (packet.trim_start() as usize).min(num_frames);
                let frame_end = num_frames.saturating_sub(packet.trim_end() as usize);
                match decoded {
                    AudioBufferRef::F32(ref buf) => {
                        for f in frame_start..frame_end {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f];
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::S16(ref buf) => {
                        for f in frame_start..frame_end {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f] as f32 / 32768.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::S32(ref buf) => {
                        for f in frame_start..frame_end {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += buf.chan(c)[f] as f32 / 2147483648.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::U8(ref buf) => {
                        for f in frame_start..frame_end {
                            let mut v = 0.0f32;
                            for c in 0..num_ch {
                                v += (buf.chan(c)[f] as f32 - 128.0) / 128.0;
                            }
                            samples.push(v / num_ch as f32);
                        }
                    }
                    AudioBufferRef::F64(ref buf) => {
                        for f in frame_start..frame_end {
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
                        for f in frame_start..frame_end {
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
            Err(e) => {
                return Err(AudioError::Decode(format!(
                    "failed to decode an audio packet from {}: {e}",
                    p.display()
                )));
            }
        }
    }

    if samples.is_empty() {
        return Err(AudioError::Decode(format!(
            "{} did not contain any decodable audio samples",
            p.display()
        )));
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

/// Normalize a 16-bit PCM WAV to a target integrated loudness (LUFS, EBU R128),
/// with a peak guard so the gain never clips. Used to level-match multi-speaker
/// segments and continuation clips whose source references differ in volume.
pub fn normalize_wav_bytes(wav: &[u8], target_lufs: f64) -> Result<Vec<u8>, String> {
    use std::io::Cursor;
    let mut reader =
        hound::WavReader::new(Cursor::new(wav)).map_err(|e| format!("read wav: {e}"))?;
    let spec = reader.spec();
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|e| format!("decode wav: {e}"))?;
    if samples.is_empty() {
        return Ok(wav.to_vec());
    }

    let mut ebu = ebur128::EbuR128::new(
        spec.channels as u32,
        spec.sample_rate,
        ebur128::Mode::I,
    )
    .map_err(|e| format!("ebur128: {e}"))?;
    ebu.add_frames_i16(&samples)
        .map_err(|e| format!("ebur128 frames: {e}"))?;
    let loudness = ebu.loudness_global().unwrap_or(f64::NEG_INFINITY);
    if !loudness.is_finite() {
        // Silence / too short to measure — leave untouched.
        return Ok(wav.to_vec());
    }

    let gain = 10f64.powf((target_lufs - loudness) / 20.0);
    // Peak guard: scale the gain down if it would push any sample past full scale.
    let mut peak = 0f64;
    for &s in &samples {
        let v = (s as f64 / 32768.0 * gain).abs();
        if v > peak {
            peak = v;
        }
    }
    let g = if peak > 0.99 { gain * (0.99 / peak) } else { gain };

    let mut out = Cursor::new(Vec::new());
    {
        let mut writer =
            hound::WavWriter::new(&mut out, spec).map_err(|e| format!("write wav: {e}"))?;
        for &s in &samples {
            let v = ((s as f64) * g).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(v).map_err(|e| format!("write sample: {e}"))?;
        }
        writer.finalize().map_err(|e| format!("finalize wav: {e}"))?;
    }
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn invalid_audio_returns_a_decode_error() {
        let path = unique_temp_wav_path("higgs_invalid_audio_test", Path::new("invalid.bin"));
        std::fs::write(&path, b"not an audio file").expect("write invalid fixture");
        let result = decode_any_format(path.to_string_lossy().as_ref());
        let _ = std::fs::remove_file(path);
        assert!(matches!(result, Err(AudioError::Decode(_))));
    }

    #[test]
    fn wave_detection_uses_content_instead_of_extension() {
        let fake_wav = unique_temp_wav_path("higgs_fake_wav_test", Path::new("voice.wav"));
        std::fs::write(&fake_wav, b"not a wave file").expect("write fake WAV");
        assert!(!has_wave_header(&fake_wav));
        assert!(matches!(
            ensure_wav(fake_wav.to_string_lossy().as_ref()),
            Err(AudioError::Decode(_))
        ));
        let _ = std::fs::remove_file(fake_wav);

        let real_wav = unique_temp_wav_path("higgs_real_wav_test", Path::new("voice.bin"));
        std::fs::write(&real_wav, encode_pcm16_wav(&[0.0, 0.25, -0.25], 24_000, 1))
            .expect("write real WAV");
        assert!(has_wave_header(&real_wav));
        assert_eq!(
            ensure_wav(real_wav.to_string_lossy().as_ref()).expect("accept WAV content"),
            real_wav.to_string_lossy()
        );
        let _ = std::fs::remove_file(real_wav);
    }

    #[test]
    fn trims_reference_audio_to_selected_range() {
        let source = unique_temp_wav_path("higgs_trim_source_test", Path::new("voice.wav"));
        let samples = vec![0.25f32; 48_000];
        std::fs::write(&source, encode_pcm16_wav(&samples, 24_000, 1)).expect("write trim fixture");

        let trimmed = trim_reference_wav(source.to_string_lossy().as_ref(), 0.5, 1.25, 30.0)
            .expect("trim reference");
        let (_, sample_rate, channels) = decode_any_format(&trimmed.path).expect("decode trim");
        assert_eq!(sample_rate, 24_000);
        assert_eq!(channels, 1);
        assert!((trimmed.duration_seconds - 0.75).abs() < 0.001);

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(trimmed.path);
    }

    #[test]
    #[ignore = "requires ffmpeg with libopus"]
    fn decodes_ogg_and_webm_opus() {
        for extension in ["ogg", "webm"] {
            let path = unique_temp_wav_path("higgs_opus_test", Path::new("voice.wav"))
                .with_extension(extension);
            let status = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.25",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-c:a",
                    "libopus",
                    "-y",
                ])
                .arg(&path)
                .status()
                .expect("launch ffmpeg");
            assert!(
                status.success(),
                "ffmpeg failed to create {extension} fixture"
            );

            let (samples, sample_rate, channels) =
                decode_any_format(path.to_string_lossy().as_ref()).expect("decode Opus fixture");
            let _ = std::fs::remove_file(path);
            assert!(!samples.is_empty());
            assert!(
                (11_800..=12_200).contains(&samples.len()),
                "unexpected decoded duration for {extension}: {} samples",
                samples.len()
            );
            assert_eq!(sample_rate, 48000);
            assert_eq!(channels, 1);
        }
    }
}
