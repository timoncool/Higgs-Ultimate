use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::collections::HashSet;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

const MAX_RECORDING_SECONDS: u32 = 30;
const METER_EVENTS_PER_SECOND: u32 = 20;

type WaveWriter = hound::WavWriter<BufWriter<std::fs::File>>;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MicrophoneDevice {
    name: String,
    is_default: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingStarted {
    session_id: String,
    device_name: String,
    sample_rate: u32,
    max_seconds: u32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingMeter {
    session_id: String,
    low: f32,
    high: f32,
    peak: f32,
    elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingResult {
    session_id: String,
    path: String,
    device_name: String,
    sample_rate: u32,
    duration_seconds: f64,
    reached_limit: bool,
}

struct ActiveRecording {
    session_id: String,
    stop_tx: std::sync::mpsc::Sender<()>,
    done_rx: Receiver<Result<RecordingResult, String>>,
}

#[derive(Clone, Copy)]
struct MeterSlice {
    low: f32,
    high: f32,
    peak: f32,
    frames: u64,
}

fn active_recording() -> &'static Mutex<Option<ActiveRecording>> {
    static ACTIVE: OnceLock<Mutex<Option<ActiveRecording>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(None))
}

fn unique_recording_path() -> std::path::PathBuf {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    crate::storage::temp_dir().join(format!(
        "higgs_microphone_{}_{}_{}.wav",
        std::process::id(),
        stamp,
        sequence
    ))
}

fn discard_recording_file(writer: &Arc<Mutex<Option<WaveWriter>>>, path: &std::path::Path) {
    if let Ok(mut guard) = writer.lock() {
        guard.take();
    }
    let _ = std::fs::remove_file(path);
}

fn input_device(device_name: Option<&str>) -> Result<(cpal::Device, String), String> {
    let host = cpal::default_host();
    if let Some(requested) = device_name.filter(|name| !name.trim().is_empty()) {
        let devices = host
            .input_devices()
            .map_err(|error| format!("Could not enumerate microphone inputs: {error}"))?;
        for device in devices {
            if device.name().ok().as_deref() == Some(requested) {
                return Ok((device, requested.to_string()));
            }
        }
        return Err(format!(
            "Microphone input '{requested}' is no longer available. Refresh the device list."
        ));
    }

    let device = host
        .default_input_device()
        .ok_or_else(|| "No microphone input is available".to_string())?;
    let name = device
        .name()
        .unwrap_or_else(|_| "Default microphone".to_string());
    Ok((device, name))
}

fn list_input_devices_blocking() -> Result<Vec<MicrophoneDevice>, String> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut seen = HashSet::new();
    let mut devices = Vec::new();

    if let Some(name) = default_name.as_ref() {
        seen.insert(name.clone());
        devices.push(MicrophoneDevice {
            name: name.clone(),
            is_default: true,
        });
    }

    for device in host
        .input_devices()
        .map_err(|error| format!("Could not enumerate microphone inputs: {error}"))?
    {
        let Ok(name) = device.name() else {
            continue;
        };
        if seen.insert(name.clone()) {
            devices.push(MicrophoneDevice {
                is_default: default_name.as_deref() == Some(name.as_str()),
                name,
            });
        }
    }
    Ok(devices)
}

#[allow(clippy::too_many_arguments)]
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    writer: Arc<Mutex<Option<WaveWriter>>>,
    recorded_frames: Arc<AtomicU64>,
    max_frames: u64,
    meter_tx: SyncSender<MeterSlice>,
    error_tx: SyncSender<String>,
    convert: fn(T) -> f32,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channel_count = usize::from(config.channels).max(1);
    let sample_rate = u64::from(config.sample_rate.0);
    let meter_frames = (sample_rate / u64::from(METER_EVENTS_PER_SECOND)).max(1);
    let stream_error_tx = error_tx.clone();
    let mut meter_low = 1.0f32;
    let mut meter_high = -1.0f32;
    let mut meter_peak = 0.0f32;
    let mut meter_slice_frames = 0u64;
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let mut written_frames = recorded_frames.load(Ordering::Relaxed);
                let Ok(mut guard) = writer.lock() else {
                    let _ = error_tx.try_send("Microphone WAV writer lock failed".to_string());
                    return;
                };
                let Some(wav) = guard.as_mut() else {
                    return;
                };

                for frame in data.chunks(channel_count) {
                    if written_frames >= max_frames {
                        break;
                    }
                    let mono = frame.iter().map(|sample| convert(*sample)).sum::<f32>()
                        / channel_count as f32;
                    let mono = mono.clamp(-1.0, 1.0);
                    let pcm = (mono * i16::MAX as f32).round() as i16;
                    if let Err(error) = wav.write_sample(pcm) {
                        let _ =
                            error_tx.try_send(format!("Could not write microphone WAV: {error}"));
                        return;
                    }
                    written_frames += 1;
                    meter_slice_frames += 1;
                    meter_low = meter_low.min(mono);
                    meter_high = meter_high.max(mono);
                    meter_peak = meter_peak.max(mono.abs());

                    if meter_slice_frames >= meter_frames {
                        let _ = meter_tx.try_send(MeterSlice {
                            low: meter_low,
                            high: meter_high,
                            peak: meter_peak,
                            frames: written_frames,
                        });
                        meter_low = 1.0;
                        meter_high = -1.0;
                        meter_peak = 0.0;
                        meter_slice_frames = 0;
                    }
                }
                recorded_frames.store(written_frames, Ordering::Relaxed);
            },
            move |error| {
                let _ = stream_error_tx.try_send(format!("Microphone stream failed: {error}"));
            },
            None,
        )
        .map_err(|error| format!("Could not open microphone input: {error}"))
}

fn run_recording(
    app: AppHandle,
    session_id: String,
    requested_device: Option<String>,
    max_seconds: u32,
    stop_rx: Receiver<()>,
    ready_tx: std::sync::mpsc::Sender<Result<RecordingStarted, String>>,
    done_tx: std::sync::mpsc::Sender<Result<RecordingResult, String>>,
) {
    let (device, device_name) = match input_device(requested_device.as_deref()) {
        Ok(value) => value,
        Err(error) => {
            let _ = ready_tx.send(Err(error));
            return;
        }
    };
    let supported = match device.default_input_config() {
        Ok(config) => config,
        Err(error) => {
            let _ = ready_tx.send(Err(format!("Could not read microphone format: {error}")));
            return;
        }
    };
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let sample_rate = config.sample_rate.0;
    let max_frames = if max_seconds == 0 { u64::MAX } else { u64::from(sample_rate) * u64::from(max_seconds) };
    let path = unique_recording_path();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let writer = match hound::WavWriter::create(&path, spec) {
        Ok(writer) => Arc::new(Mutex::new(Some(writer))),
        Err(error) => {
            let _ = ready_tx.send(Err(format!("Could not create microphone WAV: {error}")));
            return;
        }
    };
    let recorded_frames = Arc::new(AtomicU64::new(0));
    let (meter_tx, meter_rx) = std::sync::mpsc::sync_channel(8);
    let (error_tx, error_rx) = std::sync::mpsc::sync_channel(2);

    let stream_result = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream(
            &device,
            &config,
            writer.clone(),
            recorded_frames.clone(),
            max_frames,
            meter_tx,
            error_tx,
            |sample: f32| sample,
        ),
        cpal::SampleFormat::I16 => build_input_stream(
            &device,
            &config,
            writer.clone(),
            recorded_frames.clone(),
            max_frames,
            meter_tx,
            error_tx,
            |sample: i16| sample as f32 / 32768.0,
        ),
        cpal::SampleFormat::U16 => build_input_stream(
            &device,
            &config,
            writer.clone(),
            recorded_frames.clone(),
            max_frames,
            meter_tx,
            error_tx,
            |sample: u16| (sample as f32 - 32768.0) / 32768.0,
        ),
        other => Err(format!(
            "Microphone sample format {other:?} is not supported; select another input device"
        )),
    };
    let stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            discard_recording_file(&writer, &path);
            let _ = ready_tx.send(Err(error));
            return;
        }
    };
    if let Err(error) = stream.play() {
        drop(stream);
        discard_recording_file(&writer, &path);
        let _ = ready_tx.send(Err(format!("Could not start microphone input: {error}")));
        return;
    }

    let started = RecordingStarted {
        session_id: session_id.clone(),
        device_name: device_name.clone(),
        sample_rate,
        max_seconds,
    };
    if ready_tx.send(Ok(started)).is_err() {
        return;
    }

    let mut failure = None;
    let mut reached_limit = false;
    loop {
        while let Ok(meter) = meter_rx.try_recv() {
            let _ = app.emit(
                "reference-recording-meter",
                RecordingMeter {
                    session_id: session_id.clone(),
                    low: meter.low,
                    high: meter.high,
                    peak: meter.peak,
                    elapsed_seconds: meter.frames as f64 / sample_rate as f64,
                },
            );
        }
        if let Ok(error) = error_rx.try_recv() {
            failure = Some(error);
            break;
        }
        if recorded_frames.load(Ordering::Relaxed) >= max_frames {
            reached_limit = true;
            let _ = app.emit(
                "reference-recording-limit",
                serde_json::json!({ "sessionId": session_id }),
            );
            break;
        }
        match stop_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }
    }

    drop(stream);
    let finalize_result = writer
        .lock()
        .map_err(|_| "Microphone WAV writer lock failed".to_string())
        .and_then(|mut guard| {
            guard
                .take()
                .ok_or_else(|| "Microphone WAV writer was already finalized".to_string())?
                .finalize()
                .map_err(|error| format!("Could not finalize microphone WAV: {error}"))
        });
    if let Err(error) = finalize_result {
        failure = Some(error);
    }

    if let Some(error) = failure {
        let _ = std::fs::remove_file(&path);
        let _ = app.emit(
            "reference-recording-error",
            serde_json::json!({ "sessionId": session_id, "message": error }),
        );
        let _ = done_tx.send(Err(error));
        return;
    }

    let frames = recorded_frames.load(Ordering::Relaxed);
    if frames < u64::from(sample_rate) / 10 {
        let error = "Microphone recording was too short; record at least 0.1 seconds".to_string();
        let _ = std::fs::remove_file(&path);
        let _ = done_tx.send(Err(error));
        return;
    }
    let result = RecordingResult {
        session_id,
        path: path.to_string_lossy().into_owned(),
        device_name,
        sample_rate,
        duration_seconds: frames as f64 / sample_rate as f64,
        reached_limit,
    };
    let _ = done_tx.send(Ok(result));
}

fn start_recording_blocking(
    app: AppHandle,
    device_name: Option<String>,
    max_seconds: u32,
) -> Result<RecordingStarted, String> {
    let mut guard = active_recording()
        .lock()
        .map_err(|_| "Microphone recorder state is unavailable".to_string())?;
    if guard.is_some() {
        return Err("A microphone recording is already active".to_string());
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let session_id = format!("mic_{}_{}", std::process::id(), stamp);
    let (stop_tx, stop_rx) = std::sync::mpsc::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let thread_session = session_id.clone();
    std::thread::spawn(move || {
        run_recording(app, thread_session, device_name, max_seconds, stop_rx, ready_tx, done_tx);
    });

    let started = match ready_rx.recv_timeout(Duration::from_secs(8)) {
        Ok(result) => result?,
        Err(_) => {
            let _ = stop_tx.send(());
            return Err("Microphone did not start within 8 seconds".to_string());
        }
    };
    *guard = Some(ActiveRecording {
        session_id,
        stop_tx,
        done_rx,
    });
    Ok(started)
}

fn stop_recording_blocking() -> Result<RecordingResult, String> {
    let mut guard = active_recording()
        .lock()
        .map_err(|_| "Microphone recorder state is unavailable".to_string())?;
    let active = guard
        .as_ref()
        .ok_or_else(|| "No microphone recording is active".to_string())?;
    let _ = active.stop_tx.send(());
    let result = active
        .done_rx
        .recv_timeout(Duration::from_secs(8))
        .map_err(|_| {
            format!(
                "Microphone recording {} did not finalize within 8 seconds",
                active.session_id
            )
        })?;
    *guard = None;
    result
}

#[tauri::command]
pub async fn list_microphone_devices() -> Result<Vec<MicrophoneDevice>, String> {
    tauri::async_runtime::spawn_blocking(list_input_devices_blocking)
        .await
        .map_err(|error| format!("Microphone device task failed: {error}"))?
}

#[tauri::command]
pub async fn start_reference_recording(
    app: AppHandle,
    device_name: Option<String>,
    max_seconds: Option<u32>,
) -> Result<RecordingStarted, String> {
    // a reference stops at MAX_RECORDING_SECONDS; 0 records until stopped (a recording to transcribe)
    let max_seconds = max_seconds.unwrap_or(MAX_RECORDING_SECONDS);
    tauri::async_runtime::spawn_blocking(move || start_recording_blocking(app, device_name, max_seconds))
        .await
        .map_err(|error| format!("Microphone start task failed: {error}"))?
}

#[tauri::command]
pub async fn stop_reference_recording() -> Result<RecordingResult, String> {
    tauri::async_runtime::spawn_blocking(stop_recording_blocking)
        .await
        .map_err(|error| format!("Microphone stop task failed: {error}"))?
}
