//! Native microphone recording via cpal + hound → mono 16-bit WAV.
//!
//! Desktop WebView2 `getUserMedia` is unreliable, so recording is native (the
//! same cpal+hound path the Tauri recorder plugins use). Supports input-device
//! selection and emits a throttled `rec-level` event (peak 0..1) so the UI can
//! show a live meter. The WAV feeds the reference pipeline + Parakeet ASR.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

struct RecState {
    stop_tx: Sender<()>,
    path: String,
}

fn recorder() -> &'static Mutex<Option<RecState>> {
    static REC: OnceLock<Mutex<Option<RecState>>> = OnceLock::new();
    REC.get_or_init(|| Mutex::new(None))
}

/// List available microphone (input) device names.
#[tauri::command]
pub fn list_input_devices() -> Result<Vec<String>, String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok());
    let mut names = Vec::new();
    if let Some(n) = &default_name {
        names.push(n.clone());
    }
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(n) = d.name() {
                if Some(&n) != default_name.as_ref() {
                    names.push(n);
                }
            }
        }
    }
    Ok(names)
}

#[tauri::command]
pub fn start_recording(app: AppHandle, device: Option<String>) -> Result<(), String> {
    let mut guard = recorder()
        .lock()
        .map_err(|_| "recorder state poisoned".to_string())?;
    if guard.is_some() {
        return Err("Already recording".into());
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("higgs_rec_{ts}.wav"));
    let path_str = path.to_string_lossy().into_owned();

    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let thread_path = path_str.clone();

    std::thread::spawn(move || {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        // Pick the requested device by name, else the system default.
        let device = match device {
            Some(name) => host
                .input_devices()
                .ok()
                .and_then(|mut it| it.find(|d| d.name().map(|n| n == name).unwrap_or(false)))
                .or_else(|| host.default_input_device()),
            None => host.default_input_device(),
        };
        let device = match device {
            Some(d) => d,
            None => {
                let _ = ready_tx.send(Err("No microphone found".into()));
                return;
            }
        };
        let supported = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("microphone config: {e}")));
                return;
            }
        };
        let sample_format = supported.sample_format();
        let ch = (supported.channels() as usize).max(1);
        let sample_rate = supported.sample_rate().0;
        let config: cpal::StreamConfig = supported.into();

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = match hound::WavWriter::create(&thread_path, spec) {
            Ok(w) => Arc::new(Mutex::new(Some(w))),
            Err(e) => {
                let _ = ready_tx.send(Err(format!("create wav: {e}")));
                return;
            }
        };

        // Emit a peak level roughly every 50 ms so the UI meter animates.
        let emit_every = (sample_rate / 20).max(1) as usize;

        macro_rules! make_cb {
            ($ty:ty, $to_f32:expr, $to_i16:expr) => {{
                let w = writer.clone();
                let app = app.clone();
                let mut peak = 0f32;
                let mut frames = 0usize;
                move |data: &[$ty], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut g) = w.lock() {
                        if let Some(writer) = g.as_mut() {
                            for frame in data.chunks(ch) {
                                let mono_f = frame.iter().map(|&s| $to_f32(s)).sum::<f32>() / ch as f32;
                                let a = mono_f.abs();
                                if a > peak {
                                    peak = a;
                                }
                                let _ = writer.write_sample($to_i16(mono_f));
                            }
                        }
                    }
                    frames += data.len() / ch;
                    if frames >= emit_every {
                        let _ = app.emit("rec-level", peak.min(1.0));
                        peak = 0.0;
                        frames = 0;
                    }
                }
            }};
        }

        let err_fn = |e| eprintln!("microphone stream error: {e}");
        let built = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                make_cb!(f32, |s: f32| s, |f: f32| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                make_cb!(i16, |s: i16| s as f32 / i16::MAX as f32, |f: f32| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
                err_fn,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config,
                make_cb!(u16, |s: u16| (s as f32 - 32768.0) / 32768.0, |f: f32| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
                err_fn,
                None,
            ),
            other => {
                let _ = ready_tx.send(Err(format!("unsupported sample format: {other:?}")));
                return;
            }
        };

        let stream = match built {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("open microphone: {e}")));
                return;
            }
        };
        if let Err(e) = stream.play() {
            let _ = ready_tx.send(Err(format!("start microphone: {e}")));
            return;
        }
        let _ = ready_tx.send(Ok(()));

        let _ = stop_rx.recv();
        drop(stream);
        if let Some(w) = writer.lock().ok().and_then(|mut g| g.take()) {
            let _ = w.finalize();
        }
    });

    match ready_rx.recv().map_err(|e| e.to_string())? {
        Ok(()) => {
            *guard = Some(RecState {
                stop_tx,
                path: path_str,
            });
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[tauri::command]
pub fn stop_recording() -> Result<String, String> {
    let mut guard = recorder()
        .lock()
        .map_err(|_| "recorder state poisoned".to_string())?;
    let state = guard.take().ok_or_else(|| "Not recording".to_string())?;
    let _ = state.stop_tx.send(());
    std::thread::sleep(std::time::Duration::from_millis(200));
    Ok(state.path)
}
