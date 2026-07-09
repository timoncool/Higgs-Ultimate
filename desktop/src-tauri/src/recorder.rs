//! Native microphone recording via cpal + hound → mono 16-bit WAV.
//!
//! Desktop WebView2 `getUserMedia` is unreliable (flaky permissions, re-prompts,
//! no reset), so recording is done natively in Rust — the same cpal+hound path
//! the Tauri recorder plugins use. The resulting WAV feeds the existing
//! reference pipeline and Parakeet auto-transcription.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

struct RecState {
    stop_tx: Sender<()>,
    path: String,
}

fn recorder() -> &'static Mutex<Option<RecState>> {
    static REC: OnceLock<Mutex<Option<RecState>>> = OnceLock::new();
    REC.get_or_init(|| Mutex::new(None))
}

#[tauri::command]
pub fn start_recording() -> Result<(), String> {
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

    // The cpal Stream is not Send on all platforms, so it is built, played and
    // dropped entirely inside this worker thread; we only cross the boundary
    // with channels.
    std::thread::spawn(move || {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = match host.default_input_device() {
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
        let channels = supported.channels() as usize;
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

        let write_mono = move |w: &Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
                               sample: i16| {
            if let Ok(mut g) = w.lock() {
                if let Some(writer) = g.as_mut() {
                    let _ = writer.write_sample(sample);
                }
            }
        };

        let err_fn = |e| eprintln!("microphone stream error: {e}");
        let ch = channels.max(1);

        let built = match sample_format {
            cpal::SampleFormat::F32 => {
                let w = writer.clone();
                let write_mono = write_mono.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        for frame in data.chunks(ch) {
                            let avg = frame.iter().copied().sum::<f32>() / ch as f32;
                            write_mono(&w, (avg.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
                        }
                    },
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let w = writer.clone();
                let write_mono = write_mono.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        for frame in data.chunks(ch) {
                            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                            write_mono(&w, (sum / ch as i32) as i16);
                        }
                    },
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let w = writer.clone();
                let write_mono = write_mono.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        for frame in data.chunks(ch) {
                            let sum: i32 = frame.iter().map(|&s| s as i32 - 32768).sum();
                            write_mono(&w, (sum / ch as i32) as i16);
                        }
                    },
                    err_fn,
                    None,
                )
            }
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

        // Record until the stop signal, then finalize the WAV.
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
    // Let the worker flush the WAV header/tail before the caller reads the file.
    std::thread::sleep(std::time::Duration::from_millis(200));
    Ok(state.path)
}
