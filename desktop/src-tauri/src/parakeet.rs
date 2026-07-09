//! Native Parakeet ASR via parakeet.cpp's flat C-API (`parakeet_capi_*`),
//! loaded in-process from `resources/parakeet/parakeet.dll` (ggml is linked
//! inside that single DLL). Replaces whisper.cpp.
//!
//! The model is loaded ONCE into a warm session and reused across calls (the
//! C-API is designed this way — "loaded once and reused"), so only the first
//! transcription pays the ~1s model-load cost. The session is guarded by a
//! Mutex, so transcriptions are serialized (never concurrent on one ctx).

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

type LoadFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type TranscribeFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int) -> *mut c_char;
type FreeStrFn = unsafe extern "C" fn(*mut c_char);
type FreeCtxFn = unsafe extern "C" fn(*mut c_void);
type LastErrFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;

#[cfg(windows)]
mod imp {
    use super::*;
    use libloading::os::windows::{Library, Symbol};

    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

    /// A warm Parakeet session: the loaded DLL + a loaded model context, kept
    /// alive and reused across transcriptions.
    struct Session {
        _lib: Library, // keep the DLL mapped for the lifetime of the ctx
        ctx: *mut c_void,
        gguf: String,
        transcribe: Symbol<TranscribeFn>,
        free_str: Symbol<FreeStrFn>,
        free_ctx: Symbol<FreeCtxFn>,
        last_err: Symbol<LastErrFn>,
    }

    // Raw pointers are only ever touched while holding the session Mutex, so
    // access is serialized and moving the session between threads is sound.
    unsafe impl Send for Session {}

    impl Drop for Session {
        fn drop(&mut self) {
            unsafe { (self.free_ctx)(self.ctx) }
        }
    }

    fn cell() -> &'static Mutex<Option<Session>> {
        static S: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
        S.get_or_init(|| Mutex::new(None))
    }

    fn open_session(dll_path: &Path, gguf_path: &str) -> Result<Session, String> {
        unsafe {
            let lib = Library::load_with_flags(dll_path, LOAD_WITH_ALTERED_SEARCH_PATH)
                .map_err(|e| format!("failed to load parakeet.dll: {e}"))?;
            let load: Symbol<LoadFn> = lib
                .get(b"parakeet_capi_load\0")
                .map_err(|e| format!("symbol parakeet_capi_load: {e}"))?;
            let transcribe: Symbol<TranscribeFn> = lib
                .get(b"parakeet_capi_transcribe_path\0")
                .map_err(|e| format!("symbol parakeet_capi_transcribe_path: {e}"))?;
            let free_str: Symbol<FreeStrFn> = lib
                .get(b"parakeet_capi_free_string\0")
                .map_err(|e| format!("symbol parakeet_capi_free_string: {e}"))?;
            let free_ctx: Symbol<FreeCtxFn> = lib
                .get(b"parakeet_capi_free\0")
                .map_err(|e| format!("symbol parakeet_capi_free: {e}"))?;
            let last_err: Symbol<LastErrFn> = lib
                .get(b"parakeet_capi_last_error\0")
                .map_err(|e| format!("symbol parakeet_capi_last_error: {e}"))?;

            let gguf_c = CString::new(gguf_path).map_err(|e| e.to_string())?;
            let ctx = load(gguf_c.as_ptr());
            if ctx.is_null() {
                return Err(format!("parakeet: failed to load model: {gguf_path}"));
            }
            Ok(Session {
                _lib: lib,
                ctx,
                gguf: gguf_path.to_string(),
                transcribe,
                free_str,
                free_ctx,
                last_err,
            })
        }
    }

    pub fn transcribe(
        dll_path: &Path,
        gguf_path: &str,
        wav_path: &str,
        decoder: i32,
    ) -> Result<String, String> {
        let mut guard = cell()
            .lock()
            .map_err(|_| "parakeet state poisoned".to_string())?;

        // Load once; reload only if a different model path is requested.
        let need_reload = guard.as_ref().map(|s| s.gguf != gguf_path).unwrap_or(true);
        if need_reload {
            *guard = Some(open_session(dll_path, gguf_path)?);
        }
        let session = guard.as_ref().expect("session present after load");

        unsafe {
            let wav_c = CString::new(wav_path).map_err(|e| e.to_string())?;
            let out = (session.transcribe)(session.ctx, wav_c.as_ptr(), decoder as c_int);
            if out.is_null() {
                let e = (session.last_err)(session.ctx);
                let msg = if e.is_null() {
                    "unknown error".to_string()
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                };
                return Err(format!("parakeet transcribe failed: {msg}"));
            }
            let text = CStr::from_ptr(out).to_string_lossy().into_owned();
            (session.free_str)(out);
            Ok(text.trim().to_string())
        }
    }

    /// Drop the warm model, freeing its RAM (reloaded lazily on next use).
    pub fn unload() {
        if let Ok(mut g) = cell().lock() {
            *g = None;
        }
    }
}

#[cfg(windows)]
pub fn transcribe(dll: &Path, gguf: &str, wav: &str, decoder: i32) -> Result<String, String> {
    imp::transcribe(dll, gguf, wav, decoder)
}

#[cfg(windows)]
pub fn unload() {
    imp::unload();
}

#[cfg(not(windows))]
pub fn transcribe(_: &Path, _: &str, _: &str, _: i32) -> Result<String, String> {
    Err("Parakeet ASR is only wired for Windows in this build".to_string())
}

#[cfg(not(windows))]
pub fn unload() {}
