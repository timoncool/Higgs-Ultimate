//! Native Parakeet ASR via parakeet.cpp's flat C-API (`parakeet_capi_*`),
//! loaded in-process from `resources/parakeet/parakeet.dll` (ggml is linked
//! inside that single DLL). Replaces whisper.cpp.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

type LoadFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type TranscribeFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int) -> *mut c_char;
type FreeStrFn = unsafe extern "C" fn(*mut c_char);
type FreeCtxFn = unsafe extern "C" fn(*mut c_void);
type LastErrFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;

#[cfg(windows)]
const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

/// Transcribe a WAV file with a Parakeet GGUF model. `decoder`: 0 = default
/// (TDT for tdt models), 1 = force CTC, 2 = force TDT/RNN-T.
#[cfg(windows)]
pub fn transcribe(
    dll_path: &Path,
    gguf_path: &str,
    wav_path: &str,
    decoder: i32,
) -> Result<String, String> {
    use libloading::os::windows::{Library, Symbol};
    unsafe {
        let lib = Library::load_with_flags(dll_path, LOAD_WITH_ALTERED_SEARCH_PATH)
            .map_err(|e| format!("failed to load parakeet.dll: {e}"))?;
        let load: Symbol<LoadFn> = lib
            .get(b"parakeet_capi_load\0")
            .map_err(|e| format!("symbol parakeet_capi_load: {e}"))?;
        let transcribe_path: Symbol<TranscribeFn> = lib
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
        let wav_c = CString::new(wav_path).map_err(|e| e.to_string())?;
        let out = transcribe_path(ctx, wav_c.as_ptr(), decoder as c_int);
        let result = if out.is_null() {
            let e = last_err(ctx);
            let msg = if e.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            };
            Err(format!("parakeet transcribe failed: {msg}"))
        } else {
            let text = CStr::from_ptr(out).to_string_lossy().into_owned();
            free_str(out);
            Ok(text.trim().to_string())
        };
        free_ctx(ctx);
        result
    }
}

#[cfg(not(windows))]
pub fn transcribe(_: &Path, _: &str, _: &str, _: i32) -> Result<String, String> {
    Err("Parakeet ASR is only wired for Windows in this build".to_string())
}
