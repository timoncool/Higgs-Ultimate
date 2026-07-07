# Linux Branch Update Plan: Match Windows v0.2.4

This file is a handoff checklist for updating the `linux` branch to match the
Windows `main` branch release tagged `v0.2.4`.

Windows v0.2.4 is currently made of these relevant commits on `main`:

- `04e0013` - `Release v0.2.4 app updates`
- `16af86e` - `Improve first-run setup wizard`

The Linux branch currently sits after the Linux packaging work and still reports
`0.2.31` in the app metadata. Use this plan when logging into Ubuntu so the Linux
branch gets the same app behavior without losing Linux-specific engine handling.

## High-Level Goal

Bring the Linux Tauri app behavior up to Windows v0.2.4:

- App version and docs say `0.2.4`.
- Higgs model downloads fetch the whole selected model folder, not only the GGUF.
- Download UI shows `File n/6: filename` plus live per-file progress.
- API streaming can return a final MP3 payload.
- Reference voice uploads are prepared/cropped to the first 30 seconds.
- Continue Speech keeps the full source audio.
- Drag/drop works using Tauri desktop file paths.
- API Command Centre logs clear on a fresh full app launch.
- First-run wizard becomes a real setup checklist with ticks, browse, download,
  load, and refresh actions.

## Do Not Break Linux-Specific Behavior

The Linux branch already has platform-specific engine wording and runtime paths:

- Linux button text uses `Download Engine Files`.
- Linux engine package uses `engines_linux`.
- Linux engine library is `libaudiocpp_engine.so`.
- Linux packaging produces `.deb` and AppImage.

Keep those differences. Do not replace Linux labels with Windows-only `DLL`
wording, except where README text is explicitly discussing Windows.

## Files To Update

Core desktop files:

```text
desktop/index.html
desktop/src/config.ts
desktop/src/main.ts
desktop/src/styles.css
desktop/src/apiExamples.ts
desktop/src/audio.ts        # only if API/example behavior needs UI wiring
desktop/package.json
desktop/package-lock.json
desktop/src-tauri/Cargo.toml
desktop/src-tauri/Cargo.lock
desktop/src-tauri/tauri.conf.json
desktop/src-tauri/src/audio.rs
desktop/src-tauri/src/lib.rs
README.md
README_ZH.md
```

## 1. Bump Version To 0.2.4

Update:

```text
desktop/src/config.ts                         APP_VERSION
desktop/package.json                          version
desktop/package-lock.json                     root/package version
desktop/src-tauri/Cargo.toml                  package.version
desktop/src-tauri/Cargo.lock                  local package version
desktop/src-tauri/tauri.conf.json             version
desktop/index.html                            footer fallback text
README.md / README_ZH.md                      intro and Linux install commands
```

Linux README examples should become:

```bash
sudo apt install ./Higgs-Audio-v3-Studio_0.2.4_amd64.deb
chmod +x Higgs-Audio-v3-Studio_0.2.4_amd64.AppImage
./Higgs-Audio-v3-Studio_0.2.4_amd64.AppImage
```

## 2. Full Model Folder Downloads

Windows v0.2.4 changed Higgs preset downloads from one GGUF file to a package of
six files:

```text
<model>.gguf
chat_template.jinja
config.json
higgs_audio_v2_tokenizer_config.json
tokenizer.json
tokenizer_config.json
```

Add this to `desktop/src/config.ts`:

```ts
export const HIGGS_MODEL_ASSET_FILES = [
  "chat_template.jinja",
  "config.json",
  "higgs_audio_v2_tokenizer_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
];
```

In `desktop/src/main.ts`, port these helper concepts:

- `ttsPresetPackageFiles(preset)`
- `ttsPresetPackageEntries(preset)`
- `ttsPresetFromModelDownloadUrl(url)`
- `activeDownloadFileLabel`

Update `startDownload(kind, url)` for `kind === "model"`:

- If the URL matches a known Higgs preset, iterate the six package entries.
- Call existing `download_model` once per file.
- Keep `destDir` as `models/<preset.folder>`.
- Set `activeDownloadFileLabel` before each file.
- Show the progress text like:

```text
File 1/6: q8_0.gguf · 84.2 MB/s
File 2/6: chat_template.jinja · 0.1 MB/s
```

Fallback behavior:

- Custom pasted URLs should still download as one file using the old
  `modelDownloadTarget(url)` path.

Linux-specific note:

- This feature is platform-neutral. No Linux backend changes are required beyond
  the existing `download_model` command.

## 3. API Streaming Final MP3

Current Linux branch still has streaming hardcoded to WAV. Port the Windows
v0.2.4 change in `desktop/src-tauri/src/lib.rs`.

In `handle_api_stream_request`:

- Replace the manual `format != "wav"` rejection with:

```rust
let format = match api_response_format(&body) {
    Ok(format) => format,
    Err(e) => { ... return 400 ... }
};
```

- Keep live `audio` events as WAV chunks:

```json
{
  "event": "audio",
  "encoding": "wav-base64",
  "wavBase64": "..."
}
```

- Change the queued/start metadata to describe both encodings:

```json
{
  "audioEncoding": "wav-base64",
  "finalEncoding": "mp3-base64"
}
```

- On the final event, use `encode_api_audio_response(&audio, format)`.
- If format is `mp3`, emit:

```json
{
  "event": "final",
  "encoding": "mp3-base64",
  "mp3Base64": "..."
}
```

- If format is `wav`, emit:

```json
{
  "event": "final",
  "encoding": "wav-base64",
  "wavBase64": "..."
}
```

Also update:

```text
desktop/src/apiExamples.ts
desktop/index.html API Info text
README.md
README_ZH.md
```

Explain that streaming live chunks are WAV, while the final clean payload can be
WAV or MP3.

## 4. Reference Audio 30-Second Preparation

Windows v0.2.4 added backend reference preparation so long accidental reference
uploads do not waste setup/inference time.

In `desktop/src-tauri/src/audio.rs`, add:

- `PreparedAudio`
- `prepare_reference_wav(path, normalize, target_peak, max_seconds)`
- an internal helper similar to `prepare_to_temp_wav`

Behavior:

- Decode any supported audio format.
- Convert to mono WAV temp file when needed.
- If `max_seconds` is set, crop samples to that duration.
- If `normalize` is true, peak-normalize to `target_peak`.
- If the source is already WAV and no crop/normalize is needed, return the
  original path.

In `desktop/src-tauri/src/lib.rs`, add:

```rust
const REFERENCE_MAX_SECONDS: f64 = 30.0;
```

Split preparation into:

- `prepare_voice_reference_audio(...)` - uses `Some(REFERENCE_MAX_SECONDS)`
- `prepare_continuation_audio(...)` - uses `None`

Apply it like this:

- Voice Clone: cropped to 30 seconds.
- Saved speaker references: cropped to 30 seconds.
- API voice clone/saved-speaker voice clone: cropped to 30 seconds.
- Continue Speech: not cropped.
- API continue speech: not cropped.

Add a Tauri command:

```rust
#[tauri::command]
async fn prepare_reference_upload(audio_path: String, max_seconds: Option<f64>)
    -> Result<serde_json::Value, String>
```

Return:

```json
{
  "path": "...",
  "fileName": "...",
  "durationSeconds": 12.34,
  "cropped": true
}
```

Register it in `tauri::generate_handler!`.

## 5. Frontend Reference Prep

In `desktop/src/config.ts`, add:

```ts
export const HIGGS_REFERENCE_MAX_SECONDS = 30;
```

In `desktop/src/main.ts`, add:

- `PreparedReferenceUpload` type
- `prepareReferenceAudioFile(path, name?)`
- `pickReferenceAudioFile()`

Use `prepareReferenceAudioFile` for:

- Voice Clone dropzone/browse
- Speaker Gallery dropzone/browse
- Multi Speaker speaker reference browse/drop
- Multi Speaker line reference override browse/drop

Do not use it for:

- Continue Speech source audio

Update UI hints:

```text
mp3 · wav · flac · m4a — auto-cropped to 30s
```

## 6. Tauri Desktop Drag/Drop

Browser `DataTransfer.files[0].path` is unreliable in Tauri/WebView. Windows
v0.2.4 fixed this by routing Tauri native drag/drop events.

In `desktop/src/main.ts`:

- Import `getCurrentWebview` from `@tauri-apps/api/webview`.
- Add a `dropzoneHandlers` registry.
- Update `setupDropzone` so callbacks can be async.
- Add `initTauriDropzones()`.
- Use `getCurrentWebview().onDragDropEvent(...)`.
- Use `document.elementFromPoint(...)` with `devicePixelRatio` fallback to find
  the drop target.
- Route dropped paths to:
  - simple registered dropzones
  - dynamic Multi Speaker speaker reference dropzones
  - dynamic Multi Speaker line reference dropzones

Important:

- Keep the browser `drop` fallback, but do not show a scary warning when it has
  no path. The Tauri event may be the one that actually succeeds.

## 7. API Logs Clear On Fresh Launch

In `desktop/src/main.ts`:

- Add `clearApiLogsForFreshSession()`.
- It should clear the `apiLogs` array and remove `API_LOG_STORAGE_KEY`.
- Call it at main app startup.
- Call it before `quit_app` from the settings Quit button.

Do not break the popout Command Centre:

- The popout still needs localStorage while the app is running so it can sync
  live logs.

## 8. First-Run Wizard Upgrade

The old wizard was static. Windows v0.2.4 now has a setup checklist that detects
whether engine/model files already exist.

Add to `desktop/src/config.ts`:

```ts
export const ENGINE_PATH_STORAGE_KEY = "higgsAudio.selectedEnginePath";
export const FIRST_RUN_WIZARD_STORAGE_KEY = "higgsAudio.setupWizardDismissed";
```

Update `desktop/index.html`:

- Replace the simple three-step text list with checklist rows:
  - Engine files
  - Higgs model folder
  - Whisper auto-transcribe, optional
- Each row should have:
  - status icon
  - status detail text
  - action buttons

Actions:

```text
Engine row:
  Browse
  Download
  Load

Model row:
  Browse
  Download
  Load

Whisper row:
  Download

Footer:
  Refresh
  Close
  Open Studio
```

Update `desktop/src/styles.css`:

- Add `.setup-check-list`
- Add `.setup-check-row`
- Add `.setup-check-icon`
- Add ready/missing/optional row styles
- Ensure action buttons wrap on narrow windows

Update `desktop/src/main.ts`:

- `doLoadEngine(libraryPath?: string)` should use:
  1. explicit path
  2. saved `ENGINE_PATH_STORAGE_KEY`
  3. `bundled_engine_path`
- If saved path is stale, fall back to bundled/downloaded engine when possible.
- Add `doBrowseEngine()`.
- Change `doBrowseModel()` to return the selected folder path.
- Add:
  - `setSetupCheck(...)`
  - `refreshSetupWizardState()`
  - async `maybeShowSetupWizard()`
  - action handlers for Browse/Download/Load/Refresh

Linux-specific wording:

- Use existing `IS_WINDOWS`, `ENGINE_LIB_WORD`, or equivalent branch logic.
- The wizard button/detail should say `Engine Files` or `Engine Libraries` on
  Linux, not Windows-only `DLLs`.

## 9. Documentation Updates

Update both READMEs:

- Version `0.2.4`.
- Full model-folder downloads.
- `File n/6` model download progress.
- 30-second reference preparation/crop.
- Continue Speech source audio is not cropped.
- Streaming endpoint live chunks are WAV, final can be WAV or MP3.
- First-run wizard is a checklist with browse/download/load actions.
- Linux install commands use `0.2.4`.
- Linux engine package still points to `engines_linux`.

Keep existing warnings:

- Do not create malicious voices.
- Respect Boson Higgs license.
- CUDA 13 NVIDIA GPU requirement.
- VRAM recommendations.

## 10. Build And Test On Ubuntu

From repo root:

```bash
git checkout linux
git pull
cd desktop
npm ci
npm run build:vite
cd src-tauri
cargo check
cd ../..
```

Package:

```bash
cd desktop
npm run build
```

Expected Linux outputs:

```text
desktop/src-tauri/target/release/bundle/deb/Higgs Audio v3 Studio_0.2.4_amd64.deb
desktop/src-tauri/target/release/bundle/appimage/Higgs Audio v3 Studio_0.2.4_amd64.AppImage
```

## 11. Manual Smoke Test Checklist

On Ubuntu:

- Launch AppImage.
- First-run wizard appears only if not dismissed.
- Wizard shows engine status:
  - tick if Linux engine exists and dependencies are okay
  - missing if no engine
  - useful message if engine exists but dependencies fail
- Wizard Browse Engine can pick `libaudiocpp_engine.so`.
- Wizard Download Engine uses Linux `engines_linux`.
- Wizard shows detected Higgs model folders if present.
- Wizard Browse Model can pick a model directory.
- Wizard Download Model downloads six files into the selected preset folder.
- During model download, UI shows `File 1/6`, `File 2/6`, etc.
- Voice Clone reference over 30 seconds is cropped and preview uses the cropped
  audio.
- Continue Speech source audio over 30 seconds is not cropped.
- Drag/drop onto Voice Clone works.
- Drag/drop onto Speaker Gallery works.
- Drag/drop onto Multi Speaker dynamic reference boxes works.
- API `/v1/audio/speech` returns MP3 when requested.
- API `/v1/higgs/audio/stream` emits live WAV chunks and final MP3 when
  `response_format: "mp3"`.
- API log is empty after fully quitting and relaunching.

## 12. Suggested Git Flow

After porting and testing:

```bash
git status
git add README.md README_ZH.md desktop/index.html desktop/package.json \
  desktop/package-lock.json desktop/src/config.ts desktop/src/main.ts \
  desktop/src/styles.css desktop/src/apiExamples.ts \
  desktop/src-tauri/Cargo.toml desktop/src-tauri/Cargo.lock \
  desktop/src-tauri/tauri.conf.json desktop/src-tauri/src/audio.rs \
  desktop/src-tauri/src/lib.rs

git commit -m "Update Linux app to v0.2.4"
git tag -a linux-v0.2.4 -m "Higgs Audio v3 Studio Linux v0.2.4"
git push origin linux
git push origin linux-v0.2.4
```

If using the `studio` remote instead of `origin`:

```bash
git push studio linux
git push studio linux-v0.2.4
```

## 13. Avoid These Mistakes

- Do not use Windows-only `audiocpp_engine.dll` paths on Linux.
- Do not point Linux engine downloads at `engines/`; use `engines_linux/`.
- Do not crop Continue Speech source audio.
- Do not make streaming audio chunks MP3; only the final event can be MP3.
- Do not remove the saved speaker `.hspkcache` behavior.
- Do not commit local diagnostics, temporary test outputs, or packaged model
  weights into the source repo.
