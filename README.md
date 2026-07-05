# Higgs Audio v3 Studio

![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900)
![UI](https://img.shields.io/badge/UI-Tauri%202%20%2B%20Vite-24C8DB)
![Backend](https://img.shields.io/badge/backend-Rust%20%2B%20libloading-orange)
![Engine](https://img.shields.io/badge/engine-C%2B%2B%20DLL%20via%20C%20ABI-00599C)
![Build](https://img.shields.io/badge/build-MSVC%202022-success)

[中文说明](README_ZH.md)

https://github.com/user-attachments/assets/67a9eeff-415f-4f48-b65c-50c3f9bd2367

Author: [Saganaki22](https://github.com/Saganaki22)

Higgs Audio v3 Studio `0.2.0` is a Windows desktop app built with Rust/Tauri for
local Higgs Audio v3 TTS inference through a ported native C++/CUDA engine. The
app does not shell out to a CLI sidecar: the Tauri UI calls Rust commands, Rust
loads `audiocpp_engine.dll` with `libloading`, and the DLL executes the native
inference path through a small C ABI.

The goal is simple: a practical desktop workflow for local TTS, voice cloning,
speech continuation, and multi-speaker generation without making users manage a
Python environment.

## Downloads

Prebuilt packages are expected to be published from:

- GitHub releases: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/releases
- Hugging Face runtime repository: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio
- Runtime manifest: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/manifest.json
- Checksums: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/checksums/SHA256SUMS.txt

Direct runtime downloads:

| File | Recommended VRAM | Direct link |
| --- | --- | --- |
| Engine DLL | NVIDIA CUDA 13 GPU required | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/engines/audiocpp_engine.dll |
| Higgs Q8_0 recommended | 12 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q8_0/q8_0.gguf |
| Higgs Q6_K | 10 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q6_k/q6_k.gguf |
| Higgs Q5_K | 9 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q5_k/q5_k.gguf |
| Higgs Q4_K_M | 8 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q4_k_m/q4_k_m.gguf |
| Higgs BF16 | 16 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-bf16/bf16.gguf |

Recommended user flow:

1. Download the latest Windows release from GitHub.
2. Launch `Higgs Audio v3 Studio`.
3. Click `Download DLL Engine` if the engine is not bundled.
4. Download or browse to a Higgs model folder.
5. Click `Load Engine`, then `Load Model`.
6. Pick a workflow and generate audio.

<details>
<summary>Hugging Face repository layout</summary>

The Hugging Face repository root should contain `manifest.json` directly at the
top level. Put it beside `models/`, `engines/`, and `checksums/`:

```text
drbaph/Higgs-Audio-v3-Studio/
  manifest.json
  engines/
    audiocpp_engine.dll
  models/
    higgs-q8_0/
      q8_0.gguf
    higgs-q6_k/
      q6_k.gguf
    higgs-q5_k/
      q5_k.gguf
    higgs-q4_k_m/
      q4_k_m.gguf
    higgs-bf16/
      bf16.gguf
  checksums/
    SHA256SUMS.txt
```

Upload your local Hugging Face staging folder with this same structure to the
Hugging Face repo root so the runtime links resolve under `/resolve/main/...`.

</details>

## What It Does

- Runs the ported Higgs Audio v3 C++/CUDA engine inside a Tauri desktop app.
- Supports normal TTS, voice cloning, speech continuation, and multi-speaker workflows.
- Supports reference voice drag/drop, replacement, waveform previews, and remove buttons.
- Supports optional live streaming playback during generation, with the output waveform filling as chunks arrive and play/pause control for the live stream.
- Supports optional Whisper auto-transcription for reference transcripts.
- Includes a Whisper model selector with direct `whisper.cpp` model downloads.
- Includes a Speaker Gallery for reusable speaker identities with reference audio, transcript, notes, display image, normalization, and selected-speaker ZIP import/export.
- Reuses saved speaker reference caches (`.hspkcache`) after first inference to skip repeated reference-code preparation.
- Includes per-line speaker assignment, draggable line ordering, and speaker-line pauses.
- Includes a visible generation queue manager for queued UI jobs, with active-job status, edit, delete, and clear controls.
- Includes a local API with normal WAV responses, NDJSON streaming responses, saved-speaker discovery, and a detachable Command Centre log window.
- Exposes generation controls such as temperature, top-k, top-p, seed mode, max tokens, chunking, emotion, style, speed, pitch, and expressiveness.
- Exports generated audio as WAV or MP3.
- Tracks recent generations per mode.
- Shows NVIDIA hardware telemetry for VRAM, GPU load, power, RAM, and history.
- Uses default-browser links for GitHub, releases, Whisper model selection, and external downloads.

## Supported Systems

<details>
<summary>Windows, GPU, and tooling targets</summary>

Primary target:

- Windows 11 x64
- Tauri 2 / WebView2 desktop runtime
- Visual Studio 2022 MSVC toolchain for source builds
- NVIDIA RTX GPU with CUDA 13 support for the prebuilt CUDA engine
- Current NVIDIA driver compatible with CUDA 13
- Recommended: RTX 30-series, 40-series, or 50-series GPU with enough VRAM for the selected quantization

Likely compatible:

- Windows 10 x64 with current WebView2 and NVIDIA drivers
- Other CUDA-capable NVIDIA GPUs if the DLL was built for their CUDA architecture

Not the focus of this desktop package:

- CPU-only generation
- macOS desktop packaging
- Linux desktop packaging

</details>

## Architecture

<details>
<summary>Runtime architecture and source boundaries</summary>

```text
Tauri Web UI
  -> Rust command layer
    -> libloading + Windows DLL search path setup
      -> audiocpp_engine.dll C API
        -> ported Higgs Audio v3 C++ runtime
          -> ggml / CUDA backend
            -> Higgs Audio v3 TTS, voice clone, continuation
```

Important boundary:

- UI code lives in `desktop/src`.
- Rust command glue lives in `desktop/src-tauri/src`.
- Native C ABI lives in `app/desktop_api/audiocpp_api.h`.
- Native DLL implementation lives in `app/desktop_api/audiocpp_api.cpp`.
- Higgs model code lives under `src/models/higgs_tts`.

Long-running inference runs on Rust blocking worker threads so the WebView stays
responsive while the native engine is generating audio.

</details>

## Runtime Files

<details open>
<summary>Expected runtime file layout</summary>

The desktop app expects:

```text
Higgs Audio v3 Studio.exe
resources/
  engine/
    audiocpp_engine.dll
  higgs-assets/
    higgs-audio-v3-tts-4b/
      config/tokenizer assets
models/
  higgs-q8_0/
    q8_0.gguf
  higgs-q6_k/
    q6_k.gguf
  higgs-q5_k/
    q5_k.gguf
  higgs-q4_k_m/
    q4_k_m.gguf
  higgs-bf16/
    bf16.gguf
models/whisper/
  ggml-base.en-q8_0.bin      # optional, used by auto-transcribe
```

For packaged/portable builds, keep the `resources/` folder beside the executable.
For development, place the engine DLL in `desktop/src-tauri/resources/engine/`.

The in-app `Download DLL Engine` button downloads the engine as
`audiocpp_engine.dll` into the per-user app data engine folder, which avoids
Windows permission errors when the app is installed under `Program Files`.
Bundled/portable resources are still checked automatically when present.

</details>

## Speaker Gallery

<details>
<summary>Reusable speaker identity storage and ZIP import/export</summary>

Speaker Gallery identities are optional. Voice Clone, Continue Speech, and Multi
Speaker can use a saved identity, but users can still upload one-off reference
audio without saving it.

When a speaker identity is created or edited, the app keeps its files in the
user app data speaker store. Each speaker gets a folder named from the speaker
name and internal ID, with:

```text
speakers/
  Speaker_Name_persona_id/
    manifest.json
    reference.wav/mp3/flac
    display.png/jpg/webp
    transcript.txt
    notes.txt
    cache/
      speaker.hspkcache
```

`Export` opens a picker so you can choose exactly which speaker identities to
include. The portable `.zip` contains `manifest.json` plus one
`speakers/<speaker-name>_<id>/` folder per selected identity. `Import` reads that
ZIP, unpacks the audio/image assets back into the app speaker store, and restores
the speaker identity list.

Speaker export/import includes identity metadata, reference audio, transcript,
notes, display images, and the saved `.hspkcache` reference-code cache when it
exists. The cache is created after the first saved-speaker inference and is used
again by Voice Clone, Continue Speech, Multi Speaker, and saved-speaker API jobs.
Model-internal KV-prefix/activation caches are intentionally not serialized,
because those are model/quant specific.

</details>

## Local API

<details>
<summary>HTTP API, streaming, and Command Centre</summary>

The app can run a local API server from the `API` tab. The default base URL is:

```text
http://127.0.0.1:7077/v1
```

Every `/v1` route requires:

```http
Authorization: Bearer <your-api-key>
```

Useful routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | No-auth health check. |
| `GET /v1/status` | Engine, model, queue, and streaming support state. |
| `GET /v1/models` | Local Higgs model folders detected by the app. |
| `GET /v1/higgs/speakers` | Saved speaker identities, including `speaker:<id>` voice names and cache status. |
| `POST /v1/audio/speech` | OpenAI-style plain TTS or saved-speaker voice clone, returned as WAV. |
| `POST /v1/higgs/voice-clone` | Voice clone from a local reference audio path. |
| `POST /v1/higgs/continue-speech` | Continue an existing local audio file. |
| `POST /v1/higgs/audio/stream` | Streaming TTS/clone/continue response as newline-delimited JSON events. |
| `POST /v1/higgs/cancel` | Cancel the active generation. |

`/v1/higgs/audio/stream` emits NDJSON events such as `queued`, `start`,
`progress`, `audio`, `final`, `done`, and `error`. Audio chunks are delivered as
`wavBase64` fields so simple clients can parse progress and audio from one
response stream.

The API tab includes examples for curl, Python, JavaScript, and PowerShell. Its
Command Centre can be popped out into a separate window with filters for info,
warnings, errors, requests, and jobs. If the main studio window is minimized to
the system tray, the popped-out Command Centre remains visible. Speaker Gallery
changes are hot-synced into the running API, so saved speaker IDs do not require
an API restart after create/edit/delete.

</details>

## Install For Users

<details open>
<summary>Portable release</summary>

1. Download the portable release package from GitHub Releases.
2. Put it in a normal writable folder, for example:

   ```text
   C:\AI\Higgs-Audio-v3-Studio\
   ```

3. Keep the `resources/` folder beside `Higgs Audio v3 Studio.exe`.
4. Run `Higgs Audio v3 Studio.exe`.
5. If the engine is missing, click `Download DLL Engine`.
6. Use the Model panel to download or browse to a Higgs model.
7. Load the engine and model.

</details>

<details>
<summary>Installer</summary>

1. Download the `.exe` or `.msi` installer from GitHub Releases.
2. Install normally.
3. Launch the app from the Start Menu.
4. Download or browse to the engine/model files from inside the app.

</details>

## Model Setup

<details open>
<summary>Higgs and Whisper model setup</summary>

The model selector expects model folders, not loose files. A good layout is:

```text
models/
  higgs-q8_0/
    q8_0.gguf
  higgs-q6_k/
    q6_k.gguf
  higgs-q5_k/
    q5_k.gguf
  higgs-q4_k_m/
    q4_k_m.gguf
  higgs-bf16/
    bf16.gguf
```

`Higgs Audio v3 Q8_0` is the recommended model in the app. Shared config and
tokenizer assets can be packaged once with the app under
`desktop/src-tauri/resources/higgs-assets/higgs-audio-v3-tts-4b/`, so the
downloaded model folders only need the `.gguf` files.

Recommended VRAM:

| Model | VRAM |
| --- | --- |
| Higgs Q4_K_M | 8 GB |
| Higgs Q5_K | 9 GB |
| Higgs Q6_K | 10 GB |
| Higgs Q8_0 | 12 GB |
| Higgs BF16 | 16 GB |

Use `Browse...` if a model is somewhere else.
In-app model downloads use a user-writable folder by default:

```text
C:\Users\<you>\audiocpp\models\
```

For Whisper auto-transcription, use the Whisper panel on the left:

1. Select a Whisper preset.
2. The recommended default is `base.en-q8_0`.
3. The smaller turbo option `large-v3-turbo-q5_0` is also highlighted.
4. Click `Download`, or click `Browse...` and select an existing `ggml-*.bin`.

</details>

## Build Requirements

<details>
<summary>Toolchain requirements</summary>

Known-good Windows build inputs:

| Dependency | Version / Notes |
| --- | --- |
| Visual Studio Build Tools | 2022 or newer, MSVC C++ workload |
| Windows SDK | Installed with Visual Studio Build Tools |
| CMake | 3.20+ |
| Ninja | 1.11+ |
| CUDA Toolkit | CUDA 13.x for the prebuilt-compatible CUDA engine |
| Rust | Stable MSVC toolchain |
| Node.js | 20+ recommended |
| WebView2 | Required by Tauri on Windows |

</details>

## Build Native Engine

<details>
<summary>Build the C++ CUDA DLL</summary>

From the repository root:

```powershell
# Build the desktop DLL for a CUDA release preset.
.\scripts\build_windows.ps1 `
  -Preset windows-cuda-release `
  -Target audiocpp_engine `
  -Jobs 16
```

To force CUDA architectures:

```powershell
.\scripts\build_windows.ps1 `
  -Preset windows-cuda-release `
  -Target audiocpp_engine `
  -CudaArchitectures "86;89;120" `
  -Jobs 16
```

Common architecture targets:

| GPU family | CUDA architecture |
| --- | --- |
| RTX 30-series | `86` |
| RTX 40-series | `89` |
| RTX 50-series | `120` or `120a-real`, depending on CUDA/toolchain support |

The DLL is written to:

```text
build/windows-cuda-release/bin/audiocpp_engine.dll
```

For development, copy it to:

```text
desktop/src-tauri/resources/engine/audiocpp_engine.dll
```

For portable release builds, place it beside the final executable.

</details>

## Build Desktop App

<details>
<summary>Build or run the Tauri app</summary>

From the repository root:

```powershell
cd desktop
npm install
npm run build:vite
cd src-tauri
cargo check
cd ..
npm run build
```

Fast local app run:

```powershell
cd desktop
npx tauri dev
```

Build only the frontend:

```powershell
cd desktop
npm run build:vite
```

Build the Tauri app without generating installer bundles:

```powershell
cd desktop
npx tauri build --no-bundle
```

Full Tauri build output is under:

```text
desktop/src-tauri/target/release/
desktop/src-tauri/target/release/bundle/
```

</details>

## Verification

<details>
<summary>Local verification checks</summary>

Useful checks:

```powershell
# Frontend TypeScript + Vite production build
cd desktop
npm run build:vite

# Rust command layer
cd src-tauri
cargo check
cd ..\..

# Native DLL target
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_engine
```

Expected behavior:

- The app opens to the main TTS workflow.
- `Download DLL Engine` downloads `audiocpp_engine.dll`.
- `Load Engine` changes the engine chip from unloaded to loaded.
- `Load Model` enables generation after a valid model folder is selected.
- Voice clone and multi-speaker workflows require reference audio.
- Whisper auto-transcription requires a selected `ggml-*.bin` Whisper model.

</details>

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

### `Port 1420 is already in use`

Another Vite dev server is already running.

Check the port:

```powershell
Get-NetTCPConnection -LocalPort 1420 -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,State,OwningProcess
```

Stop the old process or run the app after that server exits.

### `Engine library not found`

The app could not find `audiocpp_engine.dll`.

Fix one of these:

- Click `Download DLL Engine`.
- Copy the DLL beside the release executable.
- In dev, copy it to `desktop/src-tauri/resources/engine/audiocpp_engine.dll`.

### `Load Model` is disabled

Load the engine first, then select a valid model folder.

The app lists folders containing model weights such as `model.gguf` or
`model.safetensors`. Loose files in the root `models/` folder will not appear as
full model entries.

### CUDA DLL load errors

Make sure:

- NVIDIA driver is current.
- CUDA runtime DLLs required by the engine are on `PATH` or bundled beside the executable.
- The engine DLL was built for your GPU architecture.

### VRAM spikes during generation

Short VRAM spikes can happen during inference. They usually come from temporary
workspace buffers, KV cache growth, CUDA/ggml scratch allocations, audio codec
stages, or graph execution setup. A quantized model can still need extra transient
memory while generating.

If you run out of memory:

- Use a smaller or lower-quantized model.
- Close other GPU-heavy apps.
- Reduce max tokens.
- Disable longform chunking or use smaller chunks.
- Try a smaller reference clip.

### Whisper auto-transcribe does nothing

Select a Whisper model in the left Whisper panel. The recommended default is
`base.en-q8_0`. The model file should be a `ggml-*.bin` from `whisper.cpp`.

### Downloads do not start

Check that the URL is a direct Hugging Face `resolve/main/...` URL, not a
browser `blob/main/...` page.

Good:

```text
https://huggingface.co/user/repo/resolve/main/path/file.bin
```

Not good for direct app download:

```text
https://huggingface.co/user/repo/blob/main/path/file.bin
```

### App links open inside the app instead of browser

Links should be routed through the Rust `open_external_url` command. If a new
link is added, wire it through that command instead of using an in-app WebView
navigation.

</details>

## Responsible Use

Do not use Higgs Audio v3 Studio, Higgs TTS 3, or any voice-cloning workflow to
impersonate people without consent, create malicious or deceptive voices, defraud
others, evade identification, harass people, or cause harm. Generated voices
should be used only where you have the rights and consent needed for the source
voice, transcript, and intended output.

## Project Layout

```text
app/desktop_api/               Native C ABI used by the Tauri app
desktop/                       Tauri 2 + Vite frontend
desktop/src/                   TypeScript UI
desktop/src-tauri/src/         Rust command layer and DLL loader
include/                       Public C++ framework headers
src/models/higgs_tts/          Higgs Audio v3 model implementation
external/ggml/                 Vendored ggml backend sources
external/whisper.cpp/          Whisper submodule used by the engine DLL
scripts/build_windows.ps1      Windows CMake/MSVC build helper
```

## Upstream And Credits

This desktop app builds on:

- The ported Higgs Audio v3 C++/CUDA engine implemented for this Studio app.
- `ggml`, used by the native backend.
- `whisper.cpp`, used for optional local transcription.
- Higgs Audio v3 model work from Boson AI.
- Tauri 2, Rust, Vite, and TypeScript for the desktop shell.

Check upstream model licenses before redistributing model weights or using them
commercially.

## Citation

```bibtex
@misc{bosonai_higgs_audio_tts_v3_2026,
  title  = {Higgs TTS 3: Conversational Speech for Voice AI from Boson AI},
  author = {Boson AI},
  year   = {2026},
  howpublished = {https://huggingface.co/bosonai/higgs-tts-3-4b},
}
```

## License

Application source code in this repository is Apache 2.0; see `LICENSE`.
Higgs TTS 3 model weights and upstream model assets are governed by the Boson
Higgs TTS 3 Research and Non-Commercial License; see the upstream model license.
