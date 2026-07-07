# Higgs Audio v3 Studio

![Platform](https://img.shields.io/badge/platform-Windows%20x64%20%7C%20Linux%20x64-0078D4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900)
![UI](https://img.shields.io/badge/UI-Tauri%202%20%2B%20Vite-24C8DB)
![Backend](https://img.shields.io/badge/backend-Rust%20%2B%20libloading-orange)
![Engine](https://img.shields.io/badge/engine-C%2B%2B%20via%20C%20ABI-00599C)
![Build](https://img.shields.io/badge/build-MSVC%202022%20%7C%20GCC%2FClang-success)

[中文说明](README_ZH.md) | [Linux branch](https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux)

https://github.com/user-attachments/assets/67a9eeff-415f-4f48-b65c-50c3f9bd2367

Author: [Saganaki22](https://github.com/Saganaki22)

Higgs Audio v3 Studio `0.2.40` is a Windows and Linux desktop app built with
Rust/Tauri for local Higgs Audio v3 TTS inference through a ported native
C++/CUDA engine. The app does not shell out to a CLI sidecar: the Tauri UI calls
Rust commands, Rust loads `audiocpp_engine.dll` (Windows) or
`libaudiocpp_engine.so` (Linux) with `libloading`, and the engine executes the
native inference path through a small C ABI.

This `main` branch tracks the Windows desktop release. Linux `.deb` and AppImage
builds are available from the same GitHub Releases page and are built from the
[`linux`](https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux)
branch.

The goal is simple: a practical desktop workflow for local TTS, voice cloning,
speech continuation, and multi-speaker generation without making users manage a
Python environment.

## Downloads

Prebuilt packages are published from:

- GitHub releases: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/releases
- Linux branch source: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux
- Hugging Face runtime repository: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio
- Runtime manifest: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/manifest.json
- Checksums: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/checksums/SHA256SUMS.txt

Direct runtime downloads:

| File | Recommended VRAM | Direct link |
| --- | --- | --- |
| Engine package (Windows) | NVIDIA CUDA 13 GPU/driver required | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/tree/main/engines |
| Engine package (Linux) | NVIDIA CUDA 13 GPU/driver required | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/tree/main/engines_linux |
| Higgs Q8_0 recommended | 12 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q8_0/q8_0.gguf |
| Higgs Q6_K | 10 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q6_k/q6_k.gguf |
| Higgs Q5_K | 9 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q5_k/q5_k.gguf |
| Higgs Q4_K_M | 8 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q4_k_m/q4_k_m.gguf |
| Higgs BF16 | 16 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-bf16/bf16.gguf |

Recommended user flow:

1. Download the latest release from GitHub (Windows `.exe` installer or Linux `.deb`/AppImage).
2. Launch `Higgs Audio v3 Studio`.
3. Click `Download Engine DLLs` (Windows) or `Download Engine Files` (Linux) if the engine package is not installed.
4. Download or browse to a Higgs model folder. In-app Higgs downloads fetch the whole selected folder: GGUF weights plus config/tokenizer/chat-template assets.
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
    cublas64_13.dll
    cublasLt64_13.dll
    VCOMP140.DLL
    MSVCP140.dll
    VCRUNTIME140.dll
    VCRUNTIME140_1.dll
  engines_linux/
    libaudiocpp_engine.so
    libcudart.so.13
    libcublas.so.13
    libcublasLt.so.13
  models/
    higgs-q8_0/
      q8_0.gguf
      config/tokenizer/chat-template assets
    higgs-q6_k/
      q6_k.gguf
      config/tokenizer/chat-template assets
    higgs-q5_k/
      q5_k.gguf
      config/tokenizer/chat-template assets
    higgs-q4_k_m/
      q4_k_m.gguf
      config/tokenizer/chat-template assets
    higgs-bf16/
      bf16.gguf
      config/tokenizer/chat-template assets
  checksums/
    SHA256SUMS.txt
```

Upload your local Hugging Face staging folder with this same structure to the
Hugging Face repo root so the runtime links resolve under `/resolve/main/...`.

</details>

## What It Does

- Runs the ported Higgs Audio v3 C++/CUDA engine inside a Tauri desktop app.
- Supports normal TTS, voice cloning, speech continuation, and multi-speaker workflows.
- Supports reference voice drag/drop, replacement, waveform previews, remove buttons, and automatic 30-second reference preparation for voice cloning and Speaker Gallery uploads.
- Supports optional live streaming playback during generation, with de-clicked chunk edges, waveform scrubbing, and play/pause control for the live stream.
- Supports optional Whisper auto-transcription for reference transcripts.
- Includes a Whisper model selector with direct `whisper.cpp` model downloads.
- Includes a Speaker Gallery for reusable speaker identities with reference audio, transcript, notes, display image, normalization, and selected-speaker ZIP import/export.
- Reuses saved speaker reference caches (`.hspkcache`) after first inference to skip repeated reference-code preparation.
- Includes per-line speaker assignment, draggable line ordering, speaker-line pauses, line-by-line generation progress, and preflight validation for missing speaker references.
- Includes a visible generation queue manager for queued UI jobs, with active-job status, edit, delete, and clear controls.
- Includes a local API with normal WAV/MP3 responses, NDJSON streaming responses, saved-speaker discovery, and a detachable Command Centre log window.
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

Linux target:

- Ubuntu 24.04 LTS x64 (recommended), Ubuntu 22.04 LTS acceptable
- Tauri 2 / WebKitGTK 4.1 desktop runtime
- GCC 13+ or Clang 18+ toolchain for source builds
- NVIDIA RTX GPU with CUDA 13 support for the prebuilt CUDA engine
- Current NVIDIA driver compatible with CUDA 13
- Recommended: RTX 30-series, 40-series, or 50-series GPU with enough VRAM for the selected quantization

Likely compatible:

- Windows 10 x64 with current WebView2 and NVIDIA drivers
- Other Debian/Ubuntu-based Linux distributions with WebKitGTK 4.1
- Other CUDA-capable NVIDIA GPUs if the engine was built for their CUDA architecture

Not the focus of this desktop package:

- CPU-only generation
- macOS desktop packaging

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
<summary>Runtime file locations</summary>

Portable/packaged app folder:

```text
Higgs Audio v3 Studio.exe
resources/
  engine/
    audiocpp_engine.dll          # bundled main engine DLL
  higgs-assets/
    higgs-audio-v3-tts-4b/
      config/tokenizer assets
```

Engine package downloaded by `Download Engine DLLs`:

```text
%LOCALAPPDATA%/
  Higgs Audio v3 Studio/
    engine/
      audiocpp_engine.dll
      cublas64_13.dll
      cublasLt64_13.dll
      MSVCP140.dll
      VCOMP140.DLL
      VCRUNTIME140.dll
      VCRUNTIME140_1.dll
```

Default downloaded model folders:

```text
models/
  higgs-q8_0/
    q8_0.gguf
    config/tokenizer/chat-template assets
  higgs-q6_k/
    q6_k.gguf
    config/tokenizer/chat-template assets
  higgs-q5_k/
    q5_k.gguf
    config/tokenizer/chat-template assets
  higgs-q4_k_m/
    q4_k_m.gguf
    config/tokenizer/chat-template assets
  higgs-bf16/
    bf16.gguf
    config/tokenizer/chat-template assets
models/whisper/
  ggml-base.en-q8_0.bin      # optional, used by auto-transcribe
```

For packaged/portable builds, keep the `resources/` folder beside the executable.
For development, place the engine DLL in `desktop/src-tauri/resources/engine/`.

The in-app `Download Engine DLLs` button downloads the engine package into the
per-user app data engine folder, which avoids Windows permission errors when the
app is installed under `Program Files`. The package includes
`audiocpp_engine.dll`, cuBLAS/cuBLASLt CUDA 13 runtime DLLs, and the MSVC/OpenMP
runtime DLLs required by the current engine build. Bundled/portable resources
and system-installed CUDA/MSVC runtime folders are still checked automatically
when present.

</details>

<details>
<summary>Linux runtime file locations</summary>

Portable binary or installed package:

```text
higgs-audio-studio
resources/
  higgs-assets/
    higgs-audio-v3-tts-4b/
      config/tokenizer assets
```

Engine package downloaded by `Download Engine Files`:

```text
~/.higgs-audio-v3-studio/
  engine/
    libaudiocpp_engine.so
    libcudart.so.13
    libcublas.so.13
    libcublasLt.so.13
```

Default downloaded model folders:

```text
~/audiocpp/models/
  higgs-q8_0/
    q8_0.gguf
  ...
```

The engine `.so` files use `rpath=$ORIGIN` so they find the bundled CUDA runtime
libraries next to them without requiring a system-wide CUDA Toolkit install. Users
only need a working NVIDIA driver.

</details>

## Speaker Gallery

<details>
<summary>Reusable speaker identity storage and ZIP import/export</summary>

Speaker Gallery identities are optional. Voice Clone, Continue Speech, and Multi
Speaker can use a saved identity, but users can still upload one-off reference
audio without saving it. One-off voice-clone references and Speaker Gallery
uploads are automatically prepared as WAV and capped to the first 30 seconds so
long accidental uploads do not waste inference setup time. Continue Speech keeps
the full source audio because that workflow may intentionally continue longer
material.

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
| `POST /v1/audio/speech` | OpenAI-style plain TTS or saved-speaker voice clone, returned as WAV or MP3. |
| `POST /v1/higgs/voice-clone` | Voice clone from a local reference audio path. |
| `POST /v1/higgs/continue-speech` | Continue an existing local audio file. |
| `POST /v1/higgs/audio/stream` | Streaming TTS/clone/continue response as newline-delimited JSON events. |
| `POST /v1/higgs/cancel` | Cancel the active generation. |

Finished-file routes accept `response_format: "wav"` or `response_format: "mp3"` and return `audio/wav` or `audio/mpeg` directly.

`/v1/higgs/audio/stream` emits NDJSON events such as `queued`, `start`,
`progress`, `audio`, `final`, `done`, and `error`. Audio chunks are delivered as
`wavBase64` fields so simple clients can parse progress and audio from one
response stream. The live chunks are WAV. The `final` event respects
`response_format`: it returns `wavBase64` for WAV or `mp3Base64` for MP3.

For script playback, read the HTTP response line-by-line. When `event` is
`audio`, base64-decode `wavBase64` and feed those WAV bytes to your player or
audio queue. When `event` is `final`, check `encoding`. Save `wavBase64` when it
is `wav-base64`, or `mp3Base64` when it is `mp3-base64`.

Minimal Python stream reader:

```python
import base64
import json
import requests

with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        event = json.loads(line)
        if event["event"] == "audio":
            wav_chunk = base64.b64decode(event["wavBase64"])
            # Push wav_chunk to your audio playback queue here.
        elif event["event"] == "final":
            if event.get("encoding") == "mp3-base64":
                open("final.mp3", "wb").write(base64.b64decode(event["mp3Base64"]))
            else:
                open("final.wav", "wb").write(base64.b64decode(event["wavBase64"]))
```

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
5. If the engine package is missing, click `Download Engine DLLs`.
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

<details open>
<summary>Linux install</summary>

Linux release assets are available on the GitHub Releases page. Use the `.deb`
for Ubuntu/Debian installs, or the AppImage as the portable Linux build.

**Option A: `.deb` package (Ubuntu/Debian)**

```bash
sudo apt install ./Higgs-Audio-v3-Studio_0.2.31_amd64.deb
```

Launch from the app menu or run `higgs-audio-studio` from terminal.

**Option B: AppImage (portable)**

```bash
chmod +x Higgs-Audio-v3-Studio_0.2.31_amd64.AppImage
./Higgs-Audio-v3-Studio_0.2.31_amd64.AppImage
```

**Option C: Raw binary (most portable, needs system WebKitGTK)**

```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev libsoup-3.0-dev libfuse2
./higgs-audio-studio
```

After launching by any method:

1. Click `Download Engine Files` to fetch the Linux CUDA engine package.
2. Download or browse to a Higgs model folder.
3. Load the engine and model.

</details>

## Model Setup

<details open>
<summary>Higgs and Whisper model setup</summary>

The model selector expects model folders, not loose files. A good downloaded
folder layout is:

```text
models/
  higgs-q8_0/
    q8_0.gguf
    chat_template.jinja
    config.json
    higgs_audio_v2_tokenizer_config.json
    tokenizer.json
    tokenizer_config.json
  higgs-q6_k/
    q6_k.gguf
  higgs-q5_k/
    q5_k.gguf
  higgs-q4_k_m/
    q4_k_m.gguf
  higgs-bf16/
    bf16.gguf
```

`Higgs Audio v3 Q8_0` is the recommended model in the app. In-app Higgs model
downloads now fetch the whole selected model folder: the `.gguf` file plus
`chat_template.jinja`, `config.json`,
`higgs_audio_v2_tokenizer_config.json`, `tokenizer.json`, and
`tokenizer_config.json`. The progress bar shows the current file and per-file
download progress, for example `File 1/6: q8_0.gguf`.

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

Known-good Linux build inputs:

| Dependency | Version / Notes |
| --- | --- |
| GCC or Clang | GCC 13+ or Clang 18+ |
| CMake | 3.20+ |
| Ninja | 1.11+ |
| CUDA Toolkit | CUDA 13.x (`nvcc`, `libcudart`, `libcufft`) |
| Rust | Stable toolchain |
| Node.js | 20+ recommended |
| WebKitGTK 4.1 | `libwebkit2gtk-4.1-dev` and related GTK dev packages |
| `patchelf` | For setting rpath on the engine `.so` |

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

<details>
<summary>Build the C++/CUDA shared library on Linux</summary>

Install build dependencies:

```bash
sudo apt install -y build-essential git git-lfs curl wget pkg-config \
  cmake ninja-build clang lld patchelf file unzip zip jq
sudo apt install -y libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev \
  libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev libsoup-3.0-dev libfuse2
```

Add CUDA to PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Clone and build:

```bash
git clone --recursive https://github.com/Saganaki22/Higgs-Audio-v3-Studio.git
cd Higgs-Audio-v3-Studio

cmake -S . -B build/linux-cuda-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DENGINE_ENABLE_CUDA=ON \
  -DENGINE_ENABLE_CUDA_GRAPHS=ON \
  -DENGINE_ENABLE_OPENMP=ON \
  -DENGINE_BUILD_DESKTOP_API=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;89-real;120-real"

cmake --build build/linux-cuda-release --target audiocpp_engine -j"$(nproc)"
```

The shared library is written to:

```text
build/linux-cuda-release/app/desktop_api/libaudiocpp_engine.so
```

Stage a self-contained engine package (includes CUDA runtime libs with `rpath=$ORIGIN`):

```bash
mkdir -p ~/hf-higgs-studio/engines_linux
ENGINE_SO="$(find build/linux-cuda-release -name 'libaudiocpp_engine.so' | head -n1)"
cp "$ENGINE_SO" ~/hf-higgs-studio/engines_linux/libaudiocpp_engine.so
cp -L /usr/local/cuda/lib64/libcudart.so.13   ~/hf-higgs-studio/engines_linux/
cp -L /usr/local/cuda/lib64/libcublas.so.13   ~/hf-higgs-studio/engines_linux/
cp -L /usr/local/cuda/lib64/libcublasLt.so.13 ~/hf-higgs-studio/engines_linux/
for f in ~/hf-higgs-studio/engines_linux/lib*.so*; do patchelf --set-rpath '$ORIGIN' "$f"; done
```

For development, copy the engine to:

```text
desktop/src-tauri/resources/engine/libaudiocpp_engine.so
```

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

<details>
<summary>Build the Tauri app on Linux</summary>

```bash
cd desktop
npm ci
npm run build
```

Expected outputs:

```text
desktop/src-tauri/target/release/higgs-audio-studio
desktop/src-tauri/target/release/bundle/deb/Higgs Audio v3 Studio_0.2.31_amd64.deb
desktop/src-tauri/target/release/bundle/appimage/Higgs Audio v3 Studio_0.2.31_amd64.AppImage
```

Build only the binary without installer bundles:

```bash
cd desktop
npx @tauri-apps/cli build --no-bundle
```

For local testing, stage the engine files:

```bash
mkdir -p ~/.higgs-audio-v3-studio/engine
cp ~/hf-higgs-studio/engines_linux/*.so* ~/.higgs-audio-v3-studio/engine/
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
- `Download Engine DLLs` downloads `audiocpp_engine.dll` plus the required CUDA/MSVC runtime DLLs.
- Higgs model downloads show `File n/6` and download the selected model folder assets plus GGUF weights.
- `Load Engine` changes the engine chip from unloaded to loaded.
- `Load Model` enables generation after a valid model folder is selected.
- Voice clone and multi-speaker workflows require reference audio. New uploaded clone/Speaker Gallery references are auto-cropped to 30 seconds.
- Multi-speaker generation checks all speech lines before starting. If a line points to a missing speaker or a speaker without a reference voice, the app stops immediately and tells you which line to fix.
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

- Click `Download Engine DLLs`.
- Copy the DLL beside the release executable.
- In dev, copy it to `desktop/src-tauri/resources/engine/audiocpp_engine.dll`.

### `Load Model` is disabled

Load the engine first, then select a valid model folder.

The app lists folders containing model weights such as `model.gguf` or
`model.safetensors`. Loose files in the root `models/` folder will not appear as
full model entries.

### CUDA DLL load errors

Version `0.2.31` added an engine dependency preflight. If Windows cannot load
`audiocpp_engine.dll`, the app checks common loader dependencies first and shows
a repair dialog listing missing DLLs such as `nvcuda.dll`,
`cublas64_13.dll`, `cublasLt64_13.dll`, `vcruntime140.dll`,
`vcruntime140_1.dll`, `msvcp140.dll`, or `VCOMP140.DLL`.

Make sure:

- NVIDIA driver is current.
- Click `Download Engine DLLs`, or make sure CUDA Toolkit 13.x runtime DLLs required by the engine are on `PATH`.
- Click `Download Engine DLLs`, or install Microsoft Visual C++ Redistributable 2015-2022 x64.
- The engine DLL was built for your GPU architecture.

### VRAM spikes during generation

Short VRAM spikes can happen during inference. They usually come from temporary
workspace buffers, KV cache allocation, CUDA/ggml scratch allocations, audio
codec stages, or graph execution setup. A quantized model can still need extra
transient memory while generating.

Seeing the card jump close to full VRAM for a moment does not always mean the
model weights themselves need that much memory. CUDA/ggml can reserve large
workspace and scratch regions sized for the current graph/session, and Windows
or NVML may report that reserved memory as used. The Higgs generator decode graph
allocates KV cache for the requested maximum token cap, so very high
`max_tokens` values can reserve much more VRAM up front even when the text is
short.

Version `0.2.31` and later releases Higgs runtime graphs and codec graphs after each
request, releases them when streaming is cancelled or errors out, checks cancel
inside the native decode loop, and sends per-stage native VRAM diagnostics to
the Command Centre. Use those `vram stage=...` log lines to see whether a spike
comes from reference encoding, generator decode graph allocation, streaming codec
decode, final codec decode, or cleanup. The packaged CUDA engine now uses F16
decode KV cache by default to reduce cache VRAM pressure, with an F32 diagnostic
fallback available for troubleshooting.

If you run out of memory:

- Use a smaller or lower-quantized model.
- Close other GPU-heavy apps.
- Reduce max tokens; the UI default is `1024`.
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
