# Higgs Audio v3 Studio

[English README](README.md)

作者: [Saganaki22](https://github.com/Saganaki22)

Higgs Audio v3 Studio `0.1.0` 是一个使用 Rust/Tauri 构建的 Windows 桌面
应用，用来通过原生 C++/CUDA 引擎在本机运行 Higgs Audio v3 TTS 推理。应用
不会通过 Python 环境或 CLI sidecar 绕一圈，而是由 Tauri 前端调用 Rust
命令，Rust 再通过 `libloading` 加载 `audiocpp_engine.dll`，最后进入原生
推理路径。

这个项目的目标是提供一个实用的本地语音生成工作流：普通 TTS、声音克隆、
继续说话、多说话人生成、Whisper 自动转写、WAV/MP3 导出，以及 GPU/VRAM
监控。

## 下载

预构建文件发布位置：

- GitHub Releases: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/releases
- Hugging Face 运行时仓库: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio
- Manifest: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/manifest.json
- Checksums: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/checksums/SHA256SUMS.txt

直接下载：

| 文件 | 推荐显存 | 链接 |
| --- | --- | --- |
| Engine DLL | 需要 NVIDIA CUDA 13 GPU | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/engines/audiocpp_engine.dll |
| Higgs Q8_0 推荐 | 12 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q8_0/q8_0.gguf |
| Higgs Q4_K_M | 8 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q4_k_m/q4_k_m.gguf |
| Higgs BF16 | 20 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-bf16/bf16.gguf |

## 使用流程

1. 从 GitHub Releases 下载最新版 Windows 包。
2. 启动 `Higgs Audio v3 Studio`。
3. 如果没有引擎 DLL，点击 `Download DLL Engine`。
4. 下载或浏览到 Higgs 模型文件夹。
5. 点击 `Load Engine`，再点击 `Load Model`。
6. 选择工作流并生成音频。

## Hugging Face 目录结构

Hugging Face 仓库根目录应该直接包含 `manifest.json`，并与 `models/`、
`engines/`、`checksums/` 放在同一级：

```text
drbaph/Higgs-Audio-v3-Studio/
  manifest.json
  engines/
    audiocpp_engine.dll
  models/
    higgs-q8_0/
      q8_0.gguf
    higgs-q4_k_m/
      q4_k_m.gguf
    higgs-bf16/
      bf16.gguf
  checksums/
    SHA256SUMS.txt
```

本地待上传目录为：

```text
C:\Users\drbaph\Documents\audio.cpp\models\
```

把这个目录内容上传到 Hugging Face 仓库根目录后，应用里的
`resolve/main/...` 下载链接就会正确工作。

## 运行时文件

便携版建议保持如下结构：

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
  higgs-q4_k_m/
    q4_k_m.gguf
  higgs-bf16/
    bf16.gguf
models/whisper/
  ggml-base.en-q8_0.bin
```

应用内下载的引擎 DLL 会保存到用户可写的 app data 目录，避免安装到
`Program Files` 后出现 Windows 权限错误。模型下载默认保存到：

```text
C:\Users\<you>\audiocpp\models\
```

## 系统要求

- Windows 11 x64，Windows 10 x64 也可能可用。
- Tauri 2 / WebView2 桌面运行时。
- NVIDIA RTX GPU，预构建 CUDA 引擎需要 CUDA 13 兼容驱动。
- 推荐 RTX 30 系、40 系或 50 系 GPU。
- Q4_K_M 推荐 8 GB VRAM，Q8_0 推荐 12 GB VRAM，BF16 推荐 20 GB VRAM。

## 从源码构建

桌面应用：

```powershell
cd desktop
npm ci
npm run build:vite
cd src-tauri
cargo check --locked
```

Windows CUDA DLL：

```powershell
.\scripts\build_windows.ps1 `
  -Preset windows-cuda-release `
  -Target audiocpp_engine `
  -Jobs 16
```

Tauri 打包：

```powershell
cd desktop
npm run build -- --bundles nsis
```

## 常见问题

### `Port 1420 is already in use`

已有 Vite dev server 占用了端口。关闭旧进程后重新运行。

### `Engine library not found`

点击 `Download DLL Engine`，或者把 `audiocpp_engine.dll` 放到
`desktop/src-tauri/resources/engine/`。

### 模型加载失败

模型选择器需要选择文件夹，而不是直接选择单个 `.gguf` 文件。示例：

```text
models/higgs-q8_0/q8_0.gguf
```

选择 `models/higgs-q8_0/` 这个文件夹。

### Whisper 自动转写无反应

在左侧 Whisper 面板选择并下载一个 `ggml-*.bin` 模型。推荐默认模型是
`base.en-q8_0`，`base.en` 也会在下拉框中标星。

## 负责任使用

不要使用 Higgs Audio v3 Studio、Higgs TTS 3 或任何声音克隆工作流，在未经
同意的情况下冒充他人、生成恶意或欺骗性声音、诈骗、规避身份识别、骚扰他人
或造成任何伤害。只有在你拥有声音来源、文本和输出用途所需权利与同意的情况
下才应使用生成语音。

## 引用

```bibtex
@misc{bosonai_higgs_audio_tts_v3_2026,
  title  = {Higgs TTS 3: Conversational Speech for Voice AI from Boson AI},
  author = {Boson AI},
  year   = {2026},
  howpublished = {https://huggingface.co/bosonai/higgs-tts-3-4b},
}
```

## 许可证

本仓库应用源码使用 Apache 2.0，见 `LICENSE`。

Higgs TTS 3 模型权重和上游模型资源受 Boson Higgs TTS 3 Research and
Non-Commercial License 约束，请查看上游模型许可证。
