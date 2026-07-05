# Higgs Audio v3 Studio

[English README](README.md)

作者: [Saganaki22](https://github.com/Saganaki22)

Higgs Audio v3 Studio `0.2.0` 是一个使用 Rust/Tauri 构建的 Windows 桌面
应用，用来通过移植后的原生 C++/CUDA 引擎在本机运行 Higgs Audio v3 TTS
推理。应用不会通过 Python 环境或 CLI sidecar 绕一圈，而是由 Tauri 前端
调用 Rust 命令，Rust 再通过 `libloading` 加载 `audiocpp_engine.dll`，最后
进入原生推理路径。

这个项目的目标是提供一个实用的本地语音生成工作流：普通 TTS、声音克隆、
继续说话、多说话人生成、生成时实时流式播放、本地 API 流式响应、Whisper
自动转写、WAV/MP3 导出、说话人缓存复用，以及 GPU/VRAM 监控。

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
| Higgs Q6_K | 10 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q6_k/q6_k.gguf |
| Higgs Q5_K | 9 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q5_k/q5_k.gguf |
| Higgs Q4_K_M | 8 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q4_k_m/q4_k_m.gguf |
| Higgs BF16 | 16 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-bf16/bf16.gguf |

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

把本地 Hugging Face 待上传目录按同样结构上传到 Hugging Face 仓库根目录后，
应用里的 `resolve/main/...` 下载链接就会正确工作。

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
  higgs-q6_k/
    q6_k.gguf
  higgs-q5_k/
    q5_k.gguf
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

## Speaker Gallery

Speaker Gallery 用来管理可复用的说话人身份。Voice Clone、Continue Speech
和 Multi Speaker 都可以选择已保存的说话人，但用户仍然可以直接上传一次性
参考音频，不会被强制使用图库。

创建或编辑说话人时，应用会在用户 app data 的 speaker store 中保存文件：

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

`Export` 会先打开选择窗口，让你勾选要导出的说话人身份。导出的可移植
`.zip` 里包含总 `manifest.json`，以及每个已选择说话人自己的文件夹。
`Import` 会读取 ZIP，把音频和图片解包回应用的 speaker store，并恢复说话人
列表。

导出/导入包含说话人元数据、参考音频、转写文本、备注、头像，以及已经生成
过的 `.hspkcache` 参考编码缓存。这个缓存会在保存的说话人第一次参与推理后
创建，之后 Voice Clone、Continue Speech、Multi Speaker 和保存说话人的 API
请求都可以复用它，减少重复的参考音频准备工作。模型内部的 KV prefix 或激活
缓存不会序列化，因为它们与具体模型和量化版本强相关。

生成队列管理器会显示当前正在生成的任务和等待中的 UI 任务。等待中的任务可
以编辑、删除，或者一次性清空；API 请求仍由后端队列锁串行执行，避免多个推
理任务同时抢同一个引擎。

## 本地 API

API 页可以启动本地 HTTP 服务，默认地址：

```text
http://127.0.0.1:7077/v1
```

除 `GET /health` 外，`/v1` 路由都需要：

```http
Authorization: Bearer <your-api-key>
```

常用路由：

| 路由 | 用途 |
| --- | --- |
| `GET /health` | 无需鉴权的健康检查。 |
| `GET /v1/status` | 引擎、模型、队列和流式支持状态。 |
| `GET /v1/models` | 应用检测到的本地 Higgs 模型文件夹。 |
| `GET /v1/higgs/speakers` | 已保存的说话人身份，可用作 `speaker:<id>`。 |
| `POST /v1/audio/speech` | OpenAI 风格的普通 TTS 或保存说话人克隆，返回 WAV。 |
| `POST /v1/higgs/voice-clone` | 使用本地参考音频路径进行声音克隆。 |
| `POST /v1/higgs/continue-speech` | 继续一段本地音频。 |
| `POST /v1/higgs/audio/stream` | 以 NDJSON 事件流返回进度和 wav-base64 音频块。 |
| `POST /v1/higgs/cancel` | 取消当前生成。 |

`/v1/higgs/audio/stream` 会发送 `queued`、`start`、`progress`、`audio`、
`final`、`done` 或 `error` 事件。API 页的 Command Centre 可以弹出为独立
窗口，并按 info、warning、error、request、job 过滤日志；主窗口最小化到系统
托盘时，弹出的日志窗口不会跟着隐藏。

## 系统要求

- Windows 11 x64，Windows 10 x64 也可能可用。
- Tauri 2 / WebView2 桌面运行时。
- NVIDIA RTX GPU，预构建 CUDA 引擎需要 CUDA 13 兼容驱动。
- 推荐 RTX 30 系、40 系或 50 系 GPU。
- Q4_K_M 推荐 8 GB VRAM，Q5_K 推荐 9 GB VRAM，Q6_K 推荐 10 GB VRAM，Q8_0 推荐 12 GB VRAM，BF16 推荐 16 GB VRAM。

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
