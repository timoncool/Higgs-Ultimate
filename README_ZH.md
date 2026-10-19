# Higgs Audio v3 Studio

[English README](README.md) | [Linux branch](https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux)

作者: [Saganaki22](https://github.com/Saganaki22)

Higgs Audio v3 Studio `0.3.0` 是一个使用 Rust/Tauri 构建的 Windows 桌面
应用，用来通过移植后的原生 C++/CUDA 引擎在本机运行 Higgs Audio v3 TTS
推理。应用不会通过 Python 环境或 CLI sidecar 绕一圈，而是由 Tauri 前端
调用 Rust 命令，Rust 再通过 `libloading` 加载 `audiocpp_engine.dll`，最后
进入原生推理路径。

这个 `main` 分支对应 Windows 桌面版本。Linux `.deb` 和 AppImage 包也已放在
同一个 GitHub Releases 页面，源码与 Linux 打包说明在
[`linux`](https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux)
分支。

这个项目的目标是提供一个实用的本地语音生成工作流：普通 TTS、声音克隆、
继续说话、多说话人生成、带去爆音处理和波形拖动定位的实时流式播放、本地 API 流式响应、Whisper
自动转写、WAV/MP3 导出、说话人缓存复用，以及 GPU/VRAM 监控。Telegram/WhatsApp
的 OGG/Opus 语音和 WebM/Opus 参考音频由应用内置解码，无需安装 FFmpeg。

## 下载

预构建文件发布位置：

- GitHub Releases: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/releases
- Linux 分支源码: https://github.com/Saganaki22/Higgs-Audio-v3-Studio/tree/linux
- Hugging Face 运行时仓库: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio
- Manifest: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/manifest.json
- Checksums: https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/checksums/SHA256SUMS.txt

直接下载：

| 文件 | 推荐显存 | 链接 |
| --- | --- | --- |
| Engine DLL package | 需要 NVIDIA CUDA 13 GPU/driver | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/tree/main/engines |
| Linux engine package | 需要 NVIDIA CUDA 13 GPU/driver | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/tree/main/engines_linux |
| Higgs Q8_0 推荐 | 12 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q8_0/q8_0.gguf |
| Higgs Q6_K | 10 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q6_k/q6_k.gguf |
| Higgs Q5_K | 9 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q5_k/q5_k.gguf |
| Higgs Q4_K_M | 8 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-q4_k_m/q4_k_m.gguf |
| Higgs BF16 | 16 GB VRAM | https://huggingface.co/drbaph/Higgs-Audio-v3-Studio/resolve/main/models/higgs-bf16/bf16.gguf |

## 使用流程

1. 从 GitHub Releases 下载最新版 Windows 包。Linux 用户可在同一个 Releases 页面下载 `.deb` 或 AppImage。
2. 启动 `Higgs Audio v3 Studio`。
3. 如果没有引擎包，Windows 点击 `Download Engine DLLs`，Linux 点击 `Download Engine Files`。
4. 下载或浏览到 Higgs 模型文件夹。应用内 Higgs 下载会下载整个所选模型文件夹：GGUF 权重以及 config/tokenizer/chat-template 资产。
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

把本地 Hugging Face 待上传目录按同样结构上传到 Hugging Face 仓库根目录后，
应用里的 `resolve/main/...` 下载链接就会正确工作。

## 运行时文件

三个 Windows 安装包运行的是同一个应用，但存储方式分为两种：

| 包 | 程序文件 | 引擎下载 | 模型下载 | 设置、说话人与临时音频 |
| --- | --- | --- | --- | --- |
| 便携版目录 | 与 EXE 放在一起 | EXE 旁的 `resources/engine/` | EXE 旁的 `models/` | EXE 旁的 `data/` |
| NSIS 安装 EXE | 由 NSIS 向导安装 | `%LOCALAPPDATA%\Higgs Audio v3 Studio\engine\` | `%USERPROFILE%\audiocpp\models\` | Windows 当前用户的 AppData/临时目录 |
| MSI 安装包 | 由 Windows Installer 安装 | `%LOCALAPPDATA%\Higgs Audio v3 Studio\engine\` | `%USERPROFILE%\audiocpp\models\` | Windows 当前用户的 AppData/临时目录 |

便携版对应用自己管理的全部可写数据采用自包含存储。请保持整个文件夹完整，
并把它放在普通可写目录中运行。`portable.flag` 会把引擎/模型下载、设置、
说话人、参考缓存、录音、临时音频和 WebView 数据都保存在便携版目录内：

```text
Higgs Audio v3 Studio 0.3.0 Portable/
  Higgs Audio v3 Studio.exe
  portable.flag
  resources/
    engine/                       # 点击 Download Engine DLLs 后写入
    higgs-assets/                 # Higgs config/tokenizer 资源
    api-console/                  # 浏览器 API 测试控制台
  models/                         # Higgs 与 Whisper 下载
  data/
    speakers/                     # 说话人身份与缓存
    temp/                         # 录音和准备后的临时音频
    webview/                      # 设置、API key 与 WebView2 状态
```

便携版包含 Higgs 支持资源与 API 测试控制台，但不打包原生引擎/runtime DLL，
也不包含数 GB 的 GGUF 权重。点击 `Download Engine DLLs` 后，引擎包会从
Hugging Face 下载到 `resources/engine/`。模型或 Whisper 文件会保存到便携版
的 `models/` 目录。移动整个便携版目录到另一个可写磁盘时，应用数据也会
一起移动。

NSIS 与 MSI 都属于安装版。程序和打包的只读资源位于 Windows 安装目录，
可变下载与用户数据则写入当前用户可写目录，从而避免在 `Program Files`
下出现 `Access is denied`。NSIS 是普通图形安装向导；MSI 使用 Windows
Installer，更适合集中部署、修复和 MSI 卸载工具。两者运行功能相同。

安装版没有 `portable.flag`。引擎下载目录为：

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

安装版默认模型下载目录为：

```text
C:\Users\<you>\audiocpp\models\
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
  whisper/
  ggml-base.en-q8_0.bin
```

引擎包包含
`audiocpp_engine.dll`、cuBLAS/cuBLASLt CUDA 13 runtime DLL，以及当前引擎
需要的 MSVC/OpenMP runtime DLL。应用也会自动检查打包资源与系统中兼容的
CUDA/MSVC runtime 目录。

## Speaker Gallery

Speaker Gallery 用来管理可复用的说话人身份。Voice Clone、Continue Speech
和 Multi Speaker 都可以选择已保存的说话人，但用户仍然可以直接上传一次性
参考音频，不会被强制使用图库。Voice Clone 和 Speaker Gallery 新上传的参考
音频会自动准备为 WAV，并限制为前 30 秒，避免误上传长音频浪费推理准备时间；
Continue Speech 会保留完整源音频，因为这个工作流可能需要继续较长内容。

同一组参考音频位置也可以直接使用麦克风录音。Voice Clone、Continue Speech、
Speaker Gallery，以及 Multi Speaker 的每个说话人卡片都可以选择输入设备、开始
或停止录音、查看实时波形，并在录音成功完成后替换当前参考音频。录音最长 30 秒，
保存为单声道 WAV。Whisper 自动转写仍然是可选功能；录音器本身不执行说话人分离。

参考音频还可以选择开始点和结束点、试听选区并应用裁剪。Voice Clone、
Continue Speech、Speaker Gallery、Multi Speaker 说话人以及每行独立参考音频
都支持该功能。应用裁剪后会写入新的 WAV，替换 Higgs 和 Whisper 实际使用的
参考文件，并清除旧的说话人缓存；克隆参考选区最长为 30 秒。

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
缓存不会序列化，因为它们与具体模型和量化版本强相关。`.hspkcache` 保存的是
离散 codec ID，而不是模型 KV；使用默认 `higgs_tts` 量化策略、来自同一源模型
的官方 Q4_K_M/Q5_K/Q6_K/Q8_0/BF16 可以共用，因为该策略会保留 codec encoder
张量。不同模型或 codec 版本、微调模型、以及自定义 `--policy all` 量化应重新
生成缓存。

生成队列管理器会显示当前正在生成的任务和等待中的 UI 任务。等待中的任务可
以编辑、删除，或者一次性清空；API 请求仍由后端队列锁串行执行，避免多个推
理任务同时抢同一个引擎。

Multi Speaker 生成会在开始前检查所有台词。如果某一行指向不存在的说话人，
或者说话人还没有参考音频，应用会立刻提示需要修复的行，而不是等生成到那一
行才失败。生成过程中进度文本会显示 `Line 1/42`、`Line 2/42` 这类行进度；
原生引擎提供 token 总数时，耗时旁边也会显示 `Tokens 当前值 / 最大值`。

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
| `POST /v1/audio/speech` | OpenAI 风格的普通 TTS 或保存说话人克隆，返回 WAV 或 MP3。 |
| `POST /v1/higgs/voice-clone` | 使用本地参考音频路径进行声音克隆。 |
| `POST /v1/higgs/continue-speech` | 继续一段本地音频。 |
| `POST /v1/higgs/audio/stream` | 以 NDJSON 事件流返回进度和 wav-base64 音频块。 |
| `POST /v1/higgs/cancel` | 取消当前生成。 |

`/v1/higgs/audio/stream` 会发送 `queued`、`start`、`progress`、`audio`、
`final`、`done` 或 `error` 事件。API 页的 Command Centre 可以弹出为独立
窗口，并按 info、warning、error、request、job 过滤日志；主窗口最小化到系统
托盘时，弹出的日志窗口不会跟着隐藏。

普通完成文件接口支持 `response_format: "wav"` 或 `"mp3"`，会直接返回
`audio/wav` 或 `audio/mpeg`。流式接口的实时 `audio` 事件仍返回
`wavBase64` 分块，最终 `final` 事件会遵守 `response_format`：WAV 返回
`wavBase64`，MP3 返回 `mp3Base64`。

脚本播放流式音频时，逐行读取 HTTP 响应。`event` 为 `audio` 时，把
`wavBase64` 解码成 WAV 字节并送入自己的播放器或音频队列；`event` 为
`final` 时检查 `encoding`，如果是 `wav-base64` 就保存 `wavBase64`，如果是
`mp3-base64` 就保存 `mp3Base64`。

API 页的 `Test Console` 按钮会在系统默认浏览器中打开随应用打包的独立测试
控制台，并自动填入当前 API 根地址。应用会把当前 API key 复制到剪贴板，方便
粘贴到控制台。该控制台可以测试健康状态、普通 TTS、参考音频或保存说话人的
声音克隆、Continue Speech、WAV/MP3 完成文件以及 NDJSON 流式响应。

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
            # 在这里把 wav_chunk 放进你的播放队列。
        elif event["event"] == "final":
            if event.get("encoding") == "mp3-base64":
                open("final.mp3", "wb").write(base64.b64decode(event["mp3Base64"]))
            else:
                open("final.wav", "wb").write(base64.b64decode(event["wavBase64"]))
```

## 系统要求

- Windows 11 x64，Windows 10 x64 也可能可用。
- Tauri 2 / WebView2 桌面运行时。
- NVIDIA RTX GPU，预构建 CUDA 引擎需要 CUDA 13 兼容驱动。
- 推荐 RTX 30 系、40 系或 50 系 GPU。
- Q4_K_M 推荐 8 GB VRAM，Q5_K 推荐 9 GB VRAM，Q6_K 推荐 10 GB VRAM，Q8_0 推荐 12 GB VRAM，BF16 推荐 16 GB VRAM。
- 生成时的瞬时 VRAM 峰值可能来自 ggml/CUDA 工作区、KV cache 和 codec 图。`0.2.31` 及之后版本会在每次请求结束、取消或出错后释放运行时图缓存，在原生 decode 循环内部检查取消，并把每个阶段的 VRAM 诊断写入 Command Centre。可根据 `vram stage=...` 判断峰值来自 reference encode、generator decode graph、streaming codec decode、final codec decode 还是 cleanup。打包版 CUDA 引擎现在默认使用 F16 decode KV cache 来降低缓存显存压力，并保留 F32 诊断回退路径用于排错；如果显存仍然吃紧，请降低 `max_tokens`（默认 `1024`）并使用更小的长文本分块。

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
.\scripts\package_windows_release.ps1
```

该脚本会构建 MSI、NSIS 和自包含便携目录；只有便携目录会得到
`portable.flag`。如果系统能找到 `upx.exe`，脚本只压缩便携版 EXE。所有应用
安装包都会排除引擎 DLL；用户需要时通过应用从 Hugging Face `engines/` 下载。

## 常见问题

### `Port 1420 is already in use`

已有 Vite dev server 占用了端口。关闭旧进程后重新运行。

### `Engine library not found`

点击 `Download Engine DLLs`，或者把 `audiocpp_engine.dll` 放到
`desktop/src-tauri/resources/engine/`。如果没有安装 CUDA Toolkit / VC++
运行时，也可以通过 `Download Engine DLLs` 下载当前引擎需要的
cuBLAS/cuBLASLt 与 MSVC/OpenMP DLL。

### CUDA DLL 加载错误

`0.2.31` 增加了引擎依赖预检查。 如果 Windows 无法加载
`audiocpp_engine.dll`，应用会先检查常见依赖，并显示修复窗口，列出缺失的
DLL，例如 `nvcuda.dll`、`cublas64_13.dll`、`cublasLt64_13.dll`、
`vcruntime140.dll`、`vcruntime140_1.dll`、`msvcp140.dll` 或
`VCOMP140.DLL`。

请确认：

- NVIDIA 驱动已安装并保持最新。
- 点击 `Download Engine DLLs`，或确保 CUDA Toolkit 13.x 的相关运行时 DLL 在 `PATH` 中。
- 点击 `Download Engine DLLs`，或安装 Microsoft Visual C++ Redistributable 2015-2022 x64。
- 引擎 DLL 是为当前 GPU 架构构建的。

### 模型加载失败

模型选择器需要选择文件夹，而不是直接选择单个 `.gguf` 文件。示例：

```text
models/higgs-q8_0/q8_0.gguf
```

选择 `models/higgs-q8_0/` 这个文件夹。

### Whisper 自动转写无反应

在左侧 Whisper 面板选择并下载一个 `ggml-*.bin` 模型。推荐默认模型是
`base.en-q8_0`，`base.en` 也会在下拉框中标星。

### OGG/Opus 或 WebM/Opus 参考音频无法加载

`0.3.0` 起应用通过静态链接的 `libopus` 在进程内解码 Telegram/WhatsApp
OGG/Opus 和 WebM/Opus。损坏、空白、Opus multi-stream 或超过双声道的文件会
显示明确的解码错误，不再被当成成功的空波形。

### 生成时显存突然接近满载

短时间显存尖峰通常来自 CUDA/ggml 的临时 workspace、scratch buffer、KV cache、
音频 codec 阶段或 graph/session 初始化。即使使用低 token 数，也可能先分配按
当前会话或最坏路径估算的大块工作区，所以资源监控里会看到接近满显存的瞬时占用。
这不一定表示模型权重本身需要那么多显存；高 token 数更多影响生成持续时间和 KV
cache 增长。

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
