#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "heartlib"
DEFAULT_MODEL = REPO_ROOT / "models" / "HeartMuLa"
DEFAULT_LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows, chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy, not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day
"""
DEFAULT_TAGS = "piano,happy,pop,indie-pop,warm,female-vocal,medium-tempo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference HeartMuLa warmbench.")
    parser.add_argument("--family", default="heartmula")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--max-audio-length-ms", type=int, default=60000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--version", default="3B")
    parser.add_argument("--mula-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--codec-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument("--codec-duration", type=float, default=29.76)
    parser.add_argument("--codec-steps", type=int, default=10)
    parser.add_argument("--codec-guidance-scale", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("heartmula_python_audio.wav"))
    parser.add_argument("--timing-file", type=Path, default=Path("heartmula_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def add_reference_root(reference_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    src = root / "src"
    package = src / "heartlib" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing HeartMuLa reference package: {package}")
    sys.path.insert(0, str(src))
    return root


def load_reference_symbols(reference_root: Path) -> tuple[Any, Path]:
    root = add_reference_root(reference_root)
    from heartlib import HeartMuLaGenPipeline
    import heartlib

    module_path = Path(heartlib.__file__).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"HeartMuLa imported from {module_path}, expected under {root}") from exc
    return HeartMuLaGenPipeline, module_path


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise RuntimeError(f"unsupported dtype: {name}")


def normalize_device(backend: str, device_index: int) -> torch.device:
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("HeartMuLa warmbench requested CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list):
            raise RuntimeError("--request-sequence-json must decode to a list")
        return payload
    if args.request_json:
        payload = json.loads(args.request_json)
        if not isinstance(payload, dict):
            raise RuntimeError("--request-json must decode to an object")
        return [payload]
    return [
        {
            "lyrics": args.lyrics or DEFAULT_LYRICS,
            "tags": args.tags or DEFAULT_TAGS,
            "max_audio_length_ms": args.max_audio_length_ms,
            "topk": args.topk,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "codec_duration": args.codec_duration,
            "codec_steps": args.codec_steps,
            "codec_guidance_scale": args.codec_guidance_scale,
            "seed": args.seed,
        }
    ]


def summarize_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    audio = np.asarray(audio, dtype=np.float32)
    flat = audio.reshape(-1)
    if flat.size == 0:
        raise RuntimeError("HeartMuLa warmbench summary received empty audio")
    if audio.ndim == 1:
        frames = int(audio.shape[0])
        channels = 1
    elif audio.ndim == 2:
        frames = int(audio.shape[0])
        channels = int(audio.shape[1])
    else:
        raise RuntimeError(f"HeartMuLa warmbench expected 1D or 2D audio, got shape {audio.shape}")
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(flat.size),
        "frames": frames,
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def write_audio(path: Path, audio: torch.Tensor | np.ndarray, sample_rate: int) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    elif audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    if audio.ndim not in (1, 2):
        raise RuntimeError(f"HeartMuLa warmbench expected 1D or 2D audio, got shape {audio.shape}")
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    written, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(written, dtype=np.float32)


def run_request(
    pipeline: Any,
    request: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[str]]:
    lyrics = str(request.get("lyrics", ""))
    tags = str(request.get("tags", ""))
    if not lyrics:
        raise RuntimeError("HeartMuLa warmbench request missing lyrics")
    if not tags:
        raise RuntimeError("HeartMuLa warmbench request missing tags")

    max_audio_length_ms = int(request.get("max_audio_length_ms", 60000))
    topk = int(request.get("topk", 50))
    temperature = float(request.get("temperature", 1.0))
    cfg_scale = float(request.get("cfg_scale", 1.5))
    codec_duration = float(request.get("codec_duration", 29.76))
    codec_steps = int(request.get("codec_steps", 10))
    codec_guidance_scale = float(request.get("codec_guidance_scale", 1.25))
    seed = int(request.get("seed", 1234))

    seed_all(seed, device)
    with torch.inference_mode():
        preprocess_started = time.perf_counter()
        model_inputs = pipeline.preprocess({"lyrics": lyrics, "tags": tags}, cfg_scale=cfg_scale)
        preprocess_ms = elapsed_ms(preprocess_started)

        if pipeline.lazy_load:
            pipeline.mula
            sync_device(device)
        lazy_load = pipeline.lazy_load
        pipeline.lazy_load = False
        ar_started = time.perf_counter()
        model_outputs = pipeline._forward(
            model_inputs,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
        )
        sync_device(device)
        ar_ms = elapsed_ms(ar_started)
        pipeline.lazy_load = lazy_load
        if pipeline.lazy_load:
            pipeline._unload()
            sync_device(device)

        if pipeline.lazy_load:
            pipeline.codec
            sync_device(device)
        codec_started = time.perf_counter()
        frames = model_outputs["frames"].to(pipeline.codec_device)
        wav = pipeline.codec.detokenize(
            frames,
            duration=codec_duration,
            num_steps=codec_steps,
            disable_progress=True,
            guidance_scale=codec_guidance_scale,
        )
        sync_device(device)
        codec_ms = elapsed_ms(codec_started)
        if pipeline.lazy_load:
            pipeline._unload()
            sync_device(device)
    wall_ms = preprocess_ms + ar_ms + codec_ms

    audio_path = output_dir / "audio.wav"
    written = write_audio(audio_path, wav, 48000)
    summary = summarize_audio(written, 48000)
    step = {
        "request": request,
        "stems": [
            {
                "name": "audio",
                "audio": str(audio_path),
                "summary": summary,
            }
        ],
        "metrics": {
            "wall_ms": wall_ms,
            "audio_frames": int(frames.shape[-1]),
            "audio_codebooks": int(frames.shape[0]) if frames.ndim >= 2 else 0,
        },
    }
    timing = [
        f"heartmula.wall_ms {wall_ms:.6f}",
        f"heartmula.preprocess_ms {preprocess_ms:.6f}",
        f"heartmula.ar_ms {ar_ms:.6f}",
        f"heartmula.codec_ms {codec_ms:.6f}",
        f"heartmula.audio_frames {step['metrics']['audio_frames']}",
    ]
    return step, timing


def load_pipeline(args: argparse.Namespace, device: torch.device) -> tuple[Any, Path]:
    HeartMuLaGenPipeline, module_path = load_reference_symbols(args.reference_root)
    model_root = resolve_repo_path(args.model).resolve()
    if not (model_root / "tokenizer.json").is_file():
        raise RuntimeError(f"missing HeartMuLa tokenizer: {model_root / 'tokenizer.json'}")
    if not (model_root / "gen_config.json").is_file():
        raise RuntimeError(f"missing HeartMuLa generation config: {model_root / 'gen_config.json'}")
    if not (model_root / f"HeartMuLa-oss-{args.version}").is_dir():
        raise RuntimeError(f"missing HeartMuLa model dir: {model_root / f'HeartMuLa-oss-{args.version}'}")
    if not (model_root / "HeartCodec-oss").is_dir():
        raise RuntimeError(f"missing HeartCodec model dir: {model_root / 'HeartCodec-oss'}")
    pipeline = HeartMuLaGenPipeline.from_pretrained(
        str(model_root),
        device={"mula": device, "codec": device},
        dtype={"mula": torch_dtype(args.mula_dtype), "codec": torch_dtype(args.codec_dtype)},
        version=args.version,
        lazy_load=args.lazy_load,
    )
    return pipeline, module_path


def main() -> int:
    args = parse_args()
    if args.family != "heartmula":
        raise RuntimeError(f"unsupported HeartMuLa warmbench family: {args.family}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = normalize_device(args.backend, args.device)
    seed_all(args.seed, device)

    pipeline, module_path = load_pipeline(args, device)
    requests = load_requests(args)
    if not requests:
        raise RuntimeError("HeartMuLa warmbench request sequence is empty")

    output_root = args.output_dir if args.output_dir is not None else args.audio_out.parent
    output_root = resolve_repo_path(output_root).resolve()
    timing_path = resolve_repo_path(args.timing_file)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_lines = [
        f"heartmula.python_reference_module {module_path}",
        f"heartmula.model_root {resolve_repo_path(args.model).resolve()}",
        f"heartmula.backend {args.backend}",
        f"heartmula.lazy_load {int(args.lazy_load)}",
        "heartmula.python_tf32_disabled 1",
    ]

    for warmup_index in range(max(0, args.warmup)):
        _, warmup_timing = run_request(pipeline, requests[0], output_root / "warmup" / f"{warmup_index:02d}", device)
        timing_lines.extend(warmup_timing)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_step: dict[str, Any] | None = None
        for iteration in range(max(1, args.iterations)):
            step, run_timing = run_request(
                pipeline,
                request,
                output_root / f"request_{request_index:02d}" / f"iter_{iteration:02d}",
                device,
            )
            total_ms += float(step["metrics"]["wall_ms"])
            last_step = step
            timing_lines.extend(run_timing)
        if last_step is None:
            raise RuntimeError("HeartMuLa warmbench produced no request step")
        last_step = dict(last_step)
        last_step["request_index"] = request_index
        last_step["metrics"] = dict(last_step["metrics"])
        last_step["metrics"]["wall_ms"] = total_ms / float(max(1, args.iterations))
        print(f"heartmula.wall_ms={last_step['metrics']['wall_ms']}")
        steps.append(last_step)

    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    summary = {
        "family": "heartmula",
        "backend": args.backend,
        "sequence_steps": steps,
    }
    if args.summary_file is not None:
        summary_path = resolve_repo_path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
