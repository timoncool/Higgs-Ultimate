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
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "stable-audio-3"
DEFAULT_MODEL_ROOT = REPO_ROOT / "models"


TEST_CASES: dict[str, list[dict[str, Any]]] = {
    "music_text": [
        {
            "prompt": "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            "duration": 120.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1234,
        }
    ],
    "music_cfg_guided": [
        {
            "prompt": "A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie, 120 BPM",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 8,
            "cfg_scale": 2.0,
            "seed": 1244,
        }
    ],
    "music_init_audio": [
        {
            "prompt": "A warm bossa nova groove with brushed percussion and nylon guitar, clean studio recording",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1245,
        },
        {
            "prompt": "Transform the reference into a mellow downtempo track with deeper bass, brushed hats, and wide room ambience",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1246,
            "init_audio_from_previous": 0,
            "init_noise_level": 0.45,
        },
    ],
    "music_inpaint_audio": [
        {
            "prompt": "A sparse acoustic pop verse with fingerpicked guitar, soft shaker, and a dry intimate vocal-bed feel",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1247,
        },
        {
            "prompt": "Replace only the masked sections with a tight snare fill and bright guitar swell while preserving the surrounding acoustic verse",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1248,
            "inpaint_audio_from_previous": 0,
            "inpaint_mask_start_seconds": [2.5, 6.5],
            "inpaint_mask_end_seconds": [3.5, 8.0],
        },
    ],
    "sfx_text": [
        {
            "prompt": "Footsteps on gravel, steady walking pace, close perspective, crisp natural stone texture. Length: 8 seconds",
            "duration": 8.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1235,
        }
    ],
    "medium_text": [
        {
            "prompt": "An anthemic Pop Rock instrumental that fills your head with nostalgic thoughtfulness",
            "negative_prompt": "poor quality",
            "duration": 30.0,
            "steps": 8,
            "cfg_scale": 1.0,
            "seed": 1236,
            "chunked_decode": "on",
        }
    ],
    "all_paths": [
        {
            "prompt": "A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie, 120 BPM",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 50,
            "cfg_scale": 7.0,
            "seed": 1234,
        },
        {
            "prompt": "bossa nova bassline with brushed percussion and warm nylon guitar",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 50,
            "cfg_scale": 7.0,
            "seed": 1235,
            "init_audio_from_previous": 0,
            "init_noise_level": 0.5,
        },
        {
            "prompt": "punchy kick drum fill with a clean transition back into the groove",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 10.0,
            "steps": 50,
            "cfg_scale": 7.0,
            "seed": 1236,
            "inpaint_audio_from_previous": 1,
            "inpaint_mask_start_seconds": [2.0, 6.0],
            "inpaint_mask_end_seconds": [3.5, 8.0],
        },
        {
            "prompt": "A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie",
            "negative_prompt": "poor quality, distorted, clipped, noisy",
            "duration": 14.0,
            "steps": 50,
            "cfg_scale": 7.0,
            "seed": 1237,
            "inpaint_audio_from_previous": 2,
            "inpaint_mask_start_seconds": 10.0,
            "inpaint_mask_end_seconds": 14.0,
        },
        {
            "prompt": [
                "A tight indie rock drum groove with bright guitars, 128 BPM",
                "A calm ambient piano texture with soft tape noise and wide reverb",
            ],
            "negative_prompt": [
                "poor quality, distorted, clipped, noisy",
                "poor quality, distorted, clipped, noisy",
            ],
            "duration": [8.0, 6.0],
            "steps": 50,
            "cfg_scale": 7.0,
            "seed": 1238,
            "batch_size": 2,
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Stable Audio warmbench.")
    parser.add_argument("--family", default="stable_audio")
    parser.add_argument("--model", default="models/stable-audio-3-small-music")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--case", choices=tuple(TEST_CASES), default="music_text")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--negative-prompt", default="poor quality, distorted, clipped, noisy")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-precision", choices=("native", "fp32", "fp16"), default="native")
    parser.add_argument("--chunked-decode", choices=("default", "on", "off"), default="default")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("stable_audio_python_audio.wav"))
    parser.add_argument("--timing-file", type=Path, default=Path("stable_audio_python_timing.log"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def add_reference_path(reference_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    package = root / "stable_audio_3" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing Stable Audio reference package: {package}")
    sys.path.insert(0, str(root))
    return root


def model_name_from_arg(model: str) -> str:
    name = Path(model).name
    prefix = "stable-audio-3-"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def normalize_device(args: argparse.Namespace) -> str:
    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("Stable Audio warmbench requested CUDA, but torch.cuda.is_available() is false")
    torch.cuda.set_device(args.device)
    return f"cuda:{args.device}"


def sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


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
    if args.prompts:
        return [
            {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "duration": args.duration,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed,
            }
            for prompt in args.prompts
        ]
    return [dict(request) for request in TEST_CASES[args.case]]


def chunked_decode_value(value: Any) -> bool | None:
    if value is None or value == "default":
        return None
    if value in ("on", True):
        return True
    if value in ("off", False):
        return False
    raise RuntimeError(f"invalid chunked_decode value: {value}")


def summarize_audio(path: Path) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise RuntimeError("Stable Audio warmbench received empty audio")
    audio = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    channels = int(waveform.shape[0])
    frames = int(waveform.shape[1])
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(waveform.numel()),
        "frames": frames,
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(audio, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(audio), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
    }


def load_audio(path: Path) -> tuple[int, torch.Tensor]:
    waveform, sample_rate = torchaudio.load(str(path))
    return sample_rate, waveform


def save_batch_audio(audio: torch.Tensor, output_dir: Path, sample_rate: int) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stems: list[dict[str, Any]] = []
    for index in range(audio.shape[0]):
        path = output_dir / ("audio.wav" if audio.shape[0] == 1 else f"audio_{index:02d}.wav")
        torchaudio.save(str(path), audio[index].detach().cpu(), sample_rate, encoding="PCM_F")
        stems.append({"name": f"audio_{index}", "audio": str(path), "summary": summarize_audio(path)})
    return stems


def run_request(
    model: Any,
    request: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    previous_audio: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    seed = int(request.get("seed", args.seed))
    seed_all(seed)

    init_audio = None
    if "init_audio" in request:
        init_audio = load_audio(resolve_repo_path(str(request["init_audio"])))
    elif "init_audio_from_previous" in request:
        init_audio = load_audio(previous_audio[int(request["init_audio_from_previous"])])

    inpaint_audio = None
    if "inpaint_audio" in request:
        inpaint_audio = load_audio(resolve_repo_path(str(request["inpaint_audio"])))
    elif "inpaint_audio_from_previous" in request:
        inpaint_audio = load_audio(previous_audio[int(request["inpaint_audio_from_previous"])])

    chunked_decode = chunked_decode_value(request.get("chunked_decode", args.chunked_decode))
    batch_size = int(request.get("batch_size", 1))
    prompt = request.get("prompt")
    if prompt is None:
        raise RuntimeError("Stable Audio warmbench request missing prompt")
    if isinstance(prompt, list):
        batch_size = len(prompt)

    sync_device(device)
    started = time.perf_counter()
    audio = model.generate(
        prompt=prompt,
        negative_prompt=request.get("negative_prompt"),
        duration=request.get("duration", args.duration),
        steps=int(request.get("steps", args.steps)),
        cfg_scale=float(request.get("cfg_scale", args.cfg_scale)),
        apg_scale=float(request.get("apg_scale", 1.0)),
        seed=seed,
        batch_size=batch_size,
        sampler_type=request.get("sampler_type"),
        init_audio=init_audio,
        init_noise_level=float(request.get("init_noise_level", 1.0)),
        inpaint_audio=inpaint_audio,
        inpaint_mask_start_seconds=request.get("inpaint_mask_start_seconds"),
        inpaint_mask_end_seconds=request.get("inpaint_mask_end_seconds"),
        chunked_decode=chunked_decode,
    )
    sync_device(device)
    wall_ms = (time.perf_counter() - started) * 1000.0

    stems = save_batch_audio(audio, output_dir, int(model.model.sample_rate))
    previous_audio.append(Path(stems[0]["audio"]))
    timing = [
        f"stable_audio.wall_ms {wall_ms:.6f}",
    ]
    return (
        {
            "request": request,
            "stems": stems,
            "metrics": {"wall_ms": wall_ms},
        },
        timing,
    )


def main() -> int:
    args = parse_args()
    if args.family != "stable_audio":
        raise RuntimeError(f"unsupported Stable Audio warmbench family: {args.family}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("STABLE_AUDIO_LOCAL_ONLY", "1")
    os.environ["STABLE_AUDIO_MODEL_ROOT"] = str(resolve_repo_path(args.model_root).resolve())
    add_reference_path(args.reference_root)
    device = normalize_device(args)
    seed_all(args.seed)

    from stable_audio_3 import StableAudioModel

    model_name = model_name_from_arg(args.model)
    model_half = args.model_precision == "fp16" or (args.model_precision == "native" and device == "cuda")
    model = StableAudioModel.from_pretrained(model_name, device=device, model_half=model_half)
    sync_device(device)

    requests = load_requests(args)
    if not requests:
        raise RuntimeError("Stable Audio warmbench request sequence is empty")

    output_root = args.output_dir if args.output_dir is not None else args.audio_out.parent
    output_root = resolve_repo_path(output_root).resolve()
    timing_path = resolve_repo_path(args.timing_file)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_lines: list[str] = []

    previous_audio: list[Path] = []
    for warmup_index in range(max(0, args.warmup)):
        _, warmup_timing = run_request(
            model,
            requests[0],
            output_root / "warmup" / f"{warmup_index:02d}",
            args,
            device,
            previous_audio,
        )
        timing_lines.extend(warmup_timing)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_step: dict[str, Any] | None = None
        for iteration in range(max(1, args.iterations)):
            step, run_timing = run_request(
                model,
                request,
                output_root / f"request_{request_index:02d}" / f"iter_{iteration:02d}",
                args,
                device,
                previous_audio,
            )
            total_ms += float(step["metrics"]["wall_ms"])
            last_step = step
            timing_lines.extend(run_timing)
        if last_step is None:
            raise RuntimeError("Stable Audio warmbench produced no request step")
        last_step = dict(last_step)
        last_step["request_index"] = request_index
        last_step["metrics"] = dict(last_step["metrics"])
        last_step["metrics"]["wall_ms"] = total_ms / float(max(1, args.iterations))
        print(f"stable_audio.wall_ms={last_step['metrics']['wall_ms']}")
        steps.append(last_step)

    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    summary = {
        "family": "stable_audio",
        "backend": args.backend,
        "model": model_name,
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
