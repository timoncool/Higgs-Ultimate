#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PY = REPO_ROOT / "reference" / "supertonic" / "py"
if str(REFERENCE_PY) not in sys.path:
    sys.path.insert(0, str(REFERENCE_PY))

from helper import TextToSpeech, load_cfgs, load_onnx_all, load_text_processor, load_voice_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Supertonic long-lived warmbench.")
    parser.add_argument("--family", default="supertonic")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "supertonic-3")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", type=Path, default=Path("/tmp/supertonic_python_warm_bench_timing.log"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--warmup-request-json", default="")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def require_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Supertonic request JSON entries must be objects")
    text = payload.get("text", "")
    if not isinstance(text, str) or not text:
        raise RuntimeError("Supertonic request requires non-empty text")
    return dict(payload)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.request_sequence_json:
        raise RuntimeError("Supertonic warmbench requires --request-sequence-json")
    payload = json.loads(args.request_sequence_json)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("--request-sequence-json must decode to a non-empty list")
    return [require_request(item) for item in payload]


def load_warmup_request(args: argparse.Namespace, requests: list[dict[str, Any]]) -> dict[str, Any]:
    if args.warmup_request_json:
        return require_request(json.loads(args.warmup_request_json))
    return requests[0]


def providers_for_backend(backend: str, device: int) -> list[Any]:
    if backend == "cuda":
        return [("CUDAExecutionProvider", {"device_id": device}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_tts(model_root: Path, backend: str, device: int, threads: int) -> TextToSpeech:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    onnx_dir = model_root / "onnx"
    cfgs = load_cfgs(str(onnx_dir))
    sessions = load_onnx_all(str(onnx_dir), opts, providers_for_backend(backend, device))
    return TextToSpeech(cfgs, load_text_processor(str(onnx_dir)), *sessions)


def audio_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("Supertonic Python warmbench received empty audio")
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "samples": int(flat.size),
        "frames": int(flat.size),
        "duration_sec": float(flat.size / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(flat.astype(np.float64) ** 2))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def style_for_request(model_root: Path, request: dict[str, Any], cache: dict[str, Any]) -> Any:
    voice = str(request.get("voice", "M1"))
    found = cache.get(voice)
    if found is not None:
        return found
    style_path = model_root / "voice_styles" / (voice + ".json")
    style = load_voice_style([str(style_path)])
    cache[voice] = style
    return style


def run_request(
    tts: TextToSpeech,
    model_root: Path,
    request: dict[str, Any],
    style_cache: dict[str, Any],
) -> tuple[np.ndarray, int]:
    seed = int(request.get("seed", 1234))
    np.random.seed(seed)
    style = style_for_request(model_root, request, style_cache)
    wav, duration = tts(
        str(request["text"]),
        str(request.get("language", "en")),
        style,
        int(request.get("total_steps", 10)),
        float(request.get("speed", 1.05)),
    )
    trim = int(tts.sample_rate * duration[0].item())
    return wav[0, :trim].astype(np.float32, copy=False), tts.sample_rate


def main() -> None:
    args = parse_args()
    if args.iterations != 1:
        raise RuntimeError("Supertonic warmbench records raw per-request timing; --iterations must be 1")
    model_root = resolve_path(args.model)
    requests = load_requests(args)
    warmup_request = load_warmup_request(args, requests)
    tts = load_tts(model_root, args.backend, args.device, args.threads)
    style_cache: dict[str, Any] = {}
    for _ in range(args.warmup):
        run_request(tts, model_root, warmup_request, style_cache)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        started = time.perf_counter()
        audio, sample_rate = run_request(tts, model_root, request, style_cache)
        wall_ms = (time.perf_counter() - started) * 1000.0
        audio_path = ""
        if output_dir is not None:
            path = output_dir / f"audio_{request_index}.wav"
            sf.write(path, audio, sample_rate)
            audio_path = str(path)
        text_length = len(str(request["text"]))
        print(f"supertonic.request[{request_index}].length={text_length}")
        print(f"supertonic.request[{request_index}].wall_ms={wall_ms:.6f}")
        stem = {"name": "audio", "summary": audio_summary(audio, sample_rate)}
        if audio_path:
            stem["audio"] = audio_path
        steps.append({
            "request_index": request_index,
            "text_length": text_length,
            "stems": [stem],
            "metrics": {"wall_ms": wall_ms},
        })
    print("summary_json=" + json.dumps({
        "family": "supertonic",
        "backend": args.backend,
        "sequence_steps": steps,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
