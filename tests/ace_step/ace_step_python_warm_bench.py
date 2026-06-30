#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "ACE-Step-1.5"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "models" / "Ace-Step1.5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "logs" / "parity" / "ace_step_python_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference ACE-Step warmbench.")
    parser.add_argument("--family", default="ace_step")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--config-path", default="acestep-v15-turbo")
    parser.add_argument("--lm-model-path", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--lm-backend", choices=["pt", "vllm", "mlx"], default="pt")
    parser.add_argument("--lm-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps", "xpu"], default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--noise-file", default="")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    return parser.parse_args()


def summarize_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if audio.ndim == 1:
        channels = 1
        frames = int(audio.shape[0])
        flat = audio.astype(np.float32, copy=False)
    else:
        frames = int(audio.shape[0])
        channels = int(audio.shape[1])
        flat = audio.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("ACE-Step warmbench summary received empty audio")
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(flat.size),
        "frames": frames,
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat), dtype=np.float64))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


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
    raise RuntimeError("ACE-Step warmbench requires --request-json or --request-sequence-json")


def normalize_device(backend: str, device_index: int) -> str:
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device_index)
        return "cuda"
    if backend == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS backend requested but torch.backends.mps.is_available() is false")
        return "mps"
    if backend == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU backend requested but torch.xpu.is_available() is false")
        return "xpu"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def install_controlled_vae_encode_noise(noise_file: str) -> None:
    if not noise_file:
        return
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution

    noise_values = np.fromfile(noise_file, dtype=np.float32)
    if noise_values.size == 0:
        raise RuntimeError(f"controlled VAE encode noise file is empty: {noise_file}")

    def sample(self: Any, generator: Any = None) -> torch.Tensor:
        del generator
        mean = self.mean
        if mean.dim() != 3:
            raise RuntimeError(f"controlled VAE encode noise expects [B,C,T] mean, got {tuple(mean.shape)}")
        batch, channels, frames = mean.shape
        expected = int(batch * frames * channels)
        if noise_values.size < expected:
            raise RuntimeError(
                f"controlled VAE encode noise file is too short: expected at least {expected} floats, "
                f"got {noise_values.size}"
            )
        noise = noise_values[:expected].reshape(batch, frames, channels).transpose(0, 2, 1).copy()
        noise_tensor = torch.from_numpy(noise).to(device=mean.device, dtype=mean.dtype)
        return mean + self.std * noise_tensor

    OobleckDiagonalGaussianDistribution.sample = sample


def initialize_handlers(
    checkpoint_dir: Path,
    config_path: str,
    lm_model_path: str,
    backend: str,
    lm_backend: str,
    lm_dtype: str,
    init_llm: bool,
) -> tuple[AceStepHandler, LLMHandler]:
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoint_dir)
    dit_handler = AceStepHandler()
    init_message, ok = dit_handler.initialize_service(
        project_root=str(checkpoint_dir),
        config_path=config_path,
        device=backend,
        force_dtype=torch.float32,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step DiT init failed: {init_message}")
    # Keep the Python parity path in fp32 end to end so pre-DiT, DiT, and VAE
    # all run with stable float32 math. The forced initialize_service dtype
    # above avoids loading reduced-precision weights and then upcasting them
    # later, which perturbs parity boundaries before and after DiT.
    dit_handler.dtype = torch.float32
    if dit_handler.model is not None:
        dit_handler.model = dit_handler.model.to(device=backend).to(dtype=torch.float32)
    if dit_handler.text_encoder is not None:
        dit_handler.text_encoder = dit_handler.text_encoder.to(device=backend).to(dtype=torch.float32)
    if dit_handler.vae is not None:
        dit_handler.vae = dit_handler.vae.to(device=backend).to(dtype=torch.float32)
    if dit_handler.silence_latent is not None:
        dit_handler.silence_latent = dit_handler.silence_latent.to(device=backend).to(dtype=torch.float32)

    llm_handler = LLMHandler()
    if init_llm:
        dtype = None
        if lm_dtype != "auto":
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[lm_dtype]
        init_message, ok = llm_handler.initialize(
            checkpoint_dir=str(checkpoint_dir),
            lm_model_path=lm_model_path,
            backend=lm_backend,
            device=backend,
            offload_to_cpu=False,
            dtype=dtype,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step LM init failed: {init_message}")
    return dit_handler, llm_handler


def make_generation_params(request: dict[str, Any]) -> GenerationParams:
    kwargs = dict(request)
    track_name = kwargs.pop("track_name", "")
    complete_track_classes = kwargs.pop("complete_track_classes", None)
    negative_prompt = kwargs.pop("negative_prompt", None)
    if negative_prompt is not None and "lm_negative_prompt" not in kwargs:
        kwargs["lm_negative_prompt"] = negative_prompt
    if kwargs.get("task_type") == "extract" and track_name and not kwargs.get("instruction"):
        kwargs["instruction"] = f"Extract the {track_name.upper()} track from the audio:"
    if kwargs.get("task_type") == "lego" and track_name and not kwargs.get("instruction"):
        kwargs["instruction"] = f"Generate the {track_name.upper()} track based on the audio context:"
    if kwargs.get("task_type") == "complete" and complete_track_classes and not kwargs.get("instruction"):
        track_classes = [str(item).upper() for item in complete_track_classes]
        kwargs["instruction"] = f"Complete the input track with {' | '.join(track_classes)}:"
    return GenerationParams(**kwargs)


def make_generation_config(request: dict[str, Any], noise_file: str) -> GenerationConfig:
    batch_size = int(request.get("batch_size", 1))
    return GenerationConfig(
        batch_size=batch_size,
        allow_lm_batch=False,
        use_random_seed=False,
        seeds=[int(request.get("seed", 0))],
        noise_file=noise_file or None,
        lm_batch_chunk_size=1,
        constrained_decoding_debug=False,
        audio_format="wav",
    )


def request_needs_llm(request: dict[str, Any]) -> bool:
    if bool(request.get("thinking", True)):
        return True
    return any(bool(request.get(name, False)) for name in ("use_cot_metas", "use_cot_caption", "use_cot_language"))


def run_request(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    request: dict[str, Any],
    output_dir: Path,
    device: str,
    noise_file: str,
) -> tuple[dict[str, Any], list[str]]:
    params = make_generation_params(request)
    config = make_generation_config(request, noise_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_seed = int(request.get("seed", 0))
    if request_seed < 0:
        raise RuntimeError("ACE-Step warmbench requires a non-negative request seed")
    torch.manual_seed(request_seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(request_seed)
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(request_seed)
    elif device == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(request_seed)
    np.random.seed(request_seed & 0xFFFFFFFF)

    started = time.perf_counter()
    result = generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=str(output_dir),
    )
    sync_device(device)
    wall_ms = (time.perf_counter() - started) * 1000.0

    if not result.success:
        raise RuntimeError(f"ACE-Step generation failed: {result.error or result.status_message}")
    if not result.audios:
        raise RuntimeError("ACE-Step generation returned no audios")

    stems: list[dict[str, Any]] = []
    for index, audio_entry in enumerate(result.audios):
        audio_path = Path(audio_entry["path"])
        waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
        normalized_path = output_dir / f"audio_{index:02d}.wav"
        sf.write(str(normalized_path), waveform, sample_rate, subtype="PCM_16")
        stems.append(
            {
                "name": audio_entry.get("key", f"audio_{index:02d}"),
                "audio": str(normalized_path),
                "summary": summarize_audio(np.asarray(waveform, dtype=np.float32), sample_rate),
            }
        )

    step = {
        "request": request,
        "stems": stems,
        "metrics": {"wall_ms": wall_ms},
    }
    return step, [f"ace_step.wall_ms {wall_ms:.6f}"]


def main() -> int:
    args = parse_args()
    if args.family != "ace_step":
        raise RuntimeError(f"unsupported ACE-Step warmbench family: {args.family}")

    backend = normalize_device(args.backend, args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"ACE-Step checkpoint dir not found: {checkpoint_dir}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    install_controlled_vae_encode_noise(args.noise_file)

    requests = load_requests(args)
    init_llm = any(request_needs_llm(request) for request in requests)
    dit_handler, llm_handler = initialize_handlers(
        checkpoint_dir,
        args.config_path,
        args.lm_model_path,
        backend,
        args.lm_backend,
        args.lm_dtype,
        init_llm,
    )

    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT
    timing_lines: list[str] = [
        f"ace_step.backend {backend}",
        f"ace_step.lm_backend {args.lm_backend}",
        f"ace_step.lm_dtype {args.lm_dtype}",
        f"ace_step.lm_initialized {1 if init_llm else 0}",
        "ace_step.python_tf32_disabled 1",
        "ace_step.python_tqdm_disabled 1",
    ]

    if requests:
        for warmup_index in range(max(0, args.warmup)):
            _, warmup_timing = run_request(
                dit_handler,
                llm_handler,
                requests[0],
                output_root / "warmup" / f"{warmup_index:02d}",
                backend,
                args.noise_file,
            )
            timing_lines.extend(warmup_timing)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        total_ms = 0.0
        last_step: dict[str, Any] | None = None
        for iteration in range(max(1, args.iterations)):
            step, run_timing = run_request(
                dit_handler,
                llm_handler,
                request,
                output_root / f"request_{request_index:02d}" / f"iter_{iteration:02d}",
                backend,
                args.noise_file,
            )
            total_ms += float(step["metrics"]["wall_ms"])
            last_step = step
            timing_lines.extend(run_timing)
        assert last_step is not None
        last_step = dict(last_step)
        last_step["request_index"] = request_index
        last_step["metrics"] = {"wall_ms": total_ms / float(max(1, args.iterations))}
        print(f"{args.family}.wall_ms={last_step['metrics']['wall_ms']}")
        steps.append(last_step)

    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    summary = {
        "family": args.family,
        "backend": backend,
        "sequence_steps": steps,
    }
    print(f"summary_json={json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
