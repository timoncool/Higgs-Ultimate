#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time
import types
import wave
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "VibeVoiceCommunity"
DEFAULT_MODEL = REPO_ROOT / "models" / "VibeVoice-1.5B"
DEFAULT_VOICES = (
    REFERENCE_ROOT / "demo" / "voices" / "en-Alice_woman.wav",
    REFERENCE_ROOT / "demo" / "voices" / "en-Frank_man.wav",
)
SHORT_TEXT = (
    "Speaker 1: Morning team. The harbor test is live; please check the first buoy.\n"
    "Speaker 2: Copy that. Levels look steady, and both voices should stay distinct."
)
LONG_TEXT = (
    "Speaker 1: When the overnight rain finally stopped, the research vessel "
    "was already easing away from the pier. I kept one hand on the rail and "
    "watched the city shrink behind us, because the whole morning felt like a "
    "test of patience: the weather station was late, the tide report had two "
    "conflicting numbers, and half the crew wanted to wait for clearer skies. "
    "But the acoustic buoys were drifting out of range, and if we missed this "
    "window we would lose the cleanest recording week of the season.\n"
    "Speaker 2: I remember that departure differently. From the engine room it "
    "sounded less dramatic and more like a checklist finally doing its job. The "
    "generators were stable, the satellite link came up on the second attempt, "
    "and the spare hydrophone array passed its self test after I reseated one "
    "stubborn connector. What mattered to me was not whether the sky looked "
    "friendly, but whether every system could keep running once the waves "
    "started pushing against us.\n"
    "Speaker 1: By noon the fog lifted enough to reveal a line of gulls circling "
    "over the shoals, and the first buoy came into view. We cut the engines, "
    "lowered the deck noise, and listened. There was a low mechanical hum from "
    "somewhere far north, a pattern of clicks that might have been dolphins, "
    "and beneath it all a slow pulse from the mooring chain. It was not a clean "
    "recording yet, but it was alive in the way field data always is: messy, "
    "specific, and stubbornly real.\n"
    "Speaker 2: That is why I like this test case. It has narration, technical "
    "phrasing, turn taking, and enough duration to expose whether the model can "
    "hold two voices without collapsing into a short greeting. If the output "
    "sounds natural, we know the path is exercising actual long-form synthesis "
    "instead of merely proving that a checkpoint can make a few seconds of noise."
)
TEST_CASES = {
    "short": SHORT_TEXT,
    "long": LONG_TEXT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference VibeVoice warmbench.")
    parser.add_argument("--family", default="vibevoice")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--case", choices=tuple(TEST_CASES), default="short")
    parser.add_argument("--text", action="append", dest="texts", default=[])
    parser.add_argument("--warmup-text", default="")
    parser.add_argument("--voice-sample", action="append", dest="voice_samples", default=[])
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--prompt-noise-file", default="")
    parser.add_argument("--noise-file", default="")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-length-times", type=float, default=2.0)
    parser.add_argument("--ddpm-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--disable-prefill", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audio-out", type=Path, default=Path("vibevoice_python_audio.wav"))
    parser.add_argument("--audio-out-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def add_reference_root(reference_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    package = root / "vibevoice" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing VibeVoice reference package: {package}")
    sys.path.insert(0, str(root))
    return root


def use_compatible_transformers() -> None:
    transformers = importlib.import_module("transformers457")
    sys.modules["transformers"] = transformers


def load_reference_symbols(reference_root: Path) -> tuple[Any, Any, Path]:
    use_compatible_transformers()
    expected_root = add_reference_root(reference_root)
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    import vibevoice

    module_path = Path(vibevoice.__file__).resolve()
    try:
        module_path.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(f"VibeVoice imported from {module_path}, expected under {expected_root}") from exc
    return VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor, module_path


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise RuntimeError(f"unsupported dtype: {name}")


def select_device(args: argparse.Namespace) -> str:
    torch.set_num_threads(max(1, args.threads))
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("VibeVoice warmbench requested CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(args.device)
        return f"cuda:{args.device}"
    return "cpu"


def seed_all(seed: int, backend: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


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
    texts = args.texts or [TEST_CASES[args.case]]
    voice_samples = args.voice_samples or [str(path) for path in DEFAULT_VOICES]
    return [
        {
            "text": text,
            "voice_samples": voice_samples,
            "max_new_tokens": args.max_new_tokens,
            "max_length_times": args.max_length_times,
            "ddpm_steps": args.ddpm_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
        }
        for text in texts
    ]


def summary_json(audio: np.ndarray, sample_rate: int, wall_ms: float) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    count = max(1, int(audio.size))
    payload = {
        "family": "vibevoice",
        "sample_rate": int(sample_rate),
        "channels": 1,
        "samples": int(audio.size),
        "duration_sec": float(audio.size / sample_rate) if sample_rate else 0.0,
        "sum": float(audio.sum(dtype=np.float64)) if audio.size else 0.0,
        "mean_abs": float(np.mean(np.abs(audio))) if audio.size else 0.0,
        "rms": float(np.sqrt(np.sum(audio.astype(np.float64) ** 2) / count)) if audio.size else 0.0,
        "min": float(audio.min()) if audio.size else 0.0,
        "max": float(audio.max()) if audio.size else 0.0,
        "synthesize_wall_ms": float(wall_ms),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def token_window(tokens: list[int]) -> dict[str, Any]:
    return {
        "head": tokens[:128],
        "tail": tokens[-32:],
        "count": len(tokens),
    }


def extract_audio(outputs: Any, index: int = 0) -> np.ndarray:
    speech_outputs = getattr(outputs, "speech_outputs", None)
    if not speech_outputs or index >= len(speech_outputs) or speech_outputs[index] is None:
        raise RuntimeError("VibeVoice warmbench received no speech output")
    audio = speech_outputs[index]
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def patch_transformers_cache_signature(model: Any) -> None:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "key_cache"):
        DynamicCache.key_cache = property(lambda self: [layer.keys for layer in self.layers])
    if not hasattr(DynamicCache, "value_cache"):
        DynamicCache.value_cache = property(lambda self: [layer.values for layer in self.layers])

    # Transformers 4.57 creates DynamicCache from the model config; VibeVoice keeps
    # the decoder config nested, so the reference should let Qwen create cache state
    # during the forward pass instead of preallocating it here.
    model._prepare_cache_for_generation = types.MethodType(
        lambda _self,
        generation_config,
        model_kwargs,
        assistant_model,
        batch_size,
        max_cache_length,
        device=None: None,
        model,
    )


def load_noise_values(path_text: str, label: str) -> np.ndarray | None:
    if not path_text:
        return None
    values = np.fromfile(resolve_repo_path(Path(path_text)), dtype=np.float32)
    if values.size == 0:
        raise RuntimeError(f"VibeVoice controlled {label} noise file is empty: {path_text}")
    return values


@contextlib.contextmanager
def controlled_generation_noise(
    prompt_noise_file: str,
    diffusion_noise_file: str,
    acoustic_vae_dim: int,
) -> Iterator[None]:
    prompt_noise_values = load_noise_values(prompt_noise_file, "prompt")
    diffusion_noise_values = load_noise_values(diffusion_noise_file, "diffusion")
    if prompt_noise_values is None and diffusion_noise_values is None:
        yield
        return
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    prompt_offset = 0
    diffusion_offset = 0
    prompt_used = False
    diffusion_used = False

    def consume(values: np.ndarray, offset: int, count: int, label: str) -> tuple[np.ndarray, int]:
        end = offset + count
        if end > values.size:
            raise RuntimeError(
                f"VibeVoice controlled {label} noise file is too short: need {end} floats, got {values.size}"
            )
        return values[offset:end].copy(), end

    def to_tensor(array: np.ndarray, shape: tuple[int, ...], device: Any, dtype: Any) -> torch.Tensor:
        tensor = torch.from_numpy(array.reshape(shape))
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    def randn(*shape_args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal prompt_offset, diffusion_offset, prompt_used, diffusion_used
        shape = shape_args
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        normalized_shape = tuple(int(dim) for dim in shape)
        target_device = kwargs.get("device")
        target_dtype = kwargs.get("dtype")
        if prompt_noise_values is not None and len(normalized_shape) == 1:
            count = int(np.prod(normalized_shape))
            array, prompt_offset = consume(prompt_noise_values, prompt_offset, count, "prompt")
            prompt_used = True
            return to_tensor(array, normalized_shape, target_device, target_dtype)
        if len(normalized_shape) == 2 and normalized_shape[0] % 2 == 0 and normalized_shape[-1] == acoustic_vae_dim:
            if diffusion_noise_values is None:
                raise RuntimeError("VibeVoice diffusion noise file is required for controlled diffusion randn")
            count = int(np.prod(normalized_shape))
            array, diffusion_offset = consume(diffusion_noise_values, diffusion_offset, count, "diffusion")
            diffusion_used = True
            return to_tensor(array, normalized_shape, target_device, target_dtype)
        return original_randn(*shape_args, **kwargs)

    def randn_like(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal prompt_offset, prompt_used
        shape = tuple(int(dim) for dim in input_tensor.shape)
        if prompt_noise_values is not None and len(shape) == 3 and shape[-1] == acoustic_vae_dim:
            count = int(np.prod(shape))
            array, prompt_offset = consume(prompt_noise_values, prompt_offset, count, "prompt")
            prompt_used = True
            target_device = kwargs.get("device", input_tensor.device)
            target_dtype = kwargs.get("dtype", input_tensor.dtype)
            return to_tensor(array, shape, target_device, target_dtype)
        return original_randn_like(input_tensor, *args, **kwargs)

    torch.randn = randn
    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like
    if prompt_noise_values is not None and not prompt_used:
        raise RuntimeError("VibeVoice controlled prompt noise was not consumed")
    if diffusion_noise_values is not None and not diffusion_used:
        raise RuntimeError("VibeVoice controlled diffusion noise was not consumed")


def run_once(
    model: Any,
    processor: Any,
    request: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, int, float, dict[str, Any]]:
    text = str(request.get("text", ""))
    if not text:
        raise RuntimeError("VibeVoice warmbench request missing text")
    voice_samples = [str(resolve_repo_path(Path(path))) for path in request.get("voice_samples", [])]
    if not voice_samples:
        voice_samples = [str(resolve_repo_path(Path(path))) for path in (args.voice_samples or DEFAULT_VOICES)]
    seed = int(request.get("seed", args.seed))
    ddpm_steps = int(request.get("ddpm_steps", args.ddpm_steps))
    max_new_tokens = int(request.get("max_new_tokens", args.max_new_tokens))
    max_length_times = float(request.get("max_length_times", args.max_length_times))
    cfg_scale = float(request.get("cfg_scale", args.cfg_scale))
    prompt_noise_file = str(request.get("prompt_noise_file", "")) or args.prompt_noise_file
    noise_file = str(request.get("diffusion_noise_file", "")) or args.noise_file
    seed_all(seed, args.backend)
    model.set_ddpm_inference_steps(ddpm_steps)
    inputs = processor(
        text=text,
        voice_samples=voice_samples or None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = move_batch_to_device(dict(inputs), device)
    acoustic_vae_dim = int(model.model.config.acoustic_vae_dim)
    with controlled_generation_noise(prompt_noise_file, noise_file, acoustic_vae_dim):
        started = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens if max_new_tokens > 0 else None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            show_progress_bar=False,
            is_prefill=not args.disable_prefill,
            max_length_times=max_length_times,
        )
    if args.backend == "cuda":
        torch.cuda.synchronize(args.device)
    ended = time.perf_counter()
    audio = extract_audio(outputs)
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(outputs.sequences.shape[-1])
    generated_token_ids = [int(token) for token in outputs.sequences[0, input_tokens:].detach().cpu().tolist()]
    reach_max_step = getattr(outputs, "reach_max_step_sample", None)
    reached_max_step = bool(reach_max_step.detach().cpu().any().item()) if torch.is_tensor(reach_max_step) else False
    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generated_tokens": output_tokens - input_tokens,
        "reached_max_step": reached_max_step,
        "generated_token_ids": token_window(generated_token_ids),
    }
    return audio, 24000, (ended - started) * 1000.0, metrics


def require_same_request_option(requests: list[dict[str, Any]], key: str, default: Any) -> Any:
    values = [request.get(key, default) for request in requests]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"VibeVoice batch warmbench requires identical {key} values")
    return first


def run_batch(
    model: Any,
    processor: Any,
    requests: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> list[tuple[np.ndarray, int, float, dict[str, Any]]]:
    if not requests:
        raise RuntimeError("VibeVoice batch warmbench requires at least one request")
    texts = [str(request.get("text", "")) for request in requests]
    if any(not text for text in texts):
        raise RuntimeError("VibeVoice batch warmbench request missing text")
    fallback_voices = [str(resolve_repo_path(Path(path))) for path in (args.voice_samples or DEFAULT_VOICES)]
    voice_samples = []
    for request in requests:
        samples = [str(resolve_repo_path(Path(path))) for path in request.get("voice_samples", [])]
        voice_samples.append(samples if samples else fallback_voices)

    seed = int(require_same_request_option(requests, "seed", args.seed))
    ddpm_steps = int(require_same_request_option(requests, "ddpm_steps", args.ddpm_steps))
    max_new_tokens = int(require_same_request_option(requests, "max_new_tokens", args.max_new_tokens))
    max_length_times = float(require_same_request_option(requests, "max_length_times", args.max_length_times))
    cfg_scale = float(require_same_request_option(requests, "cfg_scale", args.cfg_scale))
    prompt_noise_file = str(require_same_request_option(requests, "prompt_noise_file", "")) or args.prompt_noise_file
    noise_file = str(require_same_request_option(requests, "diffusion_noise_file", "")) or args.noise_file

    seed_all(seed, args.backend)
    model.set_ddpm_inference_steps(ddpm_steps)
    inputs = processor(
        text=texts,
        voice_samples=voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = move_batch_to_device(dict(inputs), device)
    acoustic_vae_dim = int(model.model.config.acoustic_vae_dim)
    with controlled_generation_noise(prompt_noise_file, noise_file, acoustic_vae_dim):
        started = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens if max_new_tokens > 0 else None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            show_progress_bar=False,
            is_prefill=not args.disable_prefill,
            max_length_times=max_length_times,
        )
    if args.backend == "cuda":
        torch.cuda.synchronize(args.device)
    ended = time.perf_counter()
    wall_ms = (ended - started) * 1000.0
    input_lengths = [int(value) for value in inputs["attention_mask"].sum(dim=-1).detach().cpu().tolist()]
    reach_max_step = getattr(outputs, "reach_max_step_sample", None)

    results: list[tuple[np.ndarray, int, float, dict[str, Any]]] = []
    for index in range(len(requests)):
        audio = extract_audio(outputs, index)
        input_tokens = input_lengths[index]
        output_tokens = int(outputs.sequences.shape[-1])
        generated_count = max(0, output_tokens - input_tokens)
        generated_token_ids = [
            int(token)
            for token in outputs.sequences[index, -generated_count:].detach().cpu().tolist()
        ] if generated_count else []
        reached_max_step = bool(reach_max_step[index].detach().cpu().item()) if torch.is_tensor(reach_max_step) else False
        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generated_tokens": generated_count,
            "reached_max_step": reached_max_step,
            "generated_token_ids": token_window(generated_token_ids),
        }
        results.append((audio, 24000, wall_ms, metrics))
    return results


def main() -> int:
    args = parse_args()
    if args.family != "vibevoice":
        raise RuntimeError(f"unsupported VibeVoice warmbench family: {args.family}")

    model_root = resolve_repo_path(args.model).resolve()
    if not (model_root / "config.json").is_file():
        raise RuntimeError(f"missing VibeVoice model config: {model_root / 'config.json'}")

    VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor, module_path = load_reference_symbols(args.reference_root)

    device = select_device(args)
    dtype = torch.float32 if args.backend == "cpu" else torch_dtype(args.dtype)
    tokenizer_kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.tokenizer_model:
        tokenizer_kwargs["language_model_pretrained_name"] = str(resolve_repo_path(Path(args.tokenizer_model)))
    processor = VibeVoiceProcessor.from_pretrained(str(model_root), **tokenizer_kwargs)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        str(model_root),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=("cuda" if args.backend == "cuda" else "cpu"),
    )
    model.eval()
    patch_transformers_cache_signature(model)
    requests = load_requests(args)
    if not requests:
        raise RuntimeError("VibeVoice warmbench request sequence is empty")
    warmup_request = dict(requests[0])
    if args.warmup_text:
        warmup_request["text"] = args.warmup_text
    for request in requests:
        voice_samples = request.get("voice_samples", args.voice_samples or [str(path) for path in DEFAULT_VOICES])
        if not voice_samples:
            raise RuntimeError("VibeVoice warmbench request requires voice_samples")
        for voice_sample in [str(resolve_repo_path(Path(path))) for path in voice_samples]:
            if not Path(voice_sample).is_file():
                raise RuntimeError(f"missing VibeVoice voice sample: {voice_sample}")
    for voice_sample in warmup_request.get("voice_samples", args.voice_samples or [str(path) for path in DEFAULT_VOICES]):
        voice_sample = str(resolve_repo_path(Path(voice_sample)))
        if not Path(voice_sample).is_file():
            raise RuntimeError(f"missing VibeVoice voice sample: {voice_sample}")
    audio_out_dir = args.output_dir if args.output_dir is not None else args.audio_out_dir

    warmup_outputs: list[tuple[np.ndarray, int, float, dict[str, Any]]] = []
    for _ in range(max(0, args.warmup)):
        warmup_outputs.append(run_once(model, processor, warmup_request, args, device))

    last_outputs: list[tuple[np.ndarray, int, float, dict[str, Any]]] = []
    if args.batch:
        current_batch: list[tuple[np.ndarray, int, float, dict[str, Any]]] | None = None
        for _ in range(max(1, args.iterations)):
            current_batch = run_batch(model, processor, requests, args, device)
        assert current_batch is not None
        last_outputs = current_batch
    else:
        for request in requests:
            current: tuple[np.ndarray, int, float, dict[str, Any]] | None = None
            for _ in range(max(1, args.iterations)):
                current = run_once(model, processor, request, args, device)
            assert current is not None
            last_outputs.append(current)

    for index, (audio, sample_rate, wall_ms, _metrics) in enumerate(warmup_outputs):
        print(f"warmup_text[{index}]={warmup_request['text']}")
        print(f"warmup_summary_json[{index}]=" + summary_json(audio, sample_rate, wall_ms))

    for index, (request, output) in enumerate(zip(requests, last_outputs)):
        audio, sample_rate, wall_ms, _metrics = output
        print(f"text[{index}]={request['text']}")
        print(f"summary_json[{index}]=" + summary_json(audio, sample_rate, wall_ms))

    audio_paths: list[str] = []
    if audio_out_dir is not None:
        audio_out_dir.mkdir(parents=True, exist_ok=True)
        for index, (audio, sample_rate, _wall_ms, _metrics) in enumerate(last_outputs):
            out_path = audio_out_dir / f"request_{index}.wav"
            write_wav(out_path, sample_rate, audio)
            audio_paths.append(str(out_path))
            print(f"audio_out[{index}]={out_path}")
    else:
        audio, sample_rate, _wall_ms, _metrics = last_outputs[-1]
        write_wav(args.audio_out, sample_rate, audio)
        audio_paths.append(str(args.audio_out))
        print(f"audio_out={args.audio_out}")
    steps: list[dict[str, Any]] = []
    for index, (audio, sample_rate, wall_ms, metrics) in enumerate(last_outputs):
        stem: dict[str, Any] = {
            "name": "audio",
            "summary": json.loads(summary_json(audio, sample_rate, wall_ms)),
        }
        if index < len(audio_paths):
            stem["audio"] = audio_paths[index]
        steps.append({
            "request_index": index,
            "stems": [stem],
            "metrics": {"wall_ms": float(wall_ms), **metrics},
        })
    print(
        "summary_json="
        + json.dumps(
            {"family": "vibevoice", "backend": args.backend, "sequence_steps": steps},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
