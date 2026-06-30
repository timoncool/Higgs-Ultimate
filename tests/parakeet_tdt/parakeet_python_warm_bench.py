from __future__ import annotations

import argparse
import importlib.util
import json
import time
import types
from contextlib import contextmanager
from pathlib import Path

import librosa


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DRIVER = REPO_ROOT / "reference" / "parakeet-tdt" / "tools" / "test_driver.py"


def load_reference_driver():
    spec = importlib.util.spec_from_file_location("parakeet_test_driver", REFERENCE_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load reference driver from {REFERENCE_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm repeated Python benchmark for the Parakeet TDT reference.")
    parser.add_argument("--model", default="models/parakeet-tdt-0.6b-v3")
    parser.add_argument("--audio", type=Path, default=Path("build/assets/parakeet/2086-149220-0033_5s.wav"))
    parser.add_argument("--warmup-audio", type=Path, default=None)
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-mode", choices=("offline", "longform", "streaming"), default="offline")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-secs", type=float, default=2.0)
    parser.add_argument("--left-context-secs", type=float, default=10.0)
    parser.add_argument("--right-context-secs", type=float, default=2.0)
    parser.add_argument("--timestamps", action="store_true", default=True)
    parser.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    return parser.parse_args()


def parse_audio_sequence(args: argparse.Namespace) -> list[Path]:
    if not args.audio_sequence:
        return [args.audio]
    return [Path(item) for item in args.audio_sequence.split(",") if item]


def synchronize_if_needed(torch_module, backend: str) -> None:
    if backend == "cuda":
        torch_module.cuda.synchronize()


def extract_word_timestamps(timestamp_payload: object) -> list[dict[str, object]]:
    if not isinstance(timestamp_payload, dict):
        return []
    words = timestamp_payload.get("word")
    if not isinstance(words, list):
        return []
    output: list[dict[str, object]] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            word = str(item["word"])
        except Exception:
            continue
        output.append(
            {
                "start_sample": int(round(start * 16000.0)),
                "end_sample": int(round(end * 16000.0)),
                "word": word,
                "confidence": 0.0,
            }
        )
    return output


def make_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): make_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


class SegmentTimer:
    def __init__(self, torch_module, backend: str):
        self._torch = torch_module
        self._backend = backend
        self._active = False
        self._metrics = {
            "frontend_ms": 0.0,
            "pre_encode_ms": 0.0,
            "encoder_total_ms": 0.0,
            "decoder_ms": 0.0,
            "timestamps_ms": 0.0,
        }

    def reset(self) -> None:
        for key in self._metrics:
            self._metrics[key] = 0.0

    def begin(self) -> None:
        self.reset()
        self._active = True

    def end(self) -> dict[str, float]:
        self._active = False
        encoder_ms = max(0.0, self._metrics["encoder_total_ms"] - self._metrics["pre_encode_ms"])
        return {
            "parakeet.frontend_ms": self._metrics["frontend_ms"],
            "parakeet.pre_encode_ms": self._metrics["pre_encode_ms"],
            "parakeet.encoder_ms": encoder_ms,
            "parakeet.decoder_ms": self._metrics["decoder_ms"],
            "parakeet.timestamps_ms": self._metrics["timestamps_ms"],
        }

    @contextmanager
    def measure(self, key: str):
        if not self._active:
            yield
            return
        synchronize_if_needed(self._torch, self._backend)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize_if_needed(self._torch, self._backend)
            ended = time.perf_counter()
            self._metrics[key] += (ended - started) * 1000.0


def install_segment_hooks(asr_model, torch_module, backend: str):
    import nemo.collections.asr.models.rnnt_models as rnnt_models_module

    timer = SegmentTimer(torch_module, backend)

    preprocessor = asr_model.preprocessor
    original_preprocessor_forward = preprocessor.forward

    def preprocessor_forward_wrapped(self, *args, **kwargs):
        with timer.measure("frontend_ms"):
            return original_preprocessor_forward(*args, **kwargs)

    preprocessor.forward = types.MethodType(preprocessor_forward_wrapped, preprocessor)

    encoder = asr_model.encoder
    original_encoder_forward = encoder.forward

    def encoder_forward_wrapped(self, *args, **kwargs):
        with timer.measure("encoder_total_ms"):
            return original_encoder_forward(*args, **kwargs)

    encoder.forward = types.MethodType(encoder_forward_wrapped, encoder)

    pre_encode = getattr(encoder, "pre_encode", None)
    if pre_encode is not None and hasattr(pre_encode, "forward"):
        original_pre_encode_forward = pre_encode.forward

        def pre_encode_forward_wrapped(self, *args, **kwargs):
            with timer.measure("pre_encode_ms"):
                return original_pre_encode_forward(*args, **kwargs)

        pre_encode.forward = types.MethodType(pre_encode_forward_wrapped, pre_encode)

    decoding = asr_model.decoding
    decoding_class = decoding.__class__
    original_decoder_predictions = decoding_class.rnnt_decoder_predictions_tensor

    def decoder_predictions_wrapped(self, *args, **kwargs):
        with timer.measure("decoder_ms"):
            return original_decoder_predictions(self, *args, **kwargs)

    decoding_class.rnnt_decoder_predictions_tensor = decoder_predictions_wrapped

    original_process_timestamp_outputs = rnnt_models_module.process_timestamp_outputs

    def process_timestamp_outputs_wrapped(*args, **kwargs):
        with timer.measure("timestamps_ms"):
            return original_process_timestamp_outputs(*args, **kwargs)

    rnnt_models_module.process_timestamp_outputs = process_timestamp_outputs_wrapped

    return timer


def make_divisible_by(num: int, factor: int) -> int:
    return (num // factor) * factor


def prepare_streaming_model(asr_model, open_dict_module, args: argparse.Namespace):
    from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel
    from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_dtype

    if not isinstance(asr_model, EncDecRNNTModel) and not isinstance(asr_model, EncDecHybridRNNTCTCModel):
        raise ValueError("Streaming warm bench currently supports RNNT / Hybrid RNNT models only")

    decoding_cfg = asr_model.cfg.decoding
    with open_dict_module(decoding_cfg):
        decoding_cfg.tdt_include_token_duration = args.timestamps
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.fused_batch_size = -1
        decoding_cfg.beam.return_best_hypothesis = True

    if isinstance(asr_model, EncDecRNNTModel):
        asr_model.change_decoding_strategy(decoding_cfg)
    elif hasattr(asr_model, "cur_decoder"):
        asr_model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")

    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()
    asr_model.freeze()
    asr_model.to(get_inference_dtype(None, device=asr_model.device))

    model_cfg = asr_model.cfg
    audio_sample_rate = int(model_cfg.preprocessor["sample_rate"])
    feature_stride_sec = float(model_cfg.preprocessor["window_stride"])
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = int(asr_model.encoder.subsampling_factor)
    features_frame2audio_samples = make_divisible_by(
        int(audio_sample_rate * feature_stride_sec),
        factor=encoder_subsampling_factor,
    )
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize

    context_encoder_frames = ContextSize(
        left=int(args.left_context_secs * features_per_sec / encoder_subsampling_factor),
        chunk=int(args.chunk_secs * features_per_sec / encoder_subsampling_factor),
        right=int(args.right_context_secs * features_per_sec / encoder_subsampling_factor),
    )
    context_samples = ContextSize(
        left=context_encoder_frames.left * encoder_subsampling_factor * features_frame2audio_samples,
        chunk=context_encoder_frames.chunk * encoder_subsampling_factor * features_frame2audio_samples,
        right=context_encoder_frames.right * encoder_subsampling_factor * features_frame2audio_samples,
    )

    if asr_model.cfg.encoder.att_context_style == "chunked_limited_with_rc":
        asr_model.encoder.set_default_att_context_size(
            att_context_size=[
                context_encoder_frames.left,
                context_encoder_frames.chunk,
                context_encoder_frames.right,
            ]
        )

    return {
        "audio_sample_rate": audio_sample_rate,
        "encoder_frame2audio_samples": encoder_frame2audio_samples,
        "context_samples": context_samples,
    }


def run_streaming_once_in_process(
    asr_model,
    args: argparse.Namespace,
    torch_module,
    prepared: dict[str, object],
    audio_path: Path,
) -> dict[str, object]:
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    from nemo.collections.asr.parts.utils.streaming_utils import StreamingBatchedAudioBuffer
    from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

    audio_sample_rate = int(prepared["audio_sample_rate"])
    encoder_frame2audio_samples = int(prepared["encoder_frame2audio_samples"])
    context_samples = prepared["context_samples"]

    samples, _ = librosa.load(str(audio_path), sr=audio_sample_rate, mono=True)
    audio_batch = torch_module.from_numpy(samples).unsqueeze(0).to(asr_model.device)
    audio_batch_lengths = torch_module.tensor([audio_batch.shape[1]], dtype=torch_module.long, device=asr_model.device)
    batch_size = int(audio_batch.shape[0])

    decoding_computer = asr_model.decoding.decoding.decoding_computer
    state = None
    current_batched_hyps = None
    left_sample = 0
    right_sample = min(context_samples.chunk + context_samples.right, audio_batch.shape[1])
    buffer = StreamingBatchedAudioBuffer(
        batch_size=batch_size,
        context_samples=context_samples,
        dtype=audio_batch.dtype,
        device=audio_batch.device,
    )
    rest_audio_lengths = audio_batch_lengths.clone()

    while left_sample < audio_batch.shape[1]:
        chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
        is_last_chunk_batch = chunk_length >= rest_audio_lengths
        is_last_chunk = right_sample >= audio_batch.shape[1]
        chunk_lengths_batch = torch_module.where(
            is_last_chunk_batch,
            rest_audio_lengths,
            torch_module.full_like(rest_audio_lengths, fill_value=chunk_length),
        )
        buffer.add_audio_batch_(
            audio_batch[:, left_sample:right_sample],
            audio_lengths=chunk_lengths_batch,
            is_last_chunk=is_last_chunk,
            is_last_chunk_batch=is_last_chunk_batch,
        )

        encoder_output, encoder_output_len = asr_model(
            input_signal=buffer.samples,
            input_signal_length=buffer.context_size_batch.total(),
        )
        encoder_output = encoder_output.transpose(1, 2)

        encoder_context = buffer.context_size.subsample(factor=encoder_frame2audio_samples)
        encoder_context_batch = buffer.context_size_batch.subsample(factor=encoder_frame2audio_samples)
        encoder_output = encoder_output[:, encoder_context.left :]

        chunk_batched_hyps, _, state = decoding_computer(
            x=encoder_output,
            out_len=torch_module.where(
                is_last_chunk_batch,
                encoder_output_len - encoder_context_batch.left,
                encoder_context_batch.chunk,
            ),
            prev_batched_state=state,
            multi_biasing_ids=None,
        )

        if current_batched_hyps is None:
            current_batched_hyps = chunk_batched_hyps
        else:
            current_batched_hyps.merge_(chunk_batched_hyps)

        rest_audio_lengths -= chunk_lengths_batch
        left_sample = right_sample
        right_sample = min(right_sample + context_samples.chunk, audio_batch.shape[1])

    if current_batched_hyps is None:
        raise RuntimeError("streaming decode produced no hypotheses")

    hyp = batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size)[0]
    hyp.text = asr_model.tokenizer.ids_to_text(hyp.y_sequence.tolist())
    timestamp_payload = None
    if args.timestamps:
        hyp = asr_model.decoding.compute_rnnt_timestamps(hyp)
        processed = process_timestamp_outputs(
            hyp,
            subsampling_factor=asr_model.encoder.subsampling_factor,
            window_stride=asr_model.cfg["preprocessor"]["window_stride"],
        )
        if isinstance(processed, list):
            hyp = processed[0]
        else:
            hyp = processed
        timestamp_payload = make_jsonable(getattr(hyp, "timestamp", None))
    return {
        "text": hyp.text,
        "timestamp": timestamp_payload,
    }


def main() -> int:
    args = parse_args()
    audio_sequence = parse_audio_sequence(args)
    driver = load_reference_driver()
    device_label = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"

    driver_args = argparse.Namespace(
        model=args.model,
        audio=args.audio,
        backend=args.backend,
        device=args.device,
        mode="perf",
        run_mode=args.run_mode,
        trace_log=None,
        timing_log=None,
        batch_size=args.batch_size,
        timestamps=args.timestamps,
        timestamp_levels="char,word,segment",
        self_attention_model="rel_pos_local_attn",
        att_context_left=256,
        att_context_right=256,
        chunk_secs=2.0,
        left_context_secs=10.0,
        right_context_secs=2.0,
        clean_groundtruth_text=False,
        langid="en",
    )

    driver.configure_runtime(driver_args)
    nemo_asr, open_dict, torch, _, _ = driver.import_runtime_modules()

    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    runs: list[float] = []
    run_metrics: list[dict[str, float]] = []
    text_output = ""
    word_timestamps: list[dict[str, object]] = []
    sequence_steps: list[dict[str, object]] = []
    if args.run_mode == "streaming":
        asr_model = driver.resolve_model(nemo_asr, driver_args.model)
        device = driver.configure_device(torch, asr_model, driver_args.backend)
        device_label = str(device)
        prepared_streaming = prepare_streaming_model(asr_model, open_dict, args)

        def run_streaming_once():
            synchronize_if_needed(torch, args.backend)
            started = time.perf_counter()
            last_sequence_steps: list[dict[str, object]] = []
            for audio_path in audio_sequence:
                step_started = time.perf_counter()
                result = run_streaming_once_in_process(asr_model, args, torch, prepared_streaming, audio_path)
                step_ended = time.perf_counter()
                step_metrics = {
                    "parakeet.transcribe_wall_ms": (step_ended - step_started) * 1000.0,
                }
                last_sequence_steps.append(
                    {
                        "audio": str(audio_path),
                        "text_output": str(result.get("text", "")),
                        "word_timestamps": extract_word_timestamps(result.get("timestamp")) if args.timestamps else [],
                        "diagnostics": {},
                        "metrics": step_metrics,
                    }
                )
            synchronize_if_needed(torch, args.backend)
            ended = time.perf_counter()
            metrics = {
                "parakeet.transcribe_wall_ms": (ended - started) * 1000.0,
            }
            return last_sequence_steps, metrics

        for _ in range(max(0, args.warmup)):
            run_streaming_once()

        for _ in range(max(1, args.iterations)):
            last_sequence_steps, metrics = run_streaming_once()
            run_metrics.append(metrics)
            runs.append(metrics["parakeet.transcribe_wall_ms"])
            sequence_steps = last_sequence_steps
            if last_sequence_steps:
                text_output = str(last_sequence_steps[-1]["text_output"])
                word_timestamps = list(last_sequence_steps[-1]["word_timestamps"])
    else:
        asr_model = driver.resolve_model(nemo_asr, driver_args.model)
        device = driver.configure_device(torch, asr_model, driver_args.backend)
        device_label = str(device)
        if driver_args.timestamps:
            driver.configure_timestamps(asr_model, open_dict)
        if driver_args.run_mode == "longform":
            driver.configure_longform(asr_model, driver_args)
        segment_timer = install_segment_hooks(asr_model, torch, args.backend)

        warmup_audio = args.warmup_audio if args.warmup_audio is not None else audio_sequence[0]
        for _ in range(max(0, args.warmup)):
            synchronize_if_needed(torch, args.backend)
            _ = asr_model.transcribe(
                [str(warmup_audio)],
                batch_size=args.batch_size,
                return_hypotheses=args.timestamps,
                timestamps=args.timestamps,
            )
            synchronize_if_needed(torch, args.backend)

        for _ in range(max(1, args.iterations)):
            run_total_metrics: dict[str, float] = {}
            started = time.perf_counter()
            last_sequence_steps = []
            for audio_path in audio_sequence:
                synchronize_if_needed(torch, args.backend)
                segment_timer.begin()
                step_started = time.perf_counter()
                hypotheses = asr_model.transcribe(
                    [str(audio_path)],
                    batch_size=args.batch_size,
                    return_hypotheses=args.timestamps,
                    timestamps=args.timestamps,
                )
                synchronize_if_needed(torch, args.backend)
                step_ended = time.perf_counter()
                metrics = segment_timer.end()
                metrics["parakeet.transcribe_wall_ms"] = (step_ended - step_started) * 1000.0
                result = hypotheses[0]
                step_text = driver.extract_text(result)
                step_words = extract_word_timestamps(driver.extract_timestamps(result)) if args.timestamps else []
                last_sequence_steps.append(
                    {
                        "audio": str(audio_path),
                        "text_output": step_text,
                        "word_timestamps": step_words,
                        "diagnostics": {},
                        "metrics": metrics,
                    }
                )
                for key, value in metrics.items():
                    run_total_metrics[key] = run_total_metrics.get(key, 0.0) + value
            ended = time.perf_counter()
            metrics = run_total_metrics
            metrics["parakeet.transcribe_wall_ms"] = (ended - started) * 1000.0
            run_metrics.append(metrics)
            runs.append(metrics["parakeet.transcribe_wall_ms"])
            sequence_steps = last_sequence_steps
            text_output = last_sequence_steps[-1]["text_output"] if last_sequence_steps else ""
            word_timestamps = last_sequence_steps[-1]["word_timestamps"] if last_sequence_steps else []

    average_metrics: dict[str, float] = {}
    if run_metrics:
        for key in run_metrics[0]:
            average_metrics[key] = sum(run[key] for run in run_metrics) / float(len(run_metrics))

    print("family=parakeet_tdt_reference")
    print(f"backend={args.backend}")
    print(f"run_mode={args.run_mode}")
    print(f"device={device_label}")
    print(f"threads={args.threads}")
    print(f"batch_size={args.batch_size}")
    print(f"timestamps={'true' if args.timestamps else 'false'}")
    print(f"warmup={args.warmup}")
    print(f"iterations={args.iterations}")
    print(f"text_output={text_output}")
    print("word_timestamps=" + json.dumps(word_timestamps, separators=(",", ":")))
    if len(sequence_steps) > 1:
        print(f"sequence_steps={len(sequence_steps)}")
        for idx, step in enumerate(sequence_steps):
            print(f"sequence_step[{idx}].audio={step['audio']}")
            print(f"sequence_step[{idx}].text_output={step['text_output']}")
    for idx, metrics in enumerate(run_metrics, start=1):
        print(f"run={idx}")
        for key, value in metrics.items():
            print(f"{key}={value:.6f}")
    print("average")
    for key, value in average_metrics.items():
        print(f"{key}={value:.6f}")
    print(
        "summary_json="
        + json.dumps(
            {
                "family": "parakeet_tdt_reference",
                "backend": args.backend,
                "run_mode": args.run_mode,
                "device": device_label,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "timestamps": args.timestamps,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "warmup_audio": str(warmup_audio) if args.run_mode != "streaming" else "",
                "audio_sequence": [str(path) for path in audio_sequence],
                "text_output": text_output,
                "word_timestamps": word_timestamps,
                "sequence_steps": sequence_steps,
                "runs": run_metrics,
                "average": average_metrics,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
