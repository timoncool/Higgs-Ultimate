#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "VibeVoice-1.5B"
DEFAULT_TEXT = REPO_ROOT / "reference" / "VibeVoiceCommunity" / "demo" / "text_examples" / "4p_climate_100min.txt"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "logs" / "python_vibevoice_100min"
DEFAULT_VOICES = [
    "reference/VibeVoiceCommunity/demo/voices/en-Alice_woman.wav",
    "reference/VibeVoiceCommunity/demo/voices/en-Frank_man.wav",
    "reference/VibeVoiceCommunity/demo/voices/en-Carter_man.wav",
    "reference/VibeVoiceCommunity/demo/voices/en-Maya_woman.wav",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VibeVoice Python 100min reference case.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-file", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--ddpm-steps", type=int, default=20)
    parser.add_argument("--max-length-times", type=float, default=2.0)
    parser.add_argument("--voice-sample", action="append", dest="voice_samples", default=[])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    text_path = resolve(args.text_file)
    text = text_path.read_text(encoding="utf-8")
    voice_samples = args.voice_samples or DEFAULT_VOICES
    request = {
        "id": "vibevoice_python_100min",
        "text": text,
        "voice_samples": voice_samples,
        "max_new_tokens": 0,
        "max_length_times": args.max_length_times,
        "ddpm_steps": args.ddpm_steps,
        "cfg_scale": args.cfg_scale,
        "seed": 1234,
    }

    from vibevoice_python_warm_bench import main as warmbench_main

    sys.argv = [
        "vibevoice_python_warm_bench.py",
        "--family",
        "vibevoice",
        "--model",
        str(resolve(args.model)),
        "--backend",
        args.backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--iterations",
        "1",
        "--request-json",
        json.dumps(request, ensure_ascii=False),
        "--output-dir",
        str(resolve(args.output_dir)),
    ]
    return warmbench_main()


if __name__ == "__main__":
    raise SystemExit(main())
