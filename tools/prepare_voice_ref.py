#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def default_output_path(input_path: Path, sample_rate: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_ref_{sample_rate // 1000}k.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert reference voice audio, including WAV-to-WAV normalization, "
            "to the mono WAV format accepted by audiocpp_cli --voice-ref."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input audio files, such as mp3, m4a, flac, or wav.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output WAV path. Only valid when one input file is provided.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for one or more converted WAV files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Output sample rate in Hz. Default: 24000.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path. Default: ffmpeg from PATH.")
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.output is not None and len(args.inputs) != 1:
        raise ValueError("--output can only be used with one input file")
    if args.output is not None and args.out_dir is not None:
        raise ValueError("use either --output or --out-dir, not both")

    jobs: list[tuple[Path, Path]] = []
    for input_path in args.inputs:
        src = input_path.expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"input audio file does not exist: {src}")
        if args.output is not None:
            dst = args.output.expanduser().resolve()
        elif args.out_dir is not None:
            out_dir = args.out_dir.expanduser().resolve()
            dst = out_dir / f"{src.stem}_ref_{args.sample_rate // 1000}k.wav"
        else:
            dst = default_output_path(src, args.sample_rate)
        if src == dst:
            raise ValueError("input and output paths must be different; refusing in-place audio conversion")
        jobs.append((src, dst))
    return jobs


def convert_one(ffmpeg: str, src: Path, dst: Path, sample_rate: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"output exists, pass --overwrite to replace it: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    ffmpeg_path = shutil.which(args.ffmpeg) if Path(args.ffmpeg).name == args.ffmpeg else args.ffmpeg
    if not ffmpeg_path:
        print("error: ffmpeg was not found; install ffmpeg or pass --ffmpeg /path/to/ffmpeg", file=sys.stderr)
        return 2
    try:
        jobs = resolve_outputs(args)
        for src, dst in jobs:
            convert_one(ffmpeg_path, src, dst, args.sample_rate, args.overwrite)
            print(dst)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
