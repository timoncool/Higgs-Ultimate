#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

conda run --no-capture-output -n index-tts python tests/warmbench.py \
    --family ace_step \
    --mode offline \
    --backend "$1" \
    --device 0 \
    --threads 8 \
    --warmup 0 \
    --iterations 1 \
    --requests-per-session 1 \
    --case-name reference_text2music_example_01_600s_temp0 \
    --test-noise-file build/logs/warmbench/manual_noise/ace_step_base_cfg_600s_noise_20260605_alt1.f32 \
    --artifact-stamp ace_step_base_cfg_correct \
    --cpp-session-option ace_step.dit_model_path=acestep-v15-base \
    --cpp-session-option ace_step.lm_model_path=acestep-5Hz-lm-1.7B \
    --cpp-session-option ace_step.trace_enabled=1
