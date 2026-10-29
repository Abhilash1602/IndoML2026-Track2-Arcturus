#!/bin/bash
# Usage: bash scripts/run_inference.sh <input_audio_dir> <output_dir> [gpu_index]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${3:-0}
python inference/infer.py "$1" "$2"
