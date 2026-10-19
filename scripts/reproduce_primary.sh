#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export AGGREGATION_REPRO_ROOT="${AGGREGATION_REPRO_ROOT:-$PWD}"
"$PYTHON" -B scripts/validate_package.py --prepared --runtime-root "$AGGREGATION_REPRO_ROOT"
: "${CUDA_VISIBLE_DEVICES:?Set the allocated GPU device(s); runner uses logical cuda:0}"
for model in qwen36 gemma4_31b medgemma27b qwen25 phi4mini; do
    "$PYTHON" -B scripts/primary/run_full_native_benchmark.py --model-key "$model"
done
