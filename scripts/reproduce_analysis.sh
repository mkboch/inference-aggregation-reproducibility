#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export AGGREGATION_REPRO_ROOT="${AGGREGATION_REPRO_ROOT:-$PWD}"
"$PYTHON" -B scripts/validate_package.py --prepared --runtime-root "$AGGREGATION_REPRO_ROOT"
"$PYTHON" -B scripts/analysis/analyze_final_outcomes.py
"$PYTHON" -B scripts/analysis/run_mechanistic_analysis.py
"$PYTHON" -B scripts/analysis/analyze_sc3_and_revision_evidence.py
