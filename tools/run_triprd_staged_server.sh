#!/usr/bin/env bash
# Defaults for the server paths already used by this experiment; overrides remain explicit.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "$PROJECT_ROOT"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/seed-20261012}"
export TEACHER_ROOT="${TEACHER_ROOT:-/data2/yyy/TriPRD_task_v4/results/seed-20261012/teachers}"
export SPLIT_MANIFEST="${SPLIT_MANIFEST:-/data2/yyy/TriPRD_task_v4/results/seed-20261012/_data/split_manifest.json}"
export DINO_REPO="${DINO_REPO:-/data2/yyy/TriPRD/dinov3}"
export DINO_WEIGHTS="${DINO_WEIGHTS:-$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors}"
export VDT_ROOT="${VDT_ROOT:-/data2/yyy/TriPRD/datasets/VDT-2048}"
export COD_ROOT="${COD_ROOT:-/data2/yyy/TriPRD/datasets/COD_benchmark}"
export GPUS="${GPUS:-2,3,0,1,6}"
mkdir -p -- "$OUTPUT_ROOT"
printf 'Staged v5 output: %s\n' "$OUTPUT_ROOT"
exec bash "$SCRIPT_DIR/run_triprd_staged.sh" "$@"
