#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "$PROJECT_ROOT"
phase="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/completion_controls_v1_seed-20261012}"
export SPLIT_MANIFEST="${SPLIT_MANIFEST:-/data2/yyy/TriPRD_task_v4/results/seed-20261012/_data/split_manifest.json}"
export DINO_REPO="${DINO_REPO:-/data2/yyy/TriPRD/dinov3}"
export DINO_WEIGHTS="${DINO_WEIGHTS:-$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors}"
export VDT_ROOT="${VDT_ROOT:-/data2/yyy/TriPRD/datasets/VDT-2048}"
export COD_ROOT="${COD_ROOT:-/data2/yyy/TriPRD/datasets/COD_benchmark}"
export GPUS="${GPUS:-0,1,2,3}"
extra=()
case "${DRY_RUN:-0}" in 0) extra+=(--execute) ;; 1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;; esac
case "${RESUME:-0}" in 0) ;; 1) extra+=(--resume) ;; *) echo 'RESUME must be 0 or 1' >&2; exit 2 ;; esac
exec "${PYTHON:-python}" -u tools/run_triprd_completion_controls_queue.py \
  --phase "$phase" --output-root "$OUTPUT_ROOT" --gpus "$GPUS" \
  --split-manifest "$SPLIT_MANIFEST" --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS" \
  --vdt-root "$VDT_ROOT" --cod-root "$COD_ROOT" \
  --teacher-epochs "${TEACHER_EPOCHS:-100}" --source-epochs "${SOURCE_EPOCHS:-20}" --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-8}" --micro-batch-size "${MICRO_BATCH_SIZE:-2}" --num-workers "${NUM_WORKERS:-4}" \
  --image-size "${IMAGE_SIZE:-384}" --train-backbone-blocks "${TRAIN_BACKBONE_BLOCKS:-4}" \
  --locator-warmup-epochs "${LOCATOR_WARMUP_EPOCHS:-0}" \
  --warmup-epochs "${WARMUP_EPOCHS:-5}" \
  --train-seed "${SEED:-20261012}" --split-seed "${SPLIT_SEED:-20261012}" \
  --cod-train-mode "${COD_TRAIN_MODE:-full}" "${extra[@]}" "$@"
