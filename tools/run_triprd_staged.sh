#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."
phase="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
: "${OUTPUT_ROOT:?Set a new v5 OUTPUT_ROOT}"
: "${TEACHER_ROOT:?Set the existing v4 teachers directory}"
: "${SPLIT_MANIFEST:?Set the existing v4 _data/split_manifest.json}"
: "${DINO_REPO:?Set the official local DINOv3 source directory}"
: "${DINO_WEIGHTS:?Set the generic pretrained DINOv3 weights}"
extra=()
case "${DRY_RUN:-0}" in 0) extra+=(--execute) ;; 1) ;; *) exit 2 ;; esac
case "${RESUME:-0}" in 0) ;; 1) extra+=(--resume) ;; *) exit 2 ;; esac
exec "${PYTHON:-python}" -u tools/run_triprd_staged_queue.py \
  --phase "$phase" --output-root "$OUTPUT_ROOT" \
  --teacher-root "$TEACHER_ROOT" --split-manifest "$SPLIT_MANIFEST" \
  --gpus "${GPUS:-2,3,0,1,6}" \
  --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS" \
  --vdt-root "${VDT_ROOT:-/data2/yyy/TriPRD/datasets/VDT-2048}" \
  --cod-root "${COD_ROOT:-/data2/yyy/TriPRD/datasets/COD_benchmark}" \
  --seed "${SEED:-20261012}" --split-seed "${SPLIT_SEED:-20261012}" \
  --source-epochs "${SOURCE_EPOCHS:-20}" --epochs "${EPOCHS:-100}" \
  --teacher-epochs "${TEACHER_EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-4}" \
  --image-size "${IMAGE_SIZE:-384}" --train-backbone-blocks "${TRAIN_BACKBONE_BLOCKS:-4}" \
  "${extra[@]}" "$@"
