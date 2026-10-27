#!/usr/bin/env bash
# One explicit phase per invocation. Use tmux/nohup for a long server run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."

phase="${PHASE:-baseline}"
if [[ $# -gt 0 && "$1" != --* ]]; then phase="$1"; shift; fi
case "${phase}" in baseline|teachers|screen|final) ;; *) echo 'PHASE must be baseline, teachers, screen, or final.' >&2; exit 2 ;; esac
: "${DINO_REPO:?Set DINO_REPO to the local official DINOv3 repository}"
: "${DINO_WEIGHTS:?Set DINO_WEIGHTS to the generic DINOv3 ViT-B/16 checkpoint}"
extra=()
case "${DRY_RUN:-0}" in 0) extra+=(--execute) ;; 1) ;; *) echo 'DRY_RUN must be 0 or 1.' >&2; exit 2 ;; esac
case "${RESUME:-0}" in 1) extra+=(--resume) ;; 0) ;; *) echo 'RESUME must be 0 or 1.' >&2; exit 2 ;; esac
case "${DETAIL_BRANCH:-1}" in 1) detail_flag=--detail-branch ;; 0) detail_flag=--no-detail-branch ;; *) echo 'DETAIL_BRANCH must be 0 or 1.' >&2; exit 2 ;; esac

exec "${PYTHON:-python}" -u tools/run_triprd_task_queue.py \
  --phase "${phase}" --output-root "${OUTPUT_ROOT:-results/task_v4}" --gpus "${GPUS:-0,1}" \
  --dino-repo "${DINO_REPO}" --dino-weights "${DINO_WEIGHTS}" \
  --vdt-root "${VDT_ROOT:-datasets/VDT-2048}" --cod-root "${COD_ROOT:-datasets/COD_benchmark}" \
  --seed "${SEED:-20261012}" --split-seed "${SPLIT_SEED:-20261012}" \
  --epochs "${EPOCHS:-100}" --teacher-epochs "${TEACHER_EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-4}" \
  --image-size "${IMAGE_SIZE:-384}" --base-channels "${BASE_CHANNELS:-128}" \
  --train-backbone-blocks "${TRAIN_BACKBONE_BLOCKS:-4}" "${detail_flag}" \
  --augmentation "${AUGMENTATION:-crop_flip}" --source-final-ratio "${SOURCE_FINAL_RATIO:-0.1}" \
  "${extra[@]}" "$@"
